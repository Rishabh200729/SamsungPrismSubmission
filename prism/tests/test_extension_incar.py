"""Tests for agent/extension_incar.py — run with the existing unittest/pytest commands."""
import asyncio
import json
import os
import tempfile
import unittest

from prism import TRPState
from agent.extension_incar import LocalAudioBargeDetector, NavRegistry, SilenceTicker, build_stack, is_revision


async def _wait_for(cond, timeout=3.0):
    """Poll instead of sleeping fixed times, so slow machines cannot flake the tests."""
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


class InCarExtensionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.s = build_stack(latency_s=0.02, log_path=os.path.join(self._tmp.name, "log"))

    def tearDown(self):
        self._tmp.cleanup()

    async def _commit_turn(self, destination: str) -> dict:
        """One clean, complete turn ending in a committed update_destination."""
        await self.s.gate.on_transcript_token(f"Navigate to the {destination}.", True)
        return json.loads(await self.s.model_call("update_destination", destination=destination))

    async def _two_committed_routes(self):
        await self._commit_turn("downtown hotel")
        self.s.new_turn()
        await self._commit_turn("airport")
        self.s.barge.set_agent_speaking(True)
        await self.s.barge.on_user_speech_started()      # VAD onset while agent confirms

    # -- supersession ----------------------------------------------------------

    async def test_stale_destination_never_reaches_vehicle(self):
        await self.s.gate.on_transcript_token("Take me to the city mall", False)
        stale = self.s.model_call("update_destination", destination="city mall")
        await asyncio.sleep(0.05)
        await self.s.gate.on_transcript_token("no wait, actually the airport", False)
        fresh = self.s.model_call("update_destination", destination="airport")
        stale_out, fresh_out = await asyncio.gather(stale, fresh)
        self.assertEqual(json.loads(stale_out)["status"], "superseded")
        self.assertEqual(json.loads(fresh_out)["destination"], "airport")
        self.assertEqual([m["to"] for m in self.s.nav.mutations], ["airport"])

    async def test_readonly_query_superseded(self):
        await self.s.gate.on_transcript_token("Find a gas station", False)
        stale = self.s.model_call("find_nearby", category="gas station")
        await asyncio.sleep(0.05)
        await self.s.gate.on_transcript_token("sorry, I mean EV charging", False)
        fresh = self.s.model_call("find_nearby", category="EV charging")
        stale_out, fresh_out = await asyncio.gather(stale, fresh)
        self.assertEqual(json.loads(stale_out)["status"], "superseded")
        self.assertEqual(json.loads(fresh_out)["category"], "ev_charging")

    async def test_get_route_does_not_change_vehicle_state(self):
        await self.s.gate.on_transcript_token("How long to the airport?", True)
        out = json.loads(await self.s.model_call("get_route", destination="the airport"))
        self.assertEqual(out["destination"], "airport")
        self.assertFalse(out["navigation_started"])
        self.assertIsNone(self.s.nav.active_destination)
        self.assertEqual(self.s.nav.mutations, [])

    async def test_reset_resolves_pending_calls_instead_of_hanging(self):
        await self.s.gate.on_transcript_token("Take me to the city mall", False)
        pending = self.s.model_call("update_destination", destination="city mall")
        await asyncio.sleep(0.05)
        self.s.new_turn()                                    # e.g. spurious turn boundary
        out = await asyncio.wait_for(pending, timeout=1.0)   # must not hang
        self.assertEqual(json.loads(out)["status"], "discarded")
        self.assertEqual(self.s.nav.mutations, [])

    async def test_distinct_read_only_calls_without_repair_both_commit(self):
        first = self.s.model_call("find_nearby", category="gas station")
        second = self.s.model_call("find_nearby", category="hospital")
        await self.s.gate.on_transcript_token("Find nearby places.", True)
        first_out, second_out = map(json.loads, await asyncio.gather(first, second))
        self.assertEqual(first_out["category"], "gas_station")
        self.assertEqual(second_out["category"], "hospital")

    async def test_compensation_uses_paired_audit_log(self):
        await self._two_committed_routes()
        await self.s.barge.on_interrupting_speech("Wait, no, go back")
        with open(self.s.dispatcher._log_path, encoding="utf-8") as evaluator:
            records = [json.loads(line) for line in evaluator if line.strip()]
        self.assertTrue(records)
        self.assertTrue(all("call" in record for record in records))
        with open(self.s.dispatcher._audit_path, encoding="utf-8") as audit:
            audit_records = [json.loads(line) for line in audit if line.strip()]
        self.assertTrue(any(record.get("compensation", {}).get("action") == "restore_destination"
                            for record in audit_records))

    # -- intent-aware barge-in -------------------------------------------------

    async def test_barge_in_onset_flushes_audio_but_defers_rollback(self):
        await self._two_committed_routes()
        self.assertEqual(self.s.session.interrupted, 1)
        self.assertEqual(self.s.nav.active_destination, "airport")   # not decided yet
        self.assertTrue(self.s.barge.awaiting_intent)

    async def test_correction_after_barge_in_rolls_back(self):
        await self._two_committed_routes()
        await self.s.barge.on_interrupting_speech("Wait, no, go back")
        self.assertEqual(self.s.nav.active_destination, "downtown hotel")
        self.assertFalse(self.s.barge.awaiting_intent)

    async def test_acknowledgement_after_barge_in_keeps_route(self):
        await self._two_committed_routes()
        await self.s.barge.on_interrupting_speech("Okay, thanks")
        self.assertEqual(self.s.nav.active_destination, "airport")
        self.assertEqual([m["op"] for m in self.s.nav.mutations], ["update_destination"] * 2)
        self.assertFalse(self.s.saga.has_committed_mutations)        # no stale rollback left behind

    async def test_new_command_before_transcript_still_rolls_back_first(self):
        """Realtime models can call the tool before the ASR transcript event arrives."""
        await self._two_committed_routes()
        await self.s.gate.on_transcript_token("Take me to the central station.", True)
        await self.s.model_call("update_destination", destination="central station")
        self.assertEqual([m["to"] for m in self.s.nav.mutations],
                         ["downtown hotel", "airport", "downtown hotel", "central station"])

    async def test_cough_without_words_keeps_route(self):
        await self._two_committed_routes()
        self.s.new_turn()                                    # agent returns to listening, no decision
        self.assertEqual(self.s.nav.active_destination, "airport")
        self.assertFalse(self.s.barge.awaiting_intent)

    async def test_failed_update_compensation_does_not_clear_route(self):
        await self._commit_turn("downtown hotel")
        self.s.new_turn()
        res = await self._commit_turn("")                    # empty -> not_found, nothing changed
        self.assertEqual(res["error"], "not_found")
        self.s.barge.set_agent_speaking(True)
        await self.s.barge.on_user_speech_started()
        await self.s.barge.on_interrupting_speech("Wait, no")   # compensates the failed call
        self.assertEqual(self.s.nav.active_destination, "downtown hotel")

    def test_is_revision(self):
        for text in ("Hold on, take me to the station", "wait no", "cancel that", "go back to the hotel"):
            self.assertTrue(is_revision(text), text)
        for text in ("Okay, thanks", "Great", "no problem", "Got it"):
            self.assertFalse(is_revision(text), text)

    # -- mock geocoder -----------------------------------------------------------

    def test_any_place_resolves_deterministically(self):
        a = NavRegistry(latency_s=0).call("get_route", destination="Delhi airport")
        b = NavRegistry(latency_s=0).call("get_route", destination="the Delhi airport")
        c = NavRegistry(latency_s=0).call("get_route", destination="Mumbai airport")
        self.assertEqual(a["status"], "success")
        self.assertEqual(a["destination"], "delhi airport")
        self.assertEqual(a, b)                               # article ignored, stable across instances
        self.assertNotEqual(a["distance_km"], c["distance_km"])
        self.assertEqual(NavRegistry(latency_s=0).call("get_route", destination="")["error"], "not_found")

    def test_unknown_category_returns_results(self):
        out = NavRegistry(latency_s=0).call("find_nearby", category="pharmacy")
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["results"]), 3)

    def test_local_audio_detector_rms_distinguishes_silence_from_voice(self):
        self.assertEqual(LocalAudioBargeDetector.rms([0, 0, 0]), 0.0)
        self.assertEqual(LocalAudioBargeDetector.rms([-900, 900]), 900.0)

    # -- silence ticker ----------------------------------------------------------

    async def test_silence_ticker_holds_repair_until_900ms_mark(self):
        calls = []
        real = self.s.gate.on_silence_detected

        async def spy(ms):
            calls.append(ms)
            await real(ms)

        self.s.gate.on_silence_detected = spy
        ticker = SilenceTicker(self.s.gate, time_scale=0.05)
        await self.s.gate.on_transcript_token("Take me to the mall", False)
        await self.s.gate.on_transcript_token("oh wait, the airport", False)     # -> REPAIRING
        self.assertEqual(self.s.gate.state, TRPState.REPAIRING)
        self.s.gate._quiescence_task.cancel()                # isolate the ticker's behaviour
        ticker.on_user_stopped()
        await _wait_for(lambda: calls == [300])
        self.assertEqual(self.s.gate.state, TRPState.REPAIRING)                  # 300 ms mark: held
        await _wait_for(lambda: calls == [300, 900])
        self.assertEqual(self.s.gate.state, TRPState.TRP_CONFIRMED)              # 900 ms mark: confirmed

    async def test_silence_ticker_cancelled_when_driver_resumes(self):
        calls = []
        real = self.s.gate.on_silence_detected

        async def spy(ms):
            calls.append(ms)
            await real(ms)

        self.s.gate.on_silence_detected = spy
        ticker = SilenceTicker(self.s.gate)
        await self.s.gate.on_transcript_token("Take me to the airport", False)
        ticker.on_user_stopped()
        await asyncio.sleep(0.05)
        ticker.on_user_started()                             # resumes long before the 300 ms mark
        await asyncio.sleep(0.4)
        self.assertEqual(calls, [])
        self.assertEqual(self.s.gate.state, TRPState.LISTENING)


if __name__ == "__main__":
    unittest.main()
