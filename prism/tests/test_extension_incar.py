"""Tests for agent/extension_incar.py — run with the existing unittest/pytest commands."""
import asyncio
import json
import os
import tempfile
import unittest

from prism import TRPState
from agent.extension_incar import SilenceTicker, build_stack


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

    async def test_barge_in_restores_previous_route(self):
        await self._commit_turn("downtown hotel")
        self.s.new_turn()
        await self._commit_turn("airport")
        self.s.barge.set_agent_speaking(True)
        await self.s.barge.on_user_speech_started()
        self.assertEqual(self.s.nav.active_destination, "downtown hotel")
        self.assertEqual(self.s.session.interrupted, 1)

    async def test_failed_update_compensation_does_not_clear_route(self):
        await self._commit_turn("downtown hotel")
        self.s.new_turn()
        res = await self._commit_turn("atlantis")           # not_found, nothing changed
        self.assertEqual(res["error"], "not_found")
        self.s.barge.set_agent_speaking(True)
        await self.s.barge.on_user_speech_started()          # compensates the failed call
        self.assertEqual(self.s.nav.active_destination, "downtown hotel")

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
        self.assertEqual(json.loads(out)["status"], "aborted")
        self.assertEqual(self.s.nav.mutations, [])

    async def test_silence_ticker_holds_repair_until_900ms(self):
        ticker = SilenceTicker(self.s.gate)
        await self.s.gate.on_transcript_token("Take me to the mall", False)
        await self.s.gate.on_transcript_token("oh wait, the airport", False)   # -> REPAIRING
        self.assertEqual(self.s.gate.state, TRPState.REPAIRING)
        self.s.gate._quiescence_task.cancel()                # isolate the ticker's behaviour
        ticker.on_user_stopped()
        await asyncio.sleep(0.45)
        self.assertEqual(self.s.gate.state, TRPState.REPAIRING)      # 300 ms mark: held
        await asyncio.sleep(0.6)
        self.assertEqual(self.s.gate.state, TRPState.TRP_CONFIRMED)  # 900 ms mark: confirmed

    async def test_silence_ticker_cancelled_when_driver_resumes(self):
        ticker = SilenceTicker(self.s.gate)
        await self.s.gate.on_transcript_token("Take me to the airport", False)
        ticker.on_user_stopped()
        await asyncio.sleep(0.1)
        ticker.on_user_started()                             # driver resumes before 300 ms
        await asyncio.sleep(0.4)
        self.assertEqual(self.s.gate.state, TRPState.LISTENING)


if __name__ == "__main__":
    unittest.main()
