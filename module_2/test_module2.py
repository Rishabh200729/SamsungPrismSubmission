"""
test_module2.py
Run: python3 -m unittest test_module2 -v

Two tiers:
1. Pure-logic tests (manifest, slot tracker, idempotency key) — run right
   now, no dependency on module3.
2. Integration test against the REAL Module 3 kernel (Runtime + TEST_CONFIG,
   per their Example 4) — auto-skips until module3 is importable, then
   proves your adapter actually behaves correctly end-to-end.
"""

import unittest
from tool_manifest import ToolManifest
from slot_tracker import SlotTracker, UpdateKind
from idempotency import make_idempotency_key

MANIFEST_PAYLOAD = [
    {"name": "book_flight", "side_effect": "STATE_MODIFYING",
     "parameters": {"destination": {"required": True}, "date": {"required": True}}},
    {"name": "book_hotel", "side_effect": "STATE_MODIFYING",
     "parameters": {"city": {"required": True}}},
    {"name": "get_flight_status", "side_effect": "READ_ONLY",
     "parameters": {"flight_id": {"required": True}}},
]


class TestToolManifest(unittest.TestCase):
    def setUp(self):
        self.m = ToolManifest()
        self.m.load_from_event_payload(MANIFEST_PAYLOAD)

    def test_read_only_from_side_effect_none(self):
        self.assertFalse(self.m.is_state_modifying("get_flight_status"))

    def test_state_modifying_from_side_effect(self):
        self.assertTrue(self.m.is_state_modifying("book_flight"))

    def test_side_effect_str_matches_module3_convention(self):
        self.assertEqual(self.m.side_effect_str("book_flight"), "STATE_MODIFYING")
        self.assertEqual(self.m.side_effect_str("get_flight_status"), "READ_ONLY")

    def test_heuristic_fallback_when_no_flag(self):
        m2 = ToolManifest()
        m2.load_from_event_payload([{"name": "cancel_reservation", "parameters": {}}])
        self.assertTrue(m2.is_state_modifying("cancel_reservation"))


class TestSlotTracker(unittest.TestCase):
    def setUp(self):
        self.manifest = ToolManifest()
        self.manifest.load_from_event_payload(MANIFEST_PAYLOAD)
        self.tracker = SlotTracker(self.manifest)

    def test_patch_vs_replace(self):
        self.assertEqual(
            self.tracker.update("book_flight", {"destination": "Goa"}, 1), UpdateKind.NEW)
        self.assertEqual(
            self.tracker.update("book_flight", {"date": "2026-10-01"}, 2), UpdateKind.PATCH)
        self.assertTrue(self.tracker.is_ready())
        self.assertEqual(
            self.tracker.update("book_hotel", {"city": "Goa"}, 3), UpdateKind.REPLACE)
        self.assertNotIn("destination", self.tracker.slots)

    def test_snapshot_shape(self):
        self.tracker.update("book_hotel", {"city": "Goa"}, 1)
        self.tracker.mark_completed("book_hotel")
        snap = self.tracker.snapshot()
        self.assertEqual(snap["intent"], "book_hotel")
        self.assertEqual(snap["slots"]["city"], "Goa")
        self.assertIn("book_hotel", snap["completed_actions"])


class TestIdempotencyKey(unittest.TestCase):
    def test_same_inputs_same_key(self):
        k1 = make_idempotency_key("sess-1", 1, "book_flight", {"destination": "Goa"})
        k2 = make_idempotency_key("sess-1", 1, "book_flight", {"destination": "Goa"})
        self.assertEqual(k1, k2)

    def test_new_generation_changes_key(self):
        k1 = make_idempotency_key("sess-1", 1, "book_flight", {"destination": "Goa"})
        k2 = make_idempotency_key("sess-1", 2, "book_flight", {"destination": "Goa"})
        self.assertNotEqual(k1, k2)

    def test_arg_order_does_not_change_key(self):
        k1 = make_idempotency_key("sess-1", 1, "book_flight", {"a": 1, "b": 2})
        k2 = make_idempotency_key("sess-1", 1, "book_flight", {"b": 2, "a": 1})
        self.assertEqual(k1, k2)


class TestNLU(unittest.TestCase):
    def setUp(self):
        from nlu import RuleBasedExtractor
        self.extractor = RuleBasedExtractor()
        self.manifest = ToolManifest()
        self.manifest.load_from_event_payload(MANIFEST_PAYLOAD)

    def test_extracts_intent_and_slots_from_raw_text(self):
        intent, slots = self.extractor.extract("book a flight to Goa on 2026-10-01", self.manifest)
        self.assertEqual(intent, "book_flight")
        self.assertEqual(slots.get("destination"), "Goa")
        self.assertEqual(slots.get("date"), "2026-10-01")

    def test_no_match_returns_none(self):
        intent, slots = self.extractor.extract("what a nice day", self.manifest)
        self.assertIsNone(intent)

    def test_synonym_and_multiword_place_and_relative_date(self):
        intent, slots = self.extractor.extract("reserve a flight to New York tomorrow", self.manifest)
        self.assertEqual(intent, "book_flight")
        self.assertEqual(slots.get("destination"), "New York")
        self.assertEqual(slots.get("date"), "tomorrow")


try:
    import asyncio
    from module3.runtime import Runtime, TEST_CONFIG
    from module3.runtime.events import make_text_chunk, make_end_of_turn, make_tool_manifest
    from orchestrator import Module2ReasoningAdapter
    HAVE_MODULE3 = True
except ImportError:
    HAVE_MODULE3 = False


@unittest.skipUnless(HAVE_MODULE3, "module3 package not on path yet — clone Rishabh's repo to enable")
class TestIntegrationWithRealRuntime(unittest.TestCase):
    def test_end_to_end_booking(self):
        async def scenario():
            rt = Runtime(config=TEST_CONFIG)
            await rt.start()
            adapter = Module2ReasoningAdapter(rt)
            adapter.install()

            sid = rt.create_session()
            await rt.submit_event(make_tool_manifest(sid, rt.clock.now(), tools=MANIFEST_PAYLOAD))
            eot = make_end_of_turn(sid, rt.clock.now(), utterance_id="u1", final_text="book flight to Goa")
            eot.metadata = {"intent": "book_flight", "extracted_slots": {"destination": "Goa", "date": "2026-10-01"}}
            await rt.submit_event(eot)

            # next_output() actually WAITS for output (unlike drain_outputs,
            # which is non-blocking and can fire before handlers run) —
            # keep polling until nothing new shows up within the timeout.
            outputs = []
            try:
                while True:
                    out = await rt.next_output(timeout_ms=500)
                    outputs.append(out)
            except asyncio.TimeoutError:
                pass

            await rt.close_session(sid)
            await rt.stop()
            return outputs

        outputs = asyncio.run(scenario())
        self.assertTrue(any(o.event_type.value == "TOOL_CALL" for o in outputs))

    def test_stale_result_after_interruption_is_discarded(self):
        """
        The scenario that actually determines your Interruption Recovery
        score: a tool call is in flight, the user interrupts (generation
        bumps), and the OLD call's result finally arrives late. It must be
        discarded, not committed — a wrong commit here means a duplicate
        or stale booking makes it into the final state snapshot.
        """
        async def scenario():
            from module3.runtime.events import make_interruption, make_tool_result

            rt = Runtime(config=TEST_CONFIG)
            await rt.start()
            adapter = Module2ReasoningAdapter(rt)
            adapter.install()

            sid = rt.create_session()
            await rt.submit_event(make_tool_manifest(sid, rt.clock.now(), tools=MANIFEST_PAYLOAD))

            eot = make_end_of_turn(sid, rt.clock.now(), utterance_id="u1", final_text="book flight to Goa")
            eot.metadata = {"intent": "book_flight",
                            "extracted_slots": {"destination": "Goa", "date": "2026-10-01"}}
            await rt.submit_event(eot)

            # Capture the TOOL_CALL to get its call_id before interrupting.
            tool_call_event = await rt.next_output(timeout_ms=500)
            old_call_id = tool_call_event.call_id

            # User interrupts mid-call -> generation bumps, old task fenced.
            await rt.request_cancellation(sid, reason="barge_in")

            # The old call's result straggles in late, after the interrupt.
            stale_result = make_tool_result(
                session_id=sid, timestamp_ms=rt.clock.now(),
                call_id=old_call_id, task_id=tool_call_event.task_id,
                result={"status": "ok"}, success=True,
            )
            await rt.submit_event(stale_result)

            outputs = []
            try:
                while True:
                    outputs.append(await rt.next_output(timeout_ms=300))
            except asyncio.TimeoutError:
                pass

            await rt.close_session(sid)
            await rt.stop()
            return outputs

        outputs = asyncio.run(scenario())
        # The stale result must NOT produce a FINAL_RESPONSE claiming completion.
        self.assertFalse(any(o.event_type.value == "FINAL_RESPONSE" for o in outputs))

    def test_closed_loop_no_manual_metadata(self):
        """
        The real end-to-end path: raw text in (NLU parses it), mock tool
        actually executes, TOOL_RESULT gets submitted back automatically,
        and a FINAL_RESPONSE comes out — nothing hand-fed via metadata.
        This is what proves #1 (the NLU gap) is actually closed, not just
        that the plumbing around it works.
        """
        async def scenario():
            rt = Runtime(config=TEST_CONFIG)
            await rt.start()
            adapter = Module2ReasoningAdapter(rt)
            adapter.install()

            sid = rt.create_session()
            await rt.submit_event(make_tool_manifest(sid, rt.clock.now(), tools=MANIFEST_PAYLOAD))
            eot = make_end_of_turn(sid, rt.clock.now(), utterance_id="u1",
                                    final_text="book a flight to Goa on 2026-10-01")
            await rt.submit_event(eot)  # no manual metadata this time

            outputs = []
            try:
                while True:
                    outputs.append(await rt.next_output(timeout_ms=500))
            except asyncio.TimeoutError:
                pass

            await rt.close_session(sid)
            await rt.stop()
            return outputs

        outputs = asyncio.run(scenario())
        self.assertTrue(any(o.event_type.value == "TOOL_CALL" for o in outputs))
        self.assertTrue(any(o.event_type.value == "FINAL_RESPONSE" for o in outputs))


if __name__ == "__main__":
    unittest.main()
