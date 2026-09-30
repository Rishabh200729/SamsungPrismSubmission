import unittest
import asyncio
import json
import os
import tempfile
from prism import EffectClass, ToolCall, TRPState
from prism.barge_in import BargeInController
from prism.saga_coordinator import SagaCoordinator
from prism.tool_dispatcher import ToolDispatcher, EFFECT_MAP
from prism.trp_gate import TRPGate


class MockSession:
    def __init__(self):
        self.interrupted = False

    async def interrupt(self, **kwargs):
        self.interrupted = True


class TestPrismIntegration(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        self.temp_log_path = self.temp_log.name
        self.temp_log.close()

        self.db = {
            "flights": [],
            "filters": {},
            "identity": {},
        }
        self.call_log = []

        def mock_call(name: str, **kwargs):
            self.call_log.append((name, kwargs))
            if name == "search_flights":
                return {"status": "success", "flights": [{"flight_id": "FL100", **kwargs}]}
            elif name == "book_flight":
                booking = {"status": "success", "booking_ref": "B123", "passenger": kwargs.get("passenger_name")}
                self.db["flights"].append(booking)
                return booking
            elif name == "update_search_filter":
                self.db["filters"][kwargs.get("filter_name")] = kwargs.get("value")
                return {"status": "success", "filter_updated": kwargs.get("filter_name")}
            elif name == "update_identity_doc":
                self.db["identity"] = kwargs
                return {"status": "success", "updated_doc": kwargs.get("doc_type")}
            return {"status": "success"}

        self.session = MockSession()
        self.gate = TRPGate()
        self.saga = SagaCoordinator()
        self.dispatcher = ToolDispatcher(
            registry_fn=mock_call,
            saga=self.saga,
            tool_log_path=self.temp_log_path,
            room_name="integration-room",
        )
        self.gate.add_state_listener(self.dispatcher.on_trp_state_change)
        self.barge = BargeInController(
            session=self.session,
            trp_gate=self.gate,
            dispatcher=self.dispatcher,
            saga=self.saga,
        )

    def tearDown(self):
        if os.path.exists(self.temp_log_path):
            os.remove(self.temp_log_path)

    async def test_clean_speech_turn_execution(self):
        """Clean turn with no repairs confirms at standard VAD silence (300ms)."""
        await self.gate.on_transcript_token("Search flights to Seattle on December 1st.", is_final=True)

        call = ToolCall(
            name="search_flights",
            args={"destination": "Seattle", "date": "Dec 1"},
            effect_class=EFFECT_MAP["search_flights"],
        )
        dispatch_task = asyncio.create_task(self.dispatcher.dispatch(call))
        await asyncio.sleep(0.01)

        # Silence detected ≥ 300ms triggers TRP_CONFIRMED
        await self.gate.on_silence_detected(350)
        res_str = await dispatch_task
        res = json.loads(res_str)

        self.assertEqual(res["status"], "success")
        self.assertEqual(self.gate.state, TRPState.TRP_CONFIRMED)

        # Check telemetry file
        with open(self.temp_log_path, "r") as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 1)

    async def test_disfluency_repair_self_correction(self):
        """Disfluency 'no wait actually' inflates VAD to 900ms and supersedes speculative call."""
        await self.gate.on_transcript_token("Search flights to Boston on October 5th", is_final=False)

        call1 = ToolCall(
            name="search_flights",
            args={"destination": "Boston", "date": "Oct 5"},
            effect_class=EFFECT_MAP["search_flights"],
        )
        task1 = asyncio.create_task(self.dispatcher.dispatch(call1))
        await asyncio.sleep(0.01)

        # User repairs: 'no wait actually make that October 7th'
        await self.gate.on_transcript_token("no wait actually make that October 7th.", is_final=True)
        self.assertEqual(self.gate.state, TRPState.REPAIRING)

        # 400ms pause during repair must NOT trigger confirmation
        await self.gate.on_silence_detected(400)
        self.assertEqual(self.gate.state, TRPState.REPAIRING)

        # Corrected tool emitted
        call2 = ToolCall(
            name="search_flights",
            args={"destination": "Boston", "date": "Oct 7"},
            effect_class=EFFECT_MAP["search_flights"],
        )
        task2 = asyncio.create_task(self.dispatcher.dispatch(call2))
        await asyncio.sleep(0.01)

        # Earlier call superseded
        res1 = json.loads(await task1)
        self.assertEqual(res1.get("status"), "superseded")

        # Full repair silence reached (950ms)
        await self.gate.on_silence_detected(950)
        res2 = json.loads(await task2)
        self.assertEqual(res2["status"], "success")
        self.assertEqual(res2["flights"][0]["date"], "Oct 7")

        # Verify telemetry: exactly 1 entry, Oct 7 only
        with open(self.temp_log_path, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["call"]["args"]["date"], "Oct 7")

    async def test_barge_in_and_saga_rollback_flow(self):
        """User barges in while agent is speaking; triggers session interrupt and saga compensation."""
        # Step 1: User completes turn to book flight
        await self.gate.on_transcript_token("Book a flight for passenger David.", is_final=True)
        call = ToolCall(
            name="book_flight",
            args={"passenger_name": "David"},
            effect_class=EFFECT_MAP["book_flight"],
        )
        task = asyncio.create_task(self.dispatcher.dispatch(call))
        await self.gate.on_silence_detected(350)
        await task

        self.assertEqual(len(self.db["flights"]), 1)
        self.assertTrue(self.saga.has_committed_mutations)

        # Step 2: Agent starts speaking confirmation
        self.barge.set_agent_speaking(True)

        # Step 3: User barges in: 'Wait, stop!'
        await self.barge.on_user_speech_started()

        # Step 4: Verification
        self.assertTrue(self.session.interrupted)
        self.assertFalse(self.saga.has_committed_mutations)
        self.assertEqual(self.gate.state, TRPState.LISTENING)

        # Evaluator telemetry contains tool calls only; compensation is audit-only.
        with open(self.temp_log_path, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(records), 1)
        self.assertIn("call", records[0])
        with open(self.temp_log_path + ".audit", "r") as f:
            audit_records = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(audit_records[-1]["compensation"]["action"], "cancel_flight")


if __name__ == "__main__":
    unittest.main()
