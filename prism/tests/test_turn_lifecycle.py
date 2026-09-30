"""Race-focused tests for the main LiveKit turn lifecycle coordinator."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_STUB = Path(__file__).parent / "_livekit_stub"
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_STUB))

from prism import EffectClass, TRPState, ToolCall
from prism.barge_in import BargeInController
from prism.saga_coordinator import SagaCoordinator
from prism.tool_dispatcher import ToolDispatcher
from prism.trp_gate import TRPGate
from agent.prism_agent import LatencyTracker, SilenceTicker, TurnLifecycleCoordinator


class _Session:
    async def interrupt(self, **kwargs):
        return None


class TurnLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        self.tmp.close()
        self.gate = TRPGate()
        self.saga = SagaCoordinator()
        self.dispatcher = ToolDispatcher(
            registry_fn=lambda name, **args: {"status": "success", "name": name, **args},
            saga=self.saga,
            tool_log_path=self.tmp.name,
            correction_epoch_fn=lambda: self.gate.correction_epoch,
        )
        self.gate.add_state_listener(self.dispatcher.on_trp_state_change)
        self.barge = BargeInController(_Session(), self.gate, self.dispatcher, self.saga)
        self.lifecycle = TurnLifecycleCoordinator(
            self.gate, self.dispatcher, self.saga, self.barge, SilenceTicker(self.gate),
        )

    def tearDown(self):
        os.unlink(self.tmp.name)

    async def test_resumed_repair_preserves_staged_call_and_epoch(self):
        pending = asyncio.create_task(self.dispatcher.dispatch(ToolCall(
            "book_flight", {"passenger_name": "A"}, EffectClass.COMPENSABLE,
        )))
        await asyncio.sleep(0)
        await self.gate.on_transcript_token("Book it no wait", False)
        epoch = self.gate.correction_epoch
        await self.lifecycle.user_speech_started()
        self.assertEqual(self.gate.correction_epoch, epoch)
        self.assertEqual(self.gate.state, TRPState.REPAIRING)
        self.assertEqual(len(self.dispatcher._staged_queue), 1)
        pending.cancel()

    async def test_closed_turn_resets_on_next_onset(self):
        await self.gate.on_transcript_token("Search flights to Rome.", True)
        self.assertEqual(self.gate.state, TRPState.TRP_CONFIRMED)
        await self.lifecycle.user_speech_started()
        self.assertEqual(self.gate.state, TRPState.LISTENING)

    async def test_duplicate_onsets_do_not_reset_continued_turn(self):
        await self.gate.on_transcript_token("Search flights", False)
        await asyncio.gather(self.lifecycle.user_speech_started(), self.lifecycle.user_speech_started())
        self.assertEqual(self.gate.state, TRPState.LISTENING)
        self.assertEqual(self.gate._buffer, "Search flights")

    def test_latency_record_uses_fdb_heartbeat_format(self):
        heartbeat = self.tmp.name + ".heartbeat"
        previous = os.environ.get("PRISM_HEARTBEAT_PATH")
        os.environ["PRISM_HEARTBEAT_PATH"] = heartbeat
        try:
            tracker = LatencyTracker()
            tracker.user_done_at = 10.0
            tracker.tool_start_at = 10.2
            tracker.tool_end_at = 10.5
            tracker.agent_start_at = 10.8
            tracker.log_breakdown("search_flights", "eval-room")
            with open(heartbeat, encoding="utf-8") as records:
                record = json.loads(records.read().split(": ", 1)[1])
            self.assertEqual(record["room"], "eval-room")
            self.assertEqual(record["tool_name"], "search_flights")
            self.assertEqual(set(record), {
                "room", "tool_name", "user_done_at", "tool_start_at", "tool_end_at",
                "agent_start_at", "reasoning", "execution", "synthesis", "total",
            })
        finally:
            if previous is None:
                os.environ.pop("PRISM_HEARTBEAT_PATH", None)
            else:
                os.environ["PRISM_HEARTBEAT_PATH"] = previous
            if os.path.exists(heartbeat):
                os.unlink(heartbeat)
