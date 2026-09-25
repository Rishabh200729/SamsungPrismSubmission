import unittest
import asyncio
from prism import EffectClass, ToolCall, TRPState
from prism.barge_in import BargeInController
from prism.saga_coordinator import SagaCoordinator
from prism.tool_dispatcher import ToolDispatcher
from prism.trp_gate import TRPGate


class MockSession:
    def __init__(self):
        self.interrupted = False

    async def interrupt(self):
        self.interrupted = True


import os
import tempfile

class TestBargeInController(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        self.temp_log_path = self.temp_log.name
        self.temp_log.close()

        self.session = MockSession()
        self.gate = TRPGate()
        self.saga = SagaCoordinator()
        self.dispatcher = ToolDispatcher(
            registry_fn=lambda name, **kw: {"status": "success"},
            saga=self.saga,
            tool_log_path=self.temp_log_path,
            room_name="test-room",
        )
        self.gate.add_state_listener(self.dispatcher.on_trp_state_change)
        self.controller = BargeInController(
            session=self.session,
            trp_gate=self.gate,
            dispatcher=self.dispatcher,
            saga=self.saga,
        )

    def tearDown(self):
        if os.path.exists(self.temp_log_path):
            os.remove(self.temp_log_path)

    async def test_no_barge_in_when_agent_not_speaking(self):
        """User speech onset while agent is listening must not trigger barge-in."""
        self.controller.set_agent_speaking(False)
        await self.controller.on_user_speech_started()
        self.assertFalse(self.session.interrupted)
        self.assertNotEqual(self.gate.state, TRPState.ABORTED)

    async def test_barge_in_flushes_audio_aborts_gate_and_compensates(self):
        """When agent is speaking, user speech onset triggers immediate interrupt, aborts gate, and runs saga."""
        # 1. Simulate committed mutation earlier in turn
        call = ToolCall(
            name="book_flight",
            args={"passenger_name": "Carol"},
            effect_class=EffectClass.COMPENSABLE,
        )
        task = asyncio.create_task(self.dispatcher.dispatch(call))
        await asyncio.sleep(0.01)
        await self.dispatcher.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await task
        self.assertTrue(self.saga.has_committed_mutations)

        # 2. Agent begins speaking confirmation
        self.controller.set_agent_speaking(True)

        # 3. User barges in with correction
        await self.controller.on_user_speech_started()

        # Session audio interrupted
        self.assertTrue(self.session.interrupted)
        # Saga compensated
        self.assertFalse(self.saga.has_committed_mutations)
        # State reset for new turn
        self.assertEqual(self.gate.state, TRPState.LISTENING)


if __name__ == "__main__":
    unittest.main()
