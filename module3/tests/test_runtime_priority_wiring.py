"""
module3/tests/test_runtime_priority_wiring.py

End-to-end tests proving that the live Runtime uses PriorityInputQueue
and that INTERRUPTION events can never sit behind queued normal data events.

These tests exercise the *full* Runtime → Dispatcher pipeline, not just the
queue in isolation.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from module3.runtime import Runtime, RuntimeConfig, TEST_CONFIG
from module3.runtime.events import (
    InputEventType,
    make_interruption,
    make_text_chunk,
    make_video_frame,
    make_audio_wav,
)
from module3.runtime.queues.priority_input_queue import PriorityInputQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _yield_ticks(n: int = 5) -> None:
    """Give the event loop a few ticks so the dispatch loop can process."""
    for _ in range(n):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuntimePriorityWiring:
    """Prove the Runtime uses PriorityInputQueue end-to-end."""

    @pytest.mark.asyncio
    async def test_runtime_input_queue_is_priority_type(self):
        """The runtime's input_queue must be a PriorityInputQueue instance."""
        rt = Runtime(config=TEST_CONFIG)
        assert isinstance(rt.input_queue, PriorityInputQueue)

    @pytest.mark.asyncio
    async def test_interruption_skips_queue_of_data_events(self):
        """
        With N data events already enqueued, an INTERRUPTION submitted last
        must be dispatched FIRST.
        """
        dispatch_order: list[str] = []

        async def recorder(event):
            dispatch_order.append(event.event_type.value)

        rt = Runtime(config=TEST_CONFIG)
        rt.register_handler(InputEventType.TEXT_CHUNK, recorder)
        rt.register_handler(InputEventType.VIDEO_FRAME, recorder)
        rt.register_handler(InputEventType.INTERRUPTION, recorder)

        await rt.start()
        sid = rt.create_session()

        try:
            # Enqueue 5 data events first
            for i in range(5):
                await rt.submit_event(make_text_chunk(sid, float(i), f"msg{i}"))

            # Then enqueue an INTERRUPTION
            await rt.submit_event(make_interruption(sid, 100.0, "urgent"))

            # Wait for all events to be dispatched
            await _yield_ticks(50)
        finally:
            await rt.stop()

        assert len(dispatch_order) == 6, f"Expected 6 dispatched, got {len(dispatch_order)}: {dispatch_order}"
        assert dispatch_order[0] == InputEventType.INTERRUPTION.value, (
            f"INTERRUPTION must be dispatched first, but got: {dispatch_order}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("run", range(20))
    async def test_interruption_cannot_sit_behind_normal_data(self, run: int):
        """
        Stress test (20 runs): random queue depth, INTERRUPTION always wins.
        """
        dispatch_order: list[str] = []

        async def recorder(event):
            dispatch_order.append(event.event_type.value)

        rt = Runtime(config=TEST_CONFIG)
        rt.register_handler(InputEventType.TEXT_CHUNK, recorder)
        rt.register_handler(InputEventType.VIDEO_FRAME, recorder)
        rt.register_handler(InputEventType.INTERRUPTION, recorder)

        await rt.start()
        sid = rt.create_session()

        try:
            n_data = random.randint(1, 10)
            for i in range(n_data):
                if random.random() < 0.5:
                    await rt.submit_event(make_text_chunk(sid, float(i), f"t{i}"))
                else:
                    await rt.submit_event(make_video_frame(sid, float(i), "AA==", 1, 1, i))

            await rt.submit_event(make_interruption(sid, 999.0, "barge-in"))

            await _yield_ticks(50)
        finally:
            await rt.stop()

        assert dispatch_order[0] == InputEventType.INTERRUPTION.value, (
            f"Run {run}: INTERRUPTION must be first, got: {dispatch_order}"
        )

    @pytest.mark.asyncio
    async def test_queue_shutdown_after_interruption(self):
        """
        After stop(), the queue is closed cleanly and pending control events
        are drained before the sentinel. No events are lost.
        """
        dispatched: list[str] = []

        async def recorder(event):
            dispatched.append(event.event_type.value)

        rt = Runtime(config=TEST_CONFIG)
        rt.register_handler(InputEventType.INTERRUPTION, recorder)
        rt.register_handler(InputEventType.TEXT_CHUNK, recorder)

        await rt.start()
        sid = rt.create_session()

        try:
            await rt.submit_event(make_interruption(sid, 0.0, "stop-test"))
            await _yield_ticks(20)
        finally:
            await rt.stop()

        assert InputEventType.INTERRUPTION.value in dispatched

    @pytest.mark.asyncio
    async def test_session_isolation_preserved(self):
        """
        INTERRUPTION in session A does not affect the dispatch ordering
        of session B's events.
        """
        session_a_order: list[str] = []
        session_b_order: list[str] = []

        async def recorder_a(event):
            if event.session_id == "iso-a":
                session_a_order.append(event.event_type.value)

        async def recorder_b(event):
            if event.session_id == "iso-b":
                session_b_order.append(event.event_type.value)

        rt = Runtime(config=TEST_CONFIG)
        rt.register_handler(InputEventType.TEXT_CHUNK, recorder_a)
        rt.register_handler(InputEventType.TEXT_CHUNK, recorder_b)
        rt.register_handler(InputEventType.INTERRUPTION, recorder_a)
        rt.register_handler(InputEventType.INTERRUPTION, recorder_b)

        await rt.start()
        sid_a = rt.create_session(session_id="iso-a")
        sid_b = rt.create_session(session_id="iso-b")

        try:
            # Session B gets data only
            await rt.submit_event(make_text_chunk(sid_b, 0.0, "b1"))
            await rt.submit_event(make_text_chunk(sid_b, 1.0, "b2"))

            # Session A gets data then interruption
            await rt.submit_event(make_text_chunk(sid_a, 0.0, "a1"))
            await rt.submit_event(make_interruption(sid_a, 10.0, "barge"))

            await _yield_ticks(50)
        finally:
            await rt.stop()

        # Session A: INTERRUPTION must come before TEXT_CHUNK
        assert session_a_order[0] == InputEventType.INTERRUPTION.value

        # Session B: all events are TEXT_CHUNK (no interruption leaked)
        assert all(t == InputEventType.TEXT_CHUNK.value for t in session_b_order)
        assert len(session_b_order) == 2

    @pytest.mark.asyncio
    async def test_overflow_tracing_on_bounded_data_lane(self):
        """
        With data_maxsize=2 and drop_oldest_data=True, overflow drops the
        oldest data event and increments dropped_count. Control events are
        never dropped.
        """
        cfg = RuntimeConfig(
            deterministic_mode=True,
            input_queue_maxsize=2,
            input_queue_drop_oldest=True,
            log_level="WARNING",
        )
        rt = Runtime(config=cfg)

        dispatched: list[str] = []

        async def recorder(event):
            dispatched.append(event.event_type.value)

        rt.register_handler(InputEventType.TEXT_CHUNK, recorder)
        rt.register_handler(InputEventType.INTERRUPTION, recorder)

        await rt.start()
        sid = rt.create_session()

        try:
            # Fill data lane to capacity
            await rt.submit_event(make_text_chunk(sid, 0.0, "d1"))
            await rt.submit_event(make_text_chunk(sid, 1.0, "d2"))
            # This should drop oldest data event ("d1")
            await rt.submit_event(make_text_chunk(sid, 2.0, "d3"))

            assert rt.input_queue.dropped_count == 1

            # Control lane never drops
            await rt.submit_event(make_interruption(sid, 10.0, "int1"))
            assert rt.input_queue.dropped_count == 1  # still 1

            await _yield_ticks(50)
        finally:
            await rt.stop()

        # INTERRUPTION was dispatched first despite being submitted last
        assert dispatched[0] == InputEventType.INTERRUPTION.value
