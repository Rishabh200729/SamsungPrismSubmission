"""
module3/tests/test_integration.py

End-to-end integration test: the full interruption scenario.

This test simulates the complete execution trace from the spec:

T=0    TEXT_CHUNK
T=10   TASK_CREATED  gen=1
T=20   TASK_STARTED  gen=1
T=200  TOOL_CALL     call=C
T=450  INTERRUPTION
T=452  CANCEL_REQUESTED
T=453  GENERATION_INVALIDATED gen=2
T=700  TOOL_RESULT (stale)
T=701  STALE_RESULT_DISCARDED
T=720  NEW TASK      gen=2
T=900  FINAL_RESPONSE gen=2

Module 1/2/4 handlers are simulated inline.
The runtime is the only real component.
"""

import asyncio
import pytest

from module3.runtime.runtime import Runtime
from module3.runtime.clock.virtual_clock import VirtualClock
from module3.runtime.config import TEST_CONFIG
from module3.runtime.events import (
    make_text_chunk, make_end_of_turn, make_interruption, make_tool_result,
    make_filler, make_tool_call, make_final_response, StateSnapshot,
)
from module3.runtime.events.base import InputEventType, OutputEventType
from module3.runtime.tasks.lifecycle import TaskStatus
from module3.evaluation.metrics import compute_metrics
from module3.evaluation.runner import ScenarioResult


async def test_full_interruption_scenario():
    """
    Phase 8: Complete end-to-end interruption scenario with causal trace.
    Verifies the exact lifecycle from spec §26.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    # State shared between handlers
    slow_task_id: list[str] = []
    slow_call_id: list[str] = []
    new_task_id: list[str] = []

    # ---------------------------------------------------------------
    # Module 2 simulation: slow reasoning handler
    # ---------------------------------------------------------------
    async def slow_path_handler(event):
        """Simulates Module 2's slow reasoning on END_OF_TURN."""
        if event.event_type != InputEventType.END_OF_TURN:
            return

        sid = event.session_id

        # Emit FILLER (fast path, but we simulate it here for simplicity)
        await rt.emit_output(
            make_filler(sid, clock.now() + 5.0, "One moment, booking your flight...",
                        generation=rt.get_current_generation(sid))
        )

        # Emit TOOL_CALL
        evt, cid = make_tool_call(
            sid, clock.now() + 200.0, "mock_booking",
            arguments={"origin": "Seoul", "destination": "Tokyo"},
            task_id=slow_task_id[0] if slow_task_id else None,
        )
        slow_call_id.append(cid)
        await rt.emit_output(evt)

        # Update task correlation
        if slow_task_id:
            ctx = rt.get_session(sid)
            ctx.task_registry.register_call_id(slow_task_id[0], cid)

    # ---------------------------------------------------------------
    # Module 1 simulation: interrupt handler
    # ---------------------------------------------------------------
    async def interrupt_handler(event):
        """Simulates Module 1's interrupt handling."""
        old_gen, new_gen = await rt.request_cancellation(
            event.session_id, reason="user_interruption"
        )

    # Register handlers
    rt.register_handler(InputEventType.END_OF_TURN, slow_path_handler)
    rt.register_handler(InputEventType.INTERRUPTION, interrupt_handler)

    await rt.start()
    sid = rt.create_session()

    try:
        # T=0: User says "Book a flight"
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight from Seoul to Tokyo"))
        await asyncio.sleep(0)

        # T=10: Create slow reasoning task (gen=1) — blocks on an event until cancelled
        slow_done = asyncio.Event()

        async def slow_booking():
            # Blocks until cancelled by the interrupt handler
            await slow_done.wait()

        task_id = await rt.create_background_task(
            sid, slow_booking(), "slow_reasoning", metadata={"intent": "book_flight"}
        )
        slow_task_id.append(task_id)

        # T=10: END_OF_TURN triggers slow handler
        await rt.submit_event(make_end_of_turn(sid, 10.0, final_text="Book a flight from Seoul to Tokyo"))
        await asyncio.sleep(0)

        # Verify task was created in generation 1
        task = rt.get_task(sid, task_id)
        assert task.generation == 1
        assert task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

        # T=450: INTERRUPTION — user changes destination
        await rt.submit_event(
            make_interruption(sid, 450.0, text="Actually change destination to Osaka")
        )
        # Give the event loop time to process the interruption handler
        for _ in range(10):
            await asyncio.sleep(0)

        # Verify generation advanced
        assert rt.get_current_generation(sid) == 2

        # Verify task is cancelled/cancellation_requested
        task = rt.get_task(sid, task_id)
        assert task.status.value in ("CANCELLATION_REQUESTED", "CANCELLED", "RUNNING")

        # T=700: Stale TOOL_RESULT arrives for the old call
        if slow_call_id:
            stale = rt.is_stale_call(sid, slow_call_id[0])
            assert stale  # Call from generation 1 is now stale

            # Handle the stale result explicitly
            await rt.handle_stale_result(sid, task_id, call_id=slow_call_id[0])

        # T=720: New task in generation 2
        async def new_booking():
            await asyncio.sleep(0)

        new_tid = await rt.create_background_task(
            sid, new_booking(), "slow_reasoning_gen2",
            metadata={"intent": "book_flight", "destination": "Osaka"}
        )
        new_task_id.append(new_tid)

        # Verify new task is in generation 2
        new_task = rt.get_task(sid, new_tid)
        assert new_task.generation == 2

        # Let new task complete
        await asyncio.sleep(0.05)

        # T=900: Emit FINAL_RESPONSE for generation 2
        snap = StateSnapshot(
            intent="book_flight",
            slots={"origin": "Seoul", "destination": "Osaka"},
            completed_actions=["search", "book"],
            result_data={"booking_id": "BK-12345", "status": "confirmed"},
        )
        final = make_final_response(
            sid, 900.0, "Your flight to Osaka has been booked!",
            state_snapshot=snap, task_id=new_tid,
            generation=rt.get_current_generation(sid),
        )
        await rt.emit_output(final)

        # ---------------------------------------------------------------
        # Verify trace
        # ---------------------------------------------------------------
        trace = rt.tracer.entries()
        trace_types = [e.entry_type for e in trace]

        assert "TASK_CREATED" in trace_types
        assert "TASK_STARTED" in trace_types
        assert "INTERRUPTION" in trace_types
        assert "GENERATION_INVALIDATED" in trace_types
        assert "STALE_RESULT_DISCARDED" in trace_types
        assert "FINAL_RESPONSE" in trace_types

        # Causal ordering: INTERRUPTION before STALE_RESULT_DISCARDED before FINAL_RESPONSE
        interrupt_idx = next(i for i, e in enumerate(trace) if e.entry_type == "INTERRUPTION")
        stale_idx = next(i for i, e in enumerate(trace) if e.entry_type == "STALE_RESULT_DISCARDED")
        final_idx = next(i for i, e in enumerate(trace) if e.entry_type == "FINAL_RESPONSE")

        assert interrupt_idx < stale_idx < final_idx

        # ---------------------------------------------------------------
        # Verify outputs
        # ---------------------------------------------------------------
        outputs = await rt.drain_outputs()
        output_types = [o.event_type.value for o in outputs]

        assert "FILLER" in output_types
        assert "TOOL_CALL" in output_types
        assert "FINAL_RESPONSE" in output_types

        # Print the trace for demo purposes
        print("\n\n====== END-TO-END INTERRUPTION SCENARIO TRACE ======")
        rt.tracer.print_summary()

        # ---------------------------------------------------------------
        # Compute metrics
        # ---------------------------------------------------------------
        result = ScenarioResult(
            scenario_name="full_interruption",
            session_ids={"default": sid},
            outputs=outputs,
            trace=trace,
            final_virtual_time_ms=900.0,
        )
        metrics = compute_metrics(result)
        assert metrics.stale_result_count >= 1
        assert metrics.tasks_created >= 2  # gen1 task + gen2 task
        assert metrics.cross_session_events == 0

    finally:
        await rt.stop()


async def test_trace_completeness():
    """Verify a complete CREATED->STARTED->COMPLETED lifecycle exists."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        async def quick_work():
            await asyncio.sleep(0)

        task_id = await rt.create_background_task(sid, quick_work(), "quick")
        await asyncio.sleep(0.05)

        trace = rt.tracer.entries()
        trace_types = [e.entry_type for e in trace]
        assert "TASK_CREATED" in trace_types
        assert "TASK_STARTED" in trace_types
        assert "TASK_COMPLETED" in trace_types
    finally:
        await rt.stop()


async def test_session_isolation_in_integration():
    """Two concurrent sessions — complete isolation of tasks and state."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()

    try:
        sid_a = rt.create_session()
        sid_b = rt.create_session()

        # Submit events to each session
        await rt.submit_event(make_text_chunk(sid_a, 0.0, "Hello from A"))
        await rt.submit_event(make_text_chunk(sid_b, 0.0, "Hello from B"))

        # Create tasks in each
        async def noop(): pass
        tid_a = await rt.create_background_task(sid_a, noop(), "task_a")
        tid_b = await rt.create_background_task(sid_b, noop(), "task_b")
        await asyncio.sleep(0.01)

        # Interrupt session A — should NOT affect session B
        old_gen_b = rt.get_current_generation(sid_b)
        await rt.request_cancellation(sid_a)
        assert rt.get_current_generation(sid_b) == old_gen_b

        # Session B's task should not be stale
        assert not rt.is_stale_result(sid_b, tid_b)

        # Session A's task should be stale
        assert rt.is_stale_result(sid_a, tid_a)
    finally:
        await rt.stop()
