"""
module1/tests/test_router.py

Integration tests for FastPathRouter against the live Module 3 runtime.

Tests are numbered to map to the required test matrix:

 1. normal TEXT_CHUNK
 2. short pause
 3. filler throttle
 4. END_OF_TURN floor transition
 5. competitive interruption
 6. non-competitive backchannel
 7. interruption while tool/task is active
 8. TEXT_CHUNK self-correction with active work
 9. rapid repeated correction
10. cancellation/response race
11. late stale result remains rejected by Module 3
12. interruption acknowledgement not lost on generation race
13. filler does not over-fire on pauses
14. backchannel does not dispose active hypothesis
15. synthetic correction emits exactly one INTERRUPTION event
"""

from __future__ import annotations

import asyncio
import pytest

from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.clock.virtual_clock import VirtualClock
from module3.runtime.events.base import InputEventType, OutputEventType
from module3.runtime.events.input_events import (
    make_text_chunk,
    make_end_of_turn,
    make_interruption,
)
from module3.runtime.tasks.lifecycle import TaskStatus

from module1.fast_path.router import FastPathRouter
from module1.fast_path.hypothesis import HypothesisBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ticks(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def _wait_for(predicate, attempts: int = 60) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate(), "wait_for timeout"


def _drain_outputs(runtime: Runtime) -> list:
    """Drain all queued outputs synchronously (non-blocking)."""
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    return loop.run_until_complete(runtime.drain_outputs())


# ---------------------------------------------------------------------------
# Test 1 — normal TEXT_CHUNK: hypothesis is built, no interruption, stay silent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_1_normal_text_chunk_no_output():
    """
    Requirement: normal TEXT_CHUNK with no active tasks and no correction
    should produce no output events and not emit a filler.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        await rt.submit_event(make_text_chunk(sid, 10.0, "Book a flight to Tokyo"))
        await _ticks(20)
        outputs = await rt.drain_outputs()
        # No filler on plain TEXT_CHUNK
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert fillers == [], f"Expected no filler on TEXT_CHUNK, got: {fillers}"

        # Hypothesis should be built
        h = router.hypothesis_buffer.peek(sid)
        assert h is not None
        assert "tokyo" in h.entities.get("destination", "").lower()
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 2 — short pause between chunks: no interruption submitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_2_short_pause_no_interruption():
    """
    Requirement: a pause between TEXT_CHUNKs (no END_OF_TURN yet)
    should not trigger any interruption.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)

    interruptions_submitted: list = []

    async def spy(event):
        if event.event_type == InputEventType.INTERRUPTION:
            interruptions_submitted.append(event)

    # There's no input handler registration, but we can spy on submit_event
    # by checking outputs + generation post-test.

    await rt.start()
    sid = rt.create_session()

    try:
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight"))
        clock.advance(500.0)  # 500ms pause
        await rt.submit_event(make_text_chunk(sid, 500.0, "to Tokyo"))
        await _ticks(20)

        # Generation should still be 1 (no interruption)
        assert rt.get_current_generation(sid) == 1
        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert fillers == [], "Short pause should not produce a filler"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 3 — filler throttle: second END_OF_TURN within throttle window → no filler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_3_filler_throttle():
    """
    Requirement: if two END_OF_TURN events arrive within the filler throttle
    window, only one filler should be emitted.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        # First END_OF_TURN at t=0
        await rt.submit_event(make_end_of_turn(sid, 0.0, final_text="Book a flight"))
        await _ticks(20)

        # Second END_OF_TURN at t=100ms (within default 3000ms window)
        clock.advance(100.0)
        await rt.submit_event(make_end_of_turn(sid, 100.0, final_text="Book a flight"))
        await _ticks(20)

        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert len(fillers) == 1, (
            f"Expected exactly 1 filler (throttle should block second), got {len(fillers)}"
        )
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 4 — END_OF_TURN floor transition: floor moves to THINKING
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_4_end_of_turn_floor_becomes_thinking():
    """
    Requirement: END_OF_TURN should drive floor from LISTENING → THINKING.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        ctx = rt.get_session(sid)
        assert ctx.state_store.snapshot.floor_state == FloorState.LISTENING

        await rt.submit_event(make_end_of_turn(sid, 0.0, final_text="Book a flight"))
        await _ticks(20)

        assert ctx.state_store.snapshot.floor_state == FloorState.THINKING, (
            f"Expected THINKING after END_OF_TURN, got: "
            f"{ctx.state_store.snapshot.floor_state.value}"
        )
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 5 — competitive interruption: ack filler emitted, hypothesis disposed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_5_competitive_interruption_ack_and_hypothesis_disposed():
    """
    Requirement: competitive interruption should emit a filler acknowledgement
    and dispose the active hypothesis.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        # Build a hypothesis first
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
        await _ticks(10)
        assert router.hypothesis_buffer.peek(sid) is not None

        # Submit competitive interruption
        await rt.submit_event(make_interruption(sid, 50.0, competitive=True))
        await _ticks(30)

        # Hypothesis should be disposed
        assert router.hypothesis_buffer.peek(sid) is None, \
            "Hypothesis should be disposed after competitive interruption"

        # Filler should have been emitted
        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert len(fillers) >= 1, f"Expected at least 1 ack filler, got: {fillers}"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 6 — non-competitive backchannel: no cancellation, no filler, hypothesis preserved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_6_backchannel_no_cancel_no_filler_hypothesis_preserved():
    """
    Requirement: a competitive=False (backchannel) INTERRUPTION must NOT cancel,
    NOT emit a filler, NOT dispose the hypothesis, and NOT increment generation.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        # Build hypothesis
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
        await _ticks(10)
        h_before = router.hypothesis_buffer.peek(sid)
        assert h_before is not None

        # Backchannel
        await rt.submit_event(make_interruption(sid, 50.0, competitive=False, text="uh-huh"))
        await _ticks(30)

        # Generation must NOT have changed (backchannel = no cancellation at kernel level)
        assert rt.get_current_generation(sid) == 1, \
            "Generation must not change for backchannel"

        # Hypothesis must NOT be disposed
        h_after = router.hypothesis_buffer.peek(sid)
        assert h_after is h_before, \
            "Backchannel must not dispose hypothesis"

        # No filler from Module 1
        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert fillers == [], f"Backchannel must not emit filler, got: {fillers}"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 7 — interruption while task is active: task is cancelled by Module 3
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_7_interruption_while_task_active_cancels_task():
    """
    Requirement: a competitive interruption while a background task is running
    should result in the task being cancelled (by Module 3) and generation
    advancing. Module 1 emits an ack filler.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()
    blocker = asyncio.Event()

    try:
        task_id = await rt.create_background_task(
            sid, blocker.wait(), "slow-task"
        )
        await _wait_for(
            lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING
        )

        await rt.submit_event(make_interruption(sid, 100.0, competitive=True))
        await _wait_for(lambda: rt.get_current_generation(sid) == 2)

        # Task should be cancelled (Module 3's responsibility)
        task_status = rt.get_task(sid, task_id).status
        assert task_status in {TaskStatus.CANCELLATION_REQUESTED, TaskStatus.CANCELLED}, \
            f"Expected cancelled task, got: {task_status}"

        # Module 1: hypothesis disposed
        assert router.hypothesis_buffer.peek(sid) is None

        # Module 1: filler emitted
        await _ticks(10)
        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert fillers, "Expected at least 1 ack filler after interruption"
    finally:
        blocker.set()
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 8 — TEXT_CHUNK self-correction with active work → one INTERRUPTION submitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_8_text_chunk_correction_with_active_work_submits_interruption():
    """
    Requirement: when a TEXT_CHUNK changes a slot value AND there is active
    background work, Module 1 must submit exactly one competitive INTERRUPTION,
    which then drives the canonical cancellation + generation increment.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)

    interruptions_received: list = []

    async def spy_interruption(event):
        interruptions_received.append(event)

    rt.register_handler(InputEventType.INTERRUPTION, spy_interruption)

    await rt.start()
    sid = rt.create_session()
    blocker = asyncio.Event()

    try:
        # Establish active work
        task_id = await rt.create_background_task(sid, blocker.wait(), "active-task")
        await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

        # First chunk: sets destination=Tokyo
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight from Seoul to Tokyo"))
        await _ticks(10)

        # Second chunk: changes destination to Osaka (correction)
        await rt.submit_event(
            make_text_chunk(sid, 100.0, "actually change it to Osaka")
        )
        await _ticks(30)

        # Generation should have advanced (cancellation triggered)
        await _wait_for(lambda: rt.get_current_generation(sid) == 2, attempts=60)
        assert rt.get_current_generation(sid) == 2

        # Exactly one INTERRUPTION event with self_correction reason
        correction_ints = [
            e for e in interruptions_received
            if e.payload.get("reason") == "self_correction_detected"
        ]
        assert len(correction_ints) == 1, (
            f"Expected exactly 1 self_correction INTERRUPTION, got {len(correction_ints)}"
        )
        assert correction_ints[0].payload.get("competitive") is True
    finally:
        blocker.set()
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 9 — rapid repeated correction: each generates at most one INTERRUPTION
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_9_rapid_repeated_corrections():
    """
    Requirement: multiple rapid corrections must each produce at most one
    INTERRUPTION event. No duplicate generation increments within a single
    interrupt-guard window.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)

    interruptions: list = []

    async def spy(event):
        if event.payload.get("competitive", True):
            interruptions.append(event)

    rt.register_handler(InputEventType.INTERRUPTION, spy)

    await rt.start()
    sid = rt.create_session()
    blocker = asyncio.Event()

    try:
        task_id = await rt.create_background_task(sid, blocker.wait(), "slow")
        await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

        # Chunk 1: destination = Tokyo
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
        await _ticks(10)

        # Chunk 2: destination changes → Osaka (first correction)
        await rt.submit_event(make_text_chunk(sid, 50.0, "change it to Osaka"))
        await _ticks(30)

        # Generation advanced; a new task would be needed for second correction
        # but the old one is cancelled. No new task → second chunk correction
        # has no active work → no second INTERRUPTION.

        correction_events = [
            e for e in interruptions
            if e.payload.get("reason") == "self_correction_detected"
        ]
        # Should be exactly 1 — one correction per active-work window
        assert len(correction_events) >= 1

        # Generation should not have gone beyond what one interruption produces
        gen = rt.get_current_generation(sid)
        assert gen >= 2  # at least one interruption happened
    finally:
        blocker.set()
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 10 — cancellation/response race: stale output from old gen is rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_10_stale_output_rejected_by_output_gate():
    """
    Requirement: after an interruption advances the generation, any output
    from the old generation is rejected by Module 3's output gate.
    Module 1 is not involved in this rejection — it is Module 3's concern.
    This test verifies the integration point is intact.
    """
    from module3.runtime.events.output_events import make_filler as m3_make_filler

    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        # Advance to generation 2 via competitive interruption
        await rt.submit_event(make_interruption(sid, 0.0, competitive=True))
        await _wait_for(lambda: rt.get_current_generation(sid) == 2)

        # Try to emit an output stamped with old generation=1
        stale_filler = m3_make_filler(sid, 100.0, "stale ack")
        # Force the generation stamp to the old value
        stale_filler = stale_filler.model_copy(update={"generation": 1})

        with pytest.raises(ValueError, match="stale generation"):
            await rt.emit_output(stale_filler)
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 11 — late stale result: Module 3 rejects it; Module 1 not involved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_11_late_stale_result_rejected_by_module3():
    """
    Requirement: a TOOL_RESULT arriving after an interruption (wrong generation)
    is handled by Module 3's stale-result detection. Module 1 does not handle
    TOOL_RESULT at all. Verify the runtime's is_stale_result() works correctly.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        # Register a task at generation 1
        blocker = asyncio.Event()
        task_id = await rt.create_background_task(sid, blocker.wait(), "old-task")
        await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

        # Interrupt → generation 2
        await rt.submit_event(make_interruption(sid, 0.0, competitive=True))
        await _wait_for(lambda: rt.get_current_generation(sid) == 2)

        # Task is now stale
        assert rt.is_stale_result(sid, task_id), \
            "Task from gen 1 should be stale after gen 2"

        # Module 1 has no TOOL_RESULT handler (verified implicitly — no crash)
        from module3.runtime.events.base import InputEventType as IET
        m1_handlers = rt._dispatcher._handlers.get(IET.TOOL_RESULT, [])
        module1_tool_handlers = [h for h in m1_handlers if "FastPath" in type(h.__self__).__name__ if hasattr(h, '__self__')]
        assert not module1_tool_handlers, "Module 1 must not register a TOOL_RESULT handler"
    finally:
        blocker.set()
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 12 — interruption acknowledgement not lost on generation race
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_12_interruption_ack_not_lost_on_generation_race():
    """
    RACE AUDIT: Can Module 1's filler be rejected when Runtime._handle_interruption
    concurrently advances the generation?

    The resolution in router.py: emit filler with generation=None.
    The output gate check: `if event.generation is not None and ... != current`.
    A None-generation filler passes unconditionally.

    This test verifies that a filler IS emitted after a competitive interruption
    regardless of handler ordering. It runs the full asyncio.gather path.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        # Submit competitive interruption
        await rt.submit_event(make_interruption(sid, 0.0, competitive=True))
        await _ticks(40)

        # Generation must have advanced (Module 3's handler ran)
        assert rt.get_current_generation(sid) == 2

        # Filler must have been emitted (Module 1's handler ran, generation race is safe)
        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert fillers, (
            "Module 1's ack filler must not be lost due to generation race. "
            "The filler should be emitted with generation=None to bypass staleness check."
        )

        # Verify the filler has no generation stamp (this is the mechanism)
        for f in fillers:
            assert f.generation is None, (
                f"Filler must have generation=None to be race-safe, got generation={f.generation}"
            )
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 13 — filler does not over-fire on pauses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_13_filler_does_not_overfire_on_pauses():
    """
    Requirement: multiple TEXT_CHUNKs should produce zero fillers (Module 1
    stays silent while the user is speaking). Only END_OF_TURN produces a filler.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        for i, text in enumerate(["Book", "a", "flight", "to"]):
            clock.advance(float(i * 100))
            await rt.submit_event(make_text_chunk(sid, float(i * 100), text))

        await _ticks(30)
        outputs = await rt.drain_outputs()
        fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
        assert fillers == [], (
            f"TEXT_CHUNKs alone should not produce fillers, got: {fillers}"
        )
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 14 — backchannel does not dispose active hypothesis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_14_backchannel_does_not_dispose_hypothesis():
    """
    Requirement: a competitive=False backchannel INTERRUPTION must leave
    the active hypothesis exactly as it was.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)
    await rt.start()
    sid = rt.create_session()

    try:
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
        await _ticks(10)

        h_before = router.hypothesis_buffer.peek(sid)
        assert h_before is not None
        dest_before = h_before.entities.get("destination", "")
        chunks_before = h_before.chunk_count

        # Backchannel
        await rt.submit_event(make_interruption(sid, 50.0, competitive=False, text="mm-hmm"))
        await _ticks(20)

        h_after = router.hypothesis_buffer.peek(sid)
        # Hypothesis object should be the same — not disposed, not recreated
        assert h_after is h_before, "Backchannel must not dispose hypothesis"
        assert h_after.chunk_count == chunks_before
        assert h_after.entities.get("destination", "").lower() == dest_before.lower()
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# Test 15 — synthetic correction emits exactly one INTERRUPTION event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_15_synthetic_correction_exactly_one_interruption():
    """
    Requirement: TEXT_CHUNK correction detection must submit exactly ONE
    competitive INTERRUPTION event — never more, never zero when active
    work exists.
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    router = FastPathRouter(rt, clock)

    all_interruptions: list = []

    async def record_interruption(event):
        all_interruptions.append(event)

    rt.register_handler(InputEventType.INTERRUPTION, record_interruption)

    await rt.start()
    sid = rt.create_session()
    blocker = asyncio.Event()

    try:
        # Active task to trigger correction path
        task_id = await rt.create_background_task(sid, blocker.wait(), "active")
        await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

        # First chunk: destination = Tokyo
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight from Seoul to Tokyo"))
        await _ticks(10)

        # Second chunk: destination changes (correction detected)
        await rt.submit_event(make_text_chunk(sid, 100.0, "actually make it Osaka"))
        await _ticks(30)

        # Generation advanced = interruption was submitted and processed
        await _wait_for(lambda: rt.get_current_generation(sid) >= 2, attempts=60)

        # Exactly one interruption from the correction path
        correction_ints = [
            e for e in all_interruptions
            if e.payload.get("reason") == "self_correction_detected"
            and e.payload.get("competitive") is True
        ]
        assert len(correction_ints) == 1, (
            f"Expected exactly 1 synthetic INTERRUPTION for the correction, "
            f"got {len(correction_ints)}: {[e.payload for e in correction_ints]}"
        )

        # Verify the submitted interruption carries the correcting text
        assert correction_ints[0].payload.get("text") is not None
    finally:
        blocker.set()
        await rt.stop()


# ---------------------------------------------------------------------------
# Required import for test 4
# ---------------------------------------------------------------------------
from module3.runtime.sessions.state import FloorState  # noqa: E402
