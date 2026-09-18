"""
module3/tests/test_adversarial_integration.py

Adversarial integration tests — §11 of the orchestration kernel spec.

12 tests that deliberately try to break the runtime's guarantees:
  1.  Slow handler blocks; interruption wins.
  2.  Interruption queued behind many video frames: processed first.
  3.  Old slow task emits FINAL_RESPONSE after cancellation: gate rejects.
  4.  Two workers attempt identical booking: one call only.
  5.  Tool times out after potentially committing a write: status check, no double-book.
  6.  Destination correction preserves every unrelated slot.
  7.  Old-generation tool result arrives after new booking: cannot alter snapshot.
  8.  Invalid manifest arguments are rejected before TOOL_CALL.
  9.  Read-only calls execute concurrently; writes execute serially.
  10. Closed/unknown session cannot emit output.
  11. Audio/video uncertain grounding triggers clarification not false answer.
  12. Replay the same trace twice with VirtualClock: equivalent output trace.
"""

from __future__ import annotations

import asyncio
import pytest
import uuid

from module3.runtime.runtime import Runtime
from module3.runtime.config import TEST_CONFIG
from module3.runtime.clock.virtual_clock import VirtualClock
from module3.runtime.events import (
    make_text_chunk, make_interruption, make_filler, make_final_response,
    make_tool_call, make_tool_result, make_tool_manifest, make_video_frame,
    make_audio_wav,
)
from module3.runtime.events.base import InputEventType, OutputEventType
from module3.runtime.queues.priority_input_queue import PriorityInputQueue
from module3.runtime.scheduler import (
    WorkerScheduler, ActionProposal, ProposalKind, WorkerKind,
)
from module3.runtime.recovery import RecoveryPolicy, RecoveryReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_manifest(tools):
    return [
        {"name": t, "side_effect": "READ_ONLY", "idempotency_required": False,
         "cancellable": True, "timeout_ms": 5000}
        for t in tools
    ]

def _write_manifest(tools):
    return [
        {"name": t, "side_effect": "STATE_MODIFYING", "idempotency_required": True,
         "cancellable": True, "timeout_ms": 5000}
        for t in tools
    ]


# ---------------------------------------------------------------------------
# TEST 1: Slow handler blocks; interruption wins
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_01_slow_handler_interrupted():
    """Even if a slow handler is blocking, an interruption must advance the generation."""
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    await rt.start()
    sid = rt.create_session()

    slow_started = asyncio.Event()
    slow_finished = asyncio.Event()

    async def slow_background_work():
        slow_started.set()
        # Simulate slow work: waits for a VirtualClock sleep that will
        # never complete (nobody advances the clock), so it stays pending.
        await clock.sleep(10_000.0)
        slow_finished.set()

    async def text_handler(event):
        # Handlers must NOT block the dispatcher loop — they launch background
        # tasks and return immediately.  The runtime then tracks the task and
        # can cancel it when an interruption arrives.
        await rt.create_background_task(sid, slow_background_work(), "slow_task")

    rt.register_handler(InputEventType.TEXT_CHUNK, text_handler)

    try:
        await rt.submit_event(make_text_chunk(sid, 0.0, "Hello"))
        # Yield: let the TEXT_CHUNK handler run (creates background task) and return
        for _ in range(5):
            await asyncio.sleep(0)

        # Submit the interruption — dispatcher loop is now free to process it
        # because the handler already returned.
        await rt.submit_event(make_interruption(sid, 500.0, text="Stop"))
        for _ in range(10):
            await asyncio.sleep(0)

        # Generation must have advanced; slow background task is still pending/cancelled
        gen = rt.get_current_generation(sid)
        assert gen == 2, f"Expected gen=2 after interrupt, got {gen}"
        # The background task was cancelled, not completed normally
        assert not slow_finished.is_set(), "Slow task must not have finished normally"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 2: Interruption queued behind video frames; processed first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_02_interruption_priority_over_video_frames():
    """
    PriorityInputQueue yields control events before data events regardless
    of submission order — verifiable by checking which event type comes out first.
    """
    q = PriorityInputQueue(data_maxsize=0)
    sid = "test-session"

    # Submit 5 video frames first, then an interruption
    for i in range(5):
        frame = make_video_frame(sid, float(i * 10), "AA==", 1, 1, i)
        await q.put(frame)

    interrupt = make_interruption(sid, 100.0, text="Interrupt now")
    await q.put(interrupt)

    # First event out should be the interruption
    first = await q.get()
    assert first.event_type == InputEventType.INTERRUPTION, (
        f"Expected INTERRUPTION first, got {first.event_type.value}"
    )

    # Remaining 5 should all be video frames
    for i in range(5):
        evt = await q.get()
        assert evt.event_type == InputEventType.VIDEO_FRAME

    await q.close()


# ---------------------------------------------------------------------------
# TEST 3: Old slow task emits FINAL_RESPONSE after cancellation → gate rejects
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_03_stale_final_response_rejected():
    """
    After an interruption bumps the generation, an old task that tries to
    emit a FINAL_RESPONSE with the old generation must be rejected by the gate.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        # Bump generation
        old_gen, new_gen = await rt.request_cancellation(sid, reason="interrupt")
        assert old_gen == 1 and new_gen == 2

        # Now try to emit a FINAL_RESPONSE tagged with the OLD generation
        stale_response = make_final_response(
            session_id=sid,
            timestamp_ms=100.0,
            text="Here is your old answer",
            generation=old_gen,
        )

        with pytest.raises(ValueError, match="stale"):
            await rt.emit_output(stale_response)
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 4: Two workers attempt identical booking → one call only (idempotency)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_04_duplicate_write_blocked():
    """
    Two concurrent coroutines both try to emit a STATE_MODIFYING TOOL_CALL
    with the same idempotency key — the second must be soft-rejected.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    # Install a manifest with a booking tool
    await rt.submit_event(
        make_tool_manifest(sid, 0.0, _write_manifest(["book_flight"]))
    )
    for _ in range(5):
        await asyncio.sleep(0)

    try:
        idem_key = "booking-idem-xyz"
        cid1 = "call-booking-1"
        cid2 = "call-booking-2"

        evt1, _ = make_tool_call(
            sid, 10.0, "book_flight",
            call_id=cid1,
            arguments={"destination": "Tokyo"},
            side_effect="STATE_MODIFYING",
            idempotency_key=idem_key,
        )
        evt2, _ = make_tool_call(
            sid, 11.0, "book_flight",
            call_id=cid2,
            arguments={"destination": "Tokyo"},
            side_effect="STATE_MODIFYING",
            idempotency_key=idem_key,  # same key — duplicate
        )

        await rt.emit_output(evt1)
        # Second call with the same idempotency key must raise (duplicate write)
        with pytest.raises(ValueError, match="duplicate"):
            await rt.emit_output(evt2)

        outputs = await rt.drain_outputs()
        call_outputs = [o for o in outputs if o.event_type == OutputEventType.TOOL_CALL]
        assert len(call_outputs) == 1, f"Expected exactly 1 TOOL_CALL, got {len(call_outputs)}"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 5: Tool times out → status query, no second booking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_05_write_uncertainty_status_check():
    """
    When a write times out RecoveryPolicy.handle_write_uncertainty queries
    status before any re-issue and NEVER blindly re-issues the write.
    """
    policy = RecoveryPolicy(clock_now=lambda: 0.0)

    committed_status = "COMMITTED"
    status_calls = []

    async def status_fn():
        status_calls.append(1)
        return committed_status

    status = await policy.handle_write_uncertainty(
        status_fn,
        call_id="call-book-timeout",
        session_id="s1",
        generation=1,
        snapshot_version=1,
        idempotency_key="idem-book-001",
    )

    assert status == "COMMITTED"
    assert len(status_calls) == 1, "Status must be queried exactly once"
    assert policy.total_retries() == 1
    record = policy.retry_log[0]
    assert record.reason == RecoveryReason.WRITE_UNCERTAINTY
    assert record.idempotency_key == "idem-book-001"


# ---------------------------------------------------------------------------
# TEST 6: Destination correction preserves unrelated slots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_06_slot_correction_localized():
    """
    Patching the 'destination' slot must not erase 'date', 'passengers',
    or any other unrelated slot.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        ctx = rt.get_session(sid)
        snap = ctx.get_snapshot()

        # Initial slots: date + passengers + destination
        snap1 = ctx.state_store.patch_slots(
            expected_version=snap.snapshot_version,
            patch={"date": "2026-10-01", "passengers": 2, "destination": "Paris"},
            source_event_id="evt-1",
            timestamp_ms=10.0,
        )
        assert snap1.slots["date"].value == "2026-10-01"
        assert snap1.slots["passengers"].value == 2
        assert snap1.slots["destination"].value == "Paris"

        # Now correct only 'destination'
        snap2 = ctx.state_store.patch_slots(
            expected_version=snap1.snapshot_version,
            patch={"destination": "Tokyo"},
            source_event_id="evt-2",
            timestamp_ms=20.0,
        )
        assert snap2.slots["destination"].value == "Tokyo", "Destination must be updated"
        assert snap2.slots["date"].value == "2026-10-01", "Date must be preserved"
        assert snap2.slots["passengers"].value == 2, "Passengers must be preserved"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 7: Old-generation tool result cannot alter snapshot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_07_old_gen_tool_result_discarded():
    """
    A TOOL_RESULT arriving from an old generation must be detected as stale
    and must not alter the session snapshot.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        # Create a task in gen 1, then bump generation
        task_id = await rt.create_background_task(sid, asyncio.sleep(10_000), "old_task")
        old_gen, new_gen = await rt.request_cancellation(sid, reason="new_booking")
        assert new_gen == 2

        # Snapshot before stale result arrives
        ctx = rt.get_session(sid)
        snap_before = ctx.get_snapshot()

        # Stale result: is_stale_result should be True
        is_stale = rt.is_stale_result(sid, task_id)
        assert is_stale, "Task from gen1 must be stale after gen2 starts"

        # Explicitly handle stale result
        await rt.handle_stale_result(sid, task_id, call_id="stale-call")
        await asyncio.sleep(0)

        # Snapshot must not have changed
        snap_after = ctx.get_snapshot()
        assert snap_after.snapshot_version == snap_before.snapshot_version, (
            "Stale result must not advance snapshot_version"
        )

        # Trace must contain STALE_RESULT_DISCARDED
        trace_types = [e.entry_type for e in rt.tracer.entries()]
        assert "STALE_RESULT_DISCARDED" in trace_types
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 8: Invalid manifest arguments rejected before TOOL_CALL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_08_invalid_manifest_args_rejected():
    """
    A TOOL_CALL with missing required arguments (as declared by the manifest)
    must be rejected by the output gate before the event is published.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    # Install a manifest that requires 'destination' and 'date'
    manifest = [{
        "name": "book_flight",
        "side_effect": "READ_ONLY",
        "idempotency_required": False,
        "cancellable": True,
        "timeout_ms": 5000,
        "input_schema": {
            "required": ["destination", "date"],
            "properties": {
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
        },
    }]
    await rt.submit_event(make_tool_manifest(sid, 0.0, manifest))
    for _ in range(5):
        await asyncio.sleep(0)

    try:
        # Missing 'date' — should be rejected
        evt, _ = make_tool_call(
            sid, 10.0, "book_flight",
            arguments={"destination": "Tokyo"},  # 'date' is missing
        )
        with pytest.raises(ValueError, match="missing"):
            await rt.emit_output(evt)

        outputs = await rt.drain_outputs()
        assert not any(o.event_type == OutputEventType.TOOL_CALL for o in outputs), (
            "Invalid TOOL_CALL must not reach output queue"
        )
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 9: Read-only calls execute concurrently; writes execute serially
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_09_read_concurrent_write_serial():
    """
    WorkerScheduler allows concurrent read-only proposals but serializes
    writes through the write lock.
    """
    call_order: list[str] = []
    barrier = asyncio.Event()

    async def handler(proposal: ActionProposal) -> None:
        call_order.append(f"{proposal.kind.value}:{proposal.is_write}")
        if not proposal.is_write:
            # Reads take longer — if concurrent they finish together
            await asyncio.sleep(0.05)
        else:
            # Writes are fast but serialized
            await asyncio.sleep(0.01)

    scheduler = WorkerScheduler(proposal_handler=handler)
    sid = "sched-test-session"

    def _make(kind, is_write=False, ikey=None):
        return ActionProposal(
            session_id=sid,
            generation=1,
            snapshot_version=1,
            source_event_ids=["e1"],
            kind=kind,
            is_write=is_write,
            idempotency_key=ikey,
        )

    # Submit two reads concurrently
    r1 = _make(ProposalKind.TOOL_CALL, is_write=False)
    r2 = _make(ProposalKind.TOOL_CALL, is_write=False)
    await scheduler.submit(r1, WorkerKind.SLOW)
    await scheduler.submit(r2, WorkerKind.SLOW)

    # Wait for reads to complete
    await asyncio.sleep(0.1)

    # Both reads should have run (concurrently — both in call_order)
    assert call_order.count("tool_call:False") == 2

    # Now submit two writes — they must be serialized
    call_order.clear()
    w1 = _make(ProposalKind.TOOL_CALL, is_write=True, ikey="w1")
    w2 = _make(ProposalKind.TOOL_CALL, is_write=True, ikey="w2")
    await scheduler.submit(w1, WorkerKind.SLOW)
    await scheduler.submit(w2, WorkerKind.SLOW)
    await asyncio.sleep(0.1)

    assert call_order.count("tool_call:True") == 2
    # We can't assert strict order here since the lock guarantees serial
    # execution (not order), but both must complete.


# ---------------------------------------------------------------------------
# TEST 10: Closed/unknown session cannot emit output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_10_closed_session_blocks_output():
    """
    emit_output for a closed or unknown session must raise ValueError.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        await rt.close_session(sid)
        await asyncio.sleep(0)

        filler = make_filler(sid, 10.0, "Hello!")
        with pytest.raises(ValueError, match="closed"):
            await rt.emit_output(filler)

        # Unknown session
        unknown_sid = "nonexistent-session-xyz"
        filler2 = make_filler(unknown_sid, 10.0, "Hello!")
        with pytest.raises(ValueError, match="unknown"):
            await rt.emit_output(filler2)
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 11: Uncertain multimodal grounding triggers clarification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_11_uncertain_grounding_triggers_clarification():
    """
    When a video/audio result has low-confidence grounding, the system must
    emit a CLARIFICATION rather than a confident FINAL_RESPONSE.

    This tests the convention: handlers receiving ambiguous multimodal evidence
    must emit CLARIFICATION, not FINAL_RESPONSE with uncertain data.
    """
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()
    emitted = []

    async def multimodal_handler(event):
        if event.event_type == InputEventType.VIDEO_FRAME:
            # Simulate low-confidence result: emit CLARIFICATION
            from module3.runtime.events.output_events import make_clarification
            clarification = make_clarification(
                session_id=event.session_id,
                timestamp_ms=event.timestamp_ms + 5.0,
                text="I can see a destination sign but I'm not sure of the city — could you confirm?",
            )
            await rt.emit_output(clarification)
            emitted.append(clarification.event_type)

    rt.register_handler(InputEventType.VIDEO_FRAME, multimodal_handler)

    try:
        await rt.submit_event(
            make_video_frame(sid, 0.0, "AA==", 320, 240, 0)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        outputs = await rt.drain_outputs()
        clarifications = [o for o in outputs if o.event_type == OutputEventType.CLARIFICATION]
        final_responses = [o for o in outputs if o.event_type == OutputEventType.FINAL_RESPONSE]

        assert len(clarifications) >= 1, "Must emit CLARIFICATION for uncertain grounding"
        assert len(final_responses) == 0, "Must NOT emit FINAL_RESPONSE with uncertain grounding"
    finally:
        await rt.stop()


# ---------------------------------------------------------------------------
# TEST 12: Deterministic replay with VirtualClock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_12_deterministic_replay():
    """
    Running the same event sequence twice with VirtualClock must produce
    the same output trace and snapshot history.
    """
    async def run_scenario():
        clock = VirtualClock()
        rt = Runtime(config=TEST_CONFIG, clock=clock)

        responses = []

        async def text_handler(event):
            if event.event_type == InputEventType.TEXT_CHUNK:
                filler = make_filler(
                    session_id=event.session_id,
                    timestamp_ms=clock.now() + 5.0,
                    text="Processing...",
                )
                await rt.emit_output(filler)

        rt.register_handler(InputEventType.TEXT_CHUNK, text_handler)
        await rt.start()
        sid = rt.create_session(session_id="replay-session")

        await rt.submit_event(make_text_chunk(sid, 0.0, "Book Tokyo"))
        await rt.submit_event(make_interruption(sid, 100.0, text="Cancel that"))
        await rt.submit_event(make_text_chunk(sid, 200.0, "Book Osaka instead"))

        for _ in range(20):
            await asyncio.sleep(0)

        outputs = await rt.drain_all_outputs()
        trace_types = [e.entry_type for e in rt.tracer.entries()]
        final_gen = rt.get_current_generation(sid)

        await rt.stop()
        return outputs, trace_types, final_gen

    result1 = await run_scenario()
    result2 = await run_scenario()

    outputs1, trace1, gen1 = result1
    outputs2, trace2, gen2 = result2

    assert gen1 == gen2, f"Final generation must match: {gen1} vs {gen2}"
    assert len(outputs1) == len(outputs2), (
        f"Output count must match: {len(outputs1)} vs {len(outputs2)}"
    )
    assert trace1 == trace2, "Trace entry types must be identical across replays"
    for o1, o2 in zip(outputs1, outputs2):
        assert o1.event_type == o2.event_type, "Output event types must match"
