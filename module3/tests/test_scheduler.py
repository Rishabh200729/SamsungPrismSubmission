"""
module3/tests/test_scheduler.py

Unit tests for WorkerScheduler (§7) and RecoveryPolicy (§8).
"""

from __future__ import annotations

import asyncio
import pytest

from module3.runtime.scheduler import (
    WorkerScheduler, ActionProposal, ProposalKind, WorkerKind,
)
from module3.runtime.recovery import RecoveryPolicy, RecoveryReason

SID = "sched-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proposal(kind=ProposalKind.FILLER, is_write=False, ikey=None):
    return ActionProposal(
        session_id=SID,
        generation=1,
        snapshot_version=1,
        source_event_ids=["e1"],
        kind=kind,
        is_write=is_write,
        idempotency_key=ikey,
    )


# ===========================================================================
# WorkerScheduler tests
# ===========================================================================

@pytest.mark.asyncio
async def test_scheduler_fast_worker_calls_handler():
    called = []

    async def handler(p: ActionProposal):
        called.append(p.kind)

    s = WorkerScheduler(handler)
    await s.submit(_proposal(ProposalKind.FILLER), WorkerKind.FAST)
    await asyncio.sleep(0.05)
    assert ProposalKind.FILLER in called


@pytest.mark.asyncio
async def test_scheduler_fast_worker_rejects_writes():
    called = []

    async def handler(p: ActionProposal):
        called.append(p)

    s = WorkerScheduler(handler)
    write_proposal = _proposal(ProposalKind.TOOL_CALL, is_write=True, ikey="w1")
    await s.submit(write_proposal, WorkerKind.FAST)
    await asyncio.sleep(0.05)
    # Fast workers must NOT execute write proposals
    assert len(called) == 0


@pytest.mark.asyncio
async def test_scheduler_read_proposals_run_concurrently():
    """Two slow read proposals must execute concurrently (no serialization)."""
    start_times = []
    end_times = []

    async def handler(p: ActionProposal):
        start_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)
        end_times.append(asyncio.get_event_loop().time())

    s = WorkerScheduler(handler)
    r1 = _proposal(ProposalKind.TOOL_CALL, is_write=False)
    r2 = _proposal(ProposalKind.TOOL_CALL, is_write=False)
    await s.submit(r1, WorkerKind.SLOW)
    await s.submit(r2, WorkerKind.SLOW)
    await asyncio.sleep(0.15)

    assert len(start_times) == 2, "Both reads must run"
    # Both started before the first one finished (concurrent)
    assert start_times[1] < end_times[0], "Reads must overlap (concurrent execution)"


@pytest.mark.asyncio
async def test_scheduler_write_proposals_are_serialized():
    """Two write proposals must run sequentially (write lock)."""
    sequence = []

    async def handler(p: ActionProposal):
        sequence.append(f"start:{p.idempotency_key}")
        await asyncio.sleep(0.05)
        sequence.append(f"end:{p.idempotency_key}")

    s = WorkerScheduler(handler)
    w1 = _proposal(ProposalKind.TOOL_CALL, is_write=True, ikey="w1")
    w2 = _proposal(ProposalKind.TOOL_CALL, is_write=True, ikey="w2")
    await s.submit(w1, WorkerKind.SLOW)
    await s.submit(w2, WorkerKind.SLOW)
    await asyncio.sleep(0.2)

    assert len(sequence) == 4
    # end of first must come before start of second
    assert sequence[1] == "end:w1", f"Expected end:w1 second, got {sequence}"
    assert sequence[2] == "start:w2", f"Expected start:w2 third, got {sequence}"


@pytest.mark.asyncio
async def test_scheduler_slot_ambiguous_blocks_writes():
    """Writes must be blocked when slot ambiguity flag is set."""
    called = []

    async def handler(p: ActionProposal):
        called.append(p)

    s = WorkerScheduler(handler)
    s.set_slot_ambiguous(SID, True)

    write_p = _proposal(ProposalKind.TOOL_CALL, is_write=True, ikey="w1")
    await s.submit(write_p, WorkerKind.SLOW)
    await asyncio.sleep(0.05)
    assert len(called) == 0, "Write must be blocked during slot ambiguity"

    # Reads still allowed
    s.set_slot_ambiguous(SID, False)
    read_p = _proposal(ProposalKind.TOOL_CALL, is_write=False)
    await s.submit(read_p, WorkerKind.SLOW)
    await asyncio.sleep(0.05)
    assert len(called) == 1, "Read must be allowed when ambiguity is cleared"


@pytest.mark.asyncio
async def test_scheduler_cancel_session_stops_tasks():
    """cancel_session must cancel all active worker tasks for that session."""
    started = asyncio.Event()
    finished = asyncio.Event()

    async def handler(p: ActionProposal):
        started.set()
        await asyncio.sleep(10.0)  # very long
        finished.set()

    s = WorkerScheduler(handler)
    await s.submit(_proposal(), WorkerKind.SLOW)
    await asyncio.sleep(0.02)  # let it start

    await s.cancel_session(SID)
    await asyncio.sleep(0.05)

    assert started.is_set(), "Task must have started"
    assert not finished.is_set(), "Task must have been cancelled before finishing"


@pytest.mark.asyncio
async def test_scheduler_multimodal_worker_runs_independently():
    called = []

    async def handler(p: ActionProposal):
        called.append(p.kind)

    s = WorkerScheduler(handler)
    mm = _proposal(ProposalKind.MULTIMODAL_EVIDENCE)
    await s.submit(mm, WorkerKind.MULTIMODAL)
    await asyncio.sleep(0.05)
    assert ProposalKind.MULTIMODAL_EVIDENCE in called


# ===========================================================================
# RecoveryPolicy tests
# ===========================================================================

@pytest.mark.asyncio
async def test_recovery_read_retry_succeeds_on_second_attempt():
    attempt_count = [0]

    async def call_fn():
        attempt_count[0] += 1
        if attempt_count[0] < 2:
            raise RuntimeError("transient")
        return "ok"

    policy = RecoveryPolicy(max_read_retries=3, base_backoff_ms=1.0, clock_now=lambda: 0.0)
    result = await policy.handle_read_failure(
        call_fn,
        call_id="c1",
        session_id=SID,
        generation=1,
        snapshot_version=1,
    )
    assert result == "ok"
    assert attempt_count[0] == 2
    assert policy.total_retries() == 1  # one retry record
    assert policy.retry_log[0].reason == RecoveryReason.TRANSIENT_FAILURE


@pytest.mark.asyncio
async def test_recovery_read_exhausted_raises():
    async def call_fn():
        raise RuntimeError("always fails")

    policy = RecoveryPolicy(max_read_retries=2, base_backoff_ms=1.0, clock_now=lambda: 0.0)
    with pytest.raises(RuntimeError, match="always fails"):
        await policy.handle_read_failure(
            call_fn,
            call_id="c1",
            session_id=SID,
            generation=1,
            snapshot_version=1,
        )
    assert policy.total_retries() == 2  # max_read_retries retry records


@pytest.mark.asyncio
async def test_recovery_stale_call_no_retry():
    policy = RecoveryPolicy(clock_now=lambda: 0.0)
    policy.handle_stale_call(
        call_id="c-stale",
        session_id=SID,
        generation=1,
        snapshot_version=1,
        reason="superseded",
    )
    assert policy.total_retries() == 1
    rec = policy.retry_log[0]
    assert rec.reason == RecoveryReason.STALE_SUPERSEDED
    assert rec.retry_number == 0  # no retry issued


@pytest.mark.asyncio
async def test_recovery_write_uncertainty_status_check():
    checked = []

    async def status_fn():
        checked.append(1)
        return "COMMITTED"

    policy = RecoveryPolicy(clock_now=lambda: 0.0)
    status = await policy.handle_write_uncertainty(
        status_fn,
        call_id="c-write",
        session_id=SID,
        generation=1,
        snapshot_version=1,
        idempotency_key="idem-001",
    )
    assert status == "COMMITTED"
    assert len(checked) == 1
    rec = policy.retry_log[0]
    assert rec.reason == RecoveryReason.WRITE_UNCERTAINTY
    assert rec.idempotency_key == "idem-001"


@pytest.mark.asyncio
async def test_recovery_validation_failure_calls_clarification():
    clarified = []

    async def clarify():
        clarified.append(1)
        return "clarification_sent"

    policy = RecoveryPolicy(clock_now=lambda: 0.0)
    result = await policy.handle_validation_failure(
        call_id="c-val",
        session_id=SID,
        generation=1,
        snapshot_version=1,
        error="missing required field: date",
        clarification_fn=clarify,
    )
    assert result == "clarification_sent"
    assert len(clarified) == 1
    rec = policy.retry_log[0]
    assert rec.reason == RecoveryReason.VALIDATION_ERROR


@pytest.mark.asyncio
async def test_recovery_replan_called_as_last_resort():
    replanned = []

    async def replan_fn():
        replanned.append(1)
        return "new_plan"

    policy = RecoveryPolicy(clock_now=lambda: 0.0)
    result = await policy.handle_replan(
        replan_fn,
        call_id="c-replan",
        session_id=SID,
        generation=1,
        snapshot_version=1,
    )
    assert result == "new_plan"
    assert replanned == [1]
    rec = policy.retry_log[0]
    assert rec.reason == RecoveryReason.REPLAN


def test_recovery_retries_for_call_lookup():
    policy = RecoveryPolicy(clock_now=lambda: 0.0)
    policy.handle_stale_call(call_id="orig", session_id=SID, generation=1, snapshot_version=1)
    policy.handle_stale_call(call_id="other", session_id=SID, generation=1, snapshot_version=1)

    records = policy.retries_for_call("orig")
    assert len(records) == 1
    assert records[0].original_call_id == "orig"
