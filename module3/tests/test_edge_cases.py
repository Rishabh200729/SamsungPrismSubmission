"""
module3/tests/test_edge_cases.py

Tests for the 9 bugs identified in the code review.
These specifically target the previously uncovered lines in the coverage report.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import warnings
from collections import deque

import pytest

from module3.runtime.config import TEST_CONFIG
from module3.runtime.events import make_filler, make_interruption, make_text_chunk
from module3.runtime.events.base import InputEventType
from module3.runtime.runtime import Runtime
from module3.runtime.tasks.lifecycle import InvalidTransitionError, TaskStatus
from module3.runtime.clock.virtual_clock import VirtualClock


async def _make_runtime() -> Runtime:
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    await rt.start()
    return rt


# --- Bug 3: stop() must await dispatch task after cancel ---

@pytest.mark.asyncio
async def test_stop_dispatch_loop_is_truly_done():
    rt = await _make_runtime()
    loop_task = rt._dispatch_loop_task
    assert not loop_task.done()
    await rt.stop()
    assert loop_task.done(), "Dispatch loop must be done after stop() -- Bug 3"


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    rt = await _make_runtime()
    await rt.stop()
    await rt.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent(caplog):
    import logging
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    first_task = rt._dispatch_loop_task
    with caplog.at_level(logging.WARNING, logger="module3.runtime.runtime"):
        await rt.start()
    assert rt._dispatch_loop_task is first_task, "Second start() must not spawn a new loop"
    await rt.stop()


# --- Bug 4: submit_event drops events for unknown/closed sessions ---

@pytest.mark.asyncio
async def test_submit_event_unknown_session_is_dropped(caplog):
    import logging
    rt = await _make_runtime()
    q_before = rt.input_queue.qsize()
    with caplog.at_level(logging.WARNING, logger="module3.runtime.runtime"):
        await rt.submit_event(make_text_chunk("nonexistent-session", 0.0, "hello"))
    assert rt.input_queue.qsize() == q_before, "Unknown session event must NOT enter queue"
    assert "unknown session_id" in caplog.text
    await rt.stop()


@pytest.mark.asyncio
async def test_submit_event_closed_session_is_dropped(caplog):
    import logging
    rt = await _make_runtime()
    sid = rt.create_session()
    await rt.close_session(sid)
    q_before = rt.input_queue.qsize()
    with caplog.at_level(logging.WARNING, logger="module3.runtime.runtime"):
        await rt.submit_event(make_text_chunk(sid, 0.0, "late event"))
    assert rt.input_queue.qsize() == q_before, "Closed session event must NOT enter queue"
    assert "is closed" in caplog.text
    await rt.stop()


@pytest.mark.asyncio
async def test_submit_event_valid_session_is_recorded():
    rt = await _make_runtime()
    sid = rt.create_session()
    await rt.submit_event(make_text_chunk(sid, 0.0, "hello"))
    ctx = rt.get_session(sid)
    assert len(ctx.event_log) == 1
    await rt.stop()


# --- Bug 1: coroutine must be closed when session is missing ---

@pytest.mark.asyncio
async def test_coroutine_closed_when_session_missing():
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    executed = []

    async def work():
        executed.append(True)
        await asyncio.sleep(0)

    coro = work()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        await rt._lifecycle_wrapper("missing-session", "fake-task", 1, coro)

    assert executed == [], "Coroutine must not execute when session is missing"
    await rt.stop()


# --- Bug 2: already-cancelled task must not execute ---

@pytest.mark.asyncio
async def test_cancelled_task_does_not_execute():
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()
    ctx = rt.get_session(sid)
    executed = []

    async def work():
        executed.append(True)

    record = ctx.task_registry.create_task("test", generation=1)
    task_id = record.task_id
    ctx.task_registry.transition(task_id, TaskStatus.CANCELLATION_REQUESTED)

    coro = work()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        await rt._lifecycle_wrapper(sid, task_id, 1, coro)

    assert executed == [], "Already-cancelled task must not execute its coroutine"
    await rt.stop()


# --- Public cancel_task API (was 0% covered) ---

@pytest.mark.asyncio
async def test_public_cancel_task_transitions_state():
    rt = await _make_runtime()
    sid = rt.create_session()
    blocked = asyncio.Event()

    async def blocking():
        await blocked.wait()

    task_id = await rt.create_background_task(sid, blocking(), "test")
    await asyncio.sleep(0)

    result = await rt.cancel_task(sid, task_id, reason="manual_cancel")
    assert result is True
    assert rt.get_task(sid, task_id).status == TaskStatus.CANCELLATION_REQUESTED

    blocked.set()
    await asyncio.sleep(0)
    await rt.stop()


@pytest.mark.asyncio
async def test_public_cancel_task_unknown_session_returns_false():
    rt = await _make_runtime()
    result = await rt.cancel_task("nonexistent", "some-task")
    assert result is False
    await rt.stop()


@pytest.mark.asyncio
async def test_public_cancel_task_unknown_task_returns_false():
    rt = await _make_runtime()
    sid = rt.create_session()
    result = await rt.cancel_task(sid, "nonexistent-task-id")
    assert result is False
    await rt.stop()


@pytest.mark.asyncio
async def test_cancel_task_records_trace_entry():
    rt = await _make_runtime()
    sid = rt.create_session()
    blocked = asyncio.Event()

    async def blocking():
        await blocked.wait()

    task_id = await rt.create_background_task(sid, blocking(), "test")
    await asyncio.sleep(0)
    await rt.cancel_task(sid, task_id)

    trace_types = [e.entry_type for e in rt.tracer.entries()]
    assert "TASK_CANCEL_REQUESTED" in trace_types

    blocked.set()
    await asyncio.sleep(0)
    await rt.stop()


# --- Bug 6: dispatcher traceback is preserved ---

@pytest.mark.asyncio
async def test_dispatcher_error_preserves_exception_type():
    from module3.runtime.dispatcher.dispatcher import Dispatcher
    dispatcher = Dispatcher()

    async def bad_handler(event):
        raise ValueError("intentional test error")

    dispatcher.register_handler(InputEventType.TEXT_CHUNK, bad_handler)
    event = make_text_chunk("s1", 0.0, "hi")
    specific_handlers = dispatcher._handlers.get(event.event_type, [])
    results = await asyncio.gather(
        *[dispatcher._call_handler(h, event) for h in specific_handlers],
        return_exceptions=True,
    )
    assert len(results) == 1
    assert isinstance(results[0], ValueError), (
        f"Expected ValueError, got {type(results[0])}. Bug 6: raise exc loses type."
    )


# --- Bug 7: CANCELLATION_REQUESTED -> COMPLETED must be illegal ---

def test_cancellation_requested_to_completed_is_illegal():
    assert not TaskStatus.CANCELLATION_REQUESTED.can_transition_to(TaskStatus.COMPLETED), (
        "CANCELLATION_REQUESTED->COMPLETED must be illegal (Bug 7)"
    )


def test_cancellation_requested_to_cancelled_is_legal():
    assert TaskStatus.CANCELLATION_REQUESTED.can_transition_to(TaskStatus.CANCELLED)


def test_cancellation_requested_to_stale_is_legal():
    assert TaskStatus.CANCELLATION_REQUESTED.can_transition_to(TaskStatus.STALE)


def test_registry_rejects_cancellation_requested_to_completed():
    from module3.runtime.tasks.registry import TaskRegistry
    registry = TaskRegistry(session_id="test")
    record = registry.create_task("test", generation=1)
    registry.transition(record.task_id, TaskStatus.RUNNING)
    registry.transition(record.task_id, TaskStatus.CANCELLATION_REQUESTED)
    with pytest.raises(InvalidTransitionError):
        registry.transition(record.task_id, TaskStatus.COMPLETED)


# --- Bug 9: event_log must be bounded ---

def test_event_log_bounded_at_1000():
    from module3.runtime.sessions.manager import SessionContext
    ctx = SessionContext(session_id="test", created_ms=0.0)
    for i in range(2000):
        ctx.record_event_id(f"event-{i}")
    assert len(ctx.event_log) == 1000
    assert "event-1999" in ctx.event_log
    assert "event-0" not in ctx.event_log


def test_event_log_type_is_deque():
    from module3.runtime.sessions.manager import SessionContext
    ctx = SessionContext(session_id="test", created_ms=0.0)
    assert isinstance(ctx.event_log, deque)


# --- Bug 5: session_id passed directly to stale callback ---

@pytest.mark.asyncio
async def test_stale_trace_uses_correct_session_id():
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    async def on_interrupt(event):
        await rt.request_cancellation(event.session_id)

    rt.register_handler(InputEventType.INTERRUPTION, on_interrupt)
    await rt.start()
    sid = rt.create_session()

    blocked = asyncio.Event()

    async def slow():
        await blocked.wait()

    task_id = await rt.create_background_task(sid, slow(), "slow")
    await asyncio.sleep(0)

    await rt.submit_event(make_interruption(sid, 100.0))
    for _ in range(5):
        await asyncio.sleep(0)

    await rt.handle_stale_result(sid, task_id, call_id="call-xyz")
    await asyncio.sleep(0)

    stale_entries = [e for e in rt.tracer.entries() if e.entry_type == "STALE_RESULT_DISCARDED"]
    assert len(stale_entries) == 1
    assert stale_entries[0].session_id == sid, (
        f"Expected session_id={sid!r}, got {stale_entries[0].session_id!r}. Bug 5."
    )
    assert stale_entries[0].session_id != "unknown"

    blocked.set()
    await asyncio.sleep(0)
    await rt.stop()


@pytest.mark.asyncio
async def test_runtime_helpers_priority_timeout_and_trace_export(tmp_path):
    rt = await _make_runtime()
    sid = rt.create_session(metadata={"user": "ari"})
    assert rt.get_session_metadata(sid) == {"user": "ari"}
    assert rt.get_session_metadata("missing") == {}

    input_event = make_text_chunk(sid, 0, "hello")
    await rt.submit_event(input_event)
    for _ in range(3):
        await asyncio.sleep(0)
    assert input_event.event_id in rt.get_session_event_log(sid)

    with pytest.raises(asyncio.TimeoutError):
        await rt.next_output(timeout_ms=1)
    low = make_filler(sid, 2, "low")
    high = make_filler(sid, 1, "high")
    await rt.emit_output(low)
    await rt.emit_output(high, priority=3)
    assert (await rt.next_output()).event_id == high.event_id
    assert [event.event_id for event in await rt.drain_all_outputs()] == [low.event_id]

    path = tmp_path / "trace.jsonl"
    rt.export_trace_jsonl(str(path))
    assert path.exists() and path.read_text()
    await rt.stop()


@pytest.mark.asyncio
async def test_runtime_records_opt_in_input_queue_overflow():
    config = replace(TEST_CONFIG, input_queue_maxsize=1, input_queue_drop_oldest=True)
    rt = Runtime(config=config)
    sid = rt.create_session()
    first = make_text_chunk(sid, 0, "first")
    second = make_text_chunk(sid, 1, "second")
    await rt.submit_event(first)
    await rt.submit_event(second)

    overflow = [entry for entry in rt.tracer.entries() if entry.entry_type == "QUEUE_OVERFLOW"]
    assert len(overflow) == 1
    assert overflow[0].session_id == sid
    assert overflow[0].metadata["dropped_event_id"] == first.event_id
    assert overflow[0].metadata["queue_maxsize"] == 1
    assert (await rt.input_queue.get()).event_id == second.event_id
    rt.input_queue.task_done()
