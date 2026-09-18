"""
module3/runtime/events/internal_events.py

Internal runtime lifecycle event factories.

Internal events are NEVER surfaced to external callers — they exist solely
for the trace recorder and for introspection/debugging. They record every
meaningful state transition inside the runtime kernel.

Inspired by:
- Sema Code: explicit background task observability
- Structured Graph Harness: every state transition is an explicit event
- Gaia2: reproducible traces for deterministic evaluation
"""

from __future__ import annotations

from typing import Any

from .base import BaseEvent, EventCategory, InternalEventType


def _make_internal(
    event_type: InternalEventType,
    session_id: str,
    timestamp_ms: float,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseEvent:
    return BaseEvent(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        event_type=event_type,
        category=EventCategory.INTERNAL,
        payload=payload or {},
        **kwargs,
    )


def make_task_created(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    task_type: str,
    generation: int,
    parent_task_id: str | None = None,
    call_id: str | None = None,
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.TASK_CREATED,
        session_id,
        timestamp_ms,
        payload={"task_type": task_type},
        task_id=task_id,
        generation=generation,
        parent_task_id=parent_task_id,
        call_id=call_id,
        **kwargs,
    )


def make_task_started(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    generation: int,
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.TASK_STARTED,
        session_id,
        timestamp_ms,
        task_id=task_id,
        generation=generation,
        **kwargs,
    )


def make_task_completed(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    generation: int,
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.TASK_COMPLETED,
        session_id,
        timestamp_ms,
        task_id=task_id,
        generation=generation,
        **kwargs,
    )


def make_task_cancel_requested(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    generation: int,
    reason: str = "interrupted",
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.TASK_CANCEL_REQUESTED,
        session_id,
        timestamp_ms,
        payload={"reason": reason},
        task_id=task_id,
        generation=generation,
        **kwargs,
    )


def make_task_cancelled(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    generation: int,
    phase: str | None = None,
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.TASK_CANCELLED,
        session_id,
        timestamp_ms,
        payload={"phase": phase},
        task_id=task_id,
        generation=generation,
        **kwargs,
    )


def make_task_failed(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    generation: int,
    error: str = "unknown",
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.TASK_FAILED,
        session_id,
        timestamp_ms,
        payload={"error": error},
        task_id=task_id,
        generation=generation,
        **kwargs,
    )


def make_stale_result_discarded(
    session_id: str,
    timestamp_ms: float,
    task_id: str,
    call_id: str | None,
    task_generation: int,
    current_generation: int,
    **kwargs: Any,
) -> BaseEvent:
    """
    Records when a late tool result arrives from a superseded generation
    and is explicitly discarded rather than silently swallowed.

    This event is what makes stale-result protection observable.
    """
    return _make_internal(
        InternalEventType.STALE_RESULT_DISCARDED,
        session_id,
        timestamp_ms,
        payload={
            "task_generation": task_generation,
            "current_generation": current_generation,
            "discarded_call_id": call_id,
        },
        task_id=task_id,
        call_id=call_id,
        generation=current_generation,
        **kwargs,
    )


def make_generation_invalidated(
    session_id: str,
    timestamp_ms: float,
    old_generation: int,
    new_generation: int,
    reason: str = "interruption",
    **kwargs: Any,
) -> BaseEvent:
    """
    Records when a generation counter is incremented, marking all
    work from the old generation as potentially stale.

    Core of the DuplexOmni-inspired async cancellation mechanism.
    """
    return _make_internal(
        InternalEventType.GENERATION_INVALIDATED,
        session_id,
        timestamp_ms,
        payload={
            "old_generation": old_generation,
            "new_generation": new_generation,
            "reason": reason,
        },
        generation=new_generation,
        **kwargs,
    )


def make_session_created(
    session_id: str,
    timestamp_ms: float,
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.SESSION_CREATED,
        session_id,
        timestamp_ms,
        **kwargs,
    )


def make_session_closed(
    session_id: str,
    timestamp_ms: float,
    **kwargs: Any,
) -> BaseEvent:
    return _make_internal(
        InternalEventType.SESSION_CLOSED,
        session_id,
        timestamp_ms,
        **kwargs,
    )


def make_queue_overflow(
    session_id: str,
    timestamp_ms: float,
    dropped_event_id: str,
    dropped_event_type: str,
    dropped_session_id: str,
    queue_maxsize: int,
    **kwargs: Any,
) -> BaseEvent:
    """Record an input event discarded by the opt-in overflow policy."""
    return _make_internal(
        InternalEventType.QUEUE_OVERFLOW,
        session_id,
        timestamp_ms,
        payload={
            "dropped_event_id": dropped_event_id,
            "dropped_event_type": dropped_event_type,
            "dropped_session_id": dropped_session_id,
            "queue_maxsize": queue_maxsize,
        },
        **kwargs,
    )


def make_duplicate_call_id(
    session_id: str,
    timestamp_ms: float,
    call_id: str,
    tool_name: str,
    generation: int,
    rejected_event_id: str | None = None,
    **kwargs: Any,
) -> BaseEvent:
    """Record that an output-gate soft-rejected a duplicate call_id.

    The duplicate TOOL_CALL is not forwarded to the output queue; instead
    this trace entry is created so metrics can detect the protocol violation.
    """
    return _make_internal(
        InternalEventType.DUPLICATE_CALL_ID,
        session_id,
        timestamp_ms,
        payload={
            "call_id": call_id,
            "tool_name": tool_name,
            "rejected_event_id": rejected_event_id,
        },
        call_id=call_id,
        generation=generation,
        **kwargs,
    )
