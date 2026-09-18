"""
module3/runtime/events/output_events.py

Strongly typed output event factories.

Output events flow OUT of the runtime to external consumers (TTS, UI, clients).
Every output carries enough identity (session_id, task_id, generation) to be
correlated back to the originating input and traced through the full pipeline.

Note: We are NOT building TTS or a UI here. These events represent the
protocol tokens that downstream systems would consume.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseEvent, EventCategory, OutputEventType


# ---------------------------------------------------------------------------
# Payload schemas
# ---------------------------------------------------------------------------

class FillerPayload(BaseModel):
    """
    Short, immediate spoken acknowledgment emitted by the fast path.
    Keeps the interaction feeling alive while slow reasoning runs.

    Examples: "Sure, let me look into that.", "One moment...", "Got it."
    """
    text: str = Field(..., min_length=1)
    filler_type: str = Field("acknowledgment", description="acknowledgment | progress | clarification_prompt")

    model_config = {"extra": "forbid"}


class ToolCallPayload(BaseModel):
    """
    Non-blocking tool invocation request.

    call_id is generated here and must be echoed back in the
    corresponding TOOL_RESULT input event. This is the anchor for
    stale-result detection.
    """
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: float | None = Field(None, description="Expected max latency before treating as stale")
    idempotency_key: str | None = None
    generation: int | None = None
    snapshot_version: int | None = None
    manifest_version: int | None = None
    side_effect: str | None = None

    model_config = {"extra": "forbid"}


class CancellationPayload(BaseModel):
    """
    Explicit cancellation notice for a task or tool call.

    Emitted when the runtime determines work is superseded.
    Downstream consumers can use this to abort in-progress operations.
    """
    cancelled_task_id: str = Field(..., description="Task being cancelled")
    cancelled_call_id: str | None = Field(None, description="Specific tool call being cancelled, if any")
    reason: str = Field("interrupted", description="interrupted | superseded | timeout | error")
    generation_at_cancel: int | None = Field(None)

    model_config = {"extra": "forbid"}


class ClarificationPayload(BaseModel):
    """
    Clarification request emitted when the agent cannot proceed without more info.
    """
    text: str = Field(..., min_length=1, description="The clarification question")
    missing_slots: list[str] = Field(default_factory=list, description="Which information slots are missing")
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class StateSnapshot(BaseModel):
    """
    Structured state snapshot embedded in final responses.

    Allows downstream evaluators to verify not just the text but the
    structured outcome (e.g., booking_id, flight_info, confirmed slots).
    """
    intent: str | None = Field(None)
    slots: dict[str, Any] = Field(default_factory=dict)
    completed_actions: list[str] = Field(default_factory=list)
    pending_actions: list[str] = Field(default_factory=list)
    tool_calls_made: list[str] = Field(default_factory=list)
    result_data: dict[str, Any] = Field(default_factory=dict)
    snapshot_version: int | None = None
    generation: int | None = None
    floor_state: str | None = None

    model_config = {"extra": "forbid"}


class FinalResponsePayload(BaseModel):
    """
    Authoritative final response from the agent for a given task.

    Contains both the human-facing text and a structured StateSnapshot
    for evaluation purposes. The generation field ties this response
    to a specific epoch — any response from an older generation should
    have been intercepted as STALE before reaching this point.
    """
    text: str = Field(..., min_length=1)
    state_snapshot: StateSnapshot = Field(default_factory=StateSnapshot)
    is_complete: bool = Field(True, description="False if further turns are expected")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _make_output(
    event_type: OutputEventType,
    session_id: str,
    timestamp_ms: float,
    payload: BaseModel,
    **kwargs: Any,
) -> BaseEvent:
    return BaseEvent(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        event_type=event_type,
        category=EventCategory.OUTPUT,
        payload=payload.model_dump(),
        **kwargs,
    )


def make_filler(
    session_id: str,
    timestamp_ms: float,
    text: str,
    filler_type: str = "acknowledgment",
    **kwargs: Any,
) -> BaseEvent:
    p = FillerPayload(text=text, filler_type=filler_type)
    return _make_output(OutputEventType.FILLER, session_id, timestamp_ms, p, **kwargs)


def make_tool_call(
    session_id: str,
    timestamp_ms: float,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    call_id: str | None = None,
    timeout_ms: float | None = None,
    idempotency_key: str | None = None,
    generation: int | None = None,
    snapshot_version: int | None = None,
    manifest_version: int | None = None,
    side_effect: str | None = None,
    **kwargs: Any,
) -> tuple[BaseEvent, str]:
    """
    Returns (event, call_id) so the caller can register the call_id for
    result correlation without re-parsing the payload.
    """
    p = ToolCallPayload(
        call_id=call_id or str(uuid.uuid4()),
        tool_name=tool_name,
        arguments=arguments or {},
        timeout_ms=timeout_ms,
        idempotency_key=idempotency_key,
        generation=generation,
        snapshot_version=snapshot_version,
        manifest_version=manifest_version,
        side_effect=side_effect,
    )
    evt = _make_output(
        OutputEventType.TOOL_CALL,
        session_id,
        timestamp_ms,
        p,
        call_id=p.call_id,
        **kwargs,
    )
    return evt, p.call_id


def make_cancellation(
    session_id: str,
    timestamp_ms: float,
    cancelled_task_id: str,
    cancelled_call_id: str | None = None,
    reason: str = "interrupted",
    generation_at_cancel: int | None = None,
    **kwargs: Any,
) -> BaseEvent:
    p = CancellationPayload(
        cancelled_task_id=cancelled_task_id,
        cancelled_call_id=cancelled_call_id,
        reason=reason,
        generation_at_cancel=generation_at_cancel,
    )
    return _make_output(OutputEventType.CANCELLATION, session_id, timestamp_ms, p, **kwargs)


def make_clarification(
    session_id: str,
    timestamp_ms: float,
    text: str,
    missing_slots: list[str] | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseEvent:
    p = ClarificationPayload(
        text=text,
        missing_slots=missing_slots or [],
        context=context or {},
    )
    return _make_output(OutputEventType.CLARIFICATION, session_id, timestamp_ms, p, **kwargs)


def make_final_response(
    session_id: str,
    timestamp_ms: float,
    text: str,
    state_snapshot: StateSnapshot | None = None,
    is_complete: bool = True,
    **kwargs: Any,
) -> BaseEvent:
    p = FinalResponsePayload(
        text=text,
        state_snapshot=state_snapshot or StateSnapshot(),
        is_complete=is_complete,
    )
    return _make_output(OutputEventType.FINAL_RESPONSE, session_id, timestamp_ms, p, **kwargs)
