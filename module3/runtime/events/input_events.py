"""
module3/runtime/events/input_events.py

Strongly typed input event factories.

Input events flow INTO the runtime from external producers (users, sensors,
tool orchestrators). Each factory function creates a fully validated BaseEvent
with a typed payload schema enforced via a nested Pydantic model.

This module does NOT contain routing or processing logic.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseEvent, EventCategory, InputEventType


# ---------------------------------------------------------------------------
# Payload schemas — one Pydantic model per input event type
# ---------------------------------------------------------------------------

class TextChunkPayload(BaseModel):
    """Transcribed text chunk arriving from the ASR layer (already transcribed)."""
    text: str = Field(..., min_length=1)
    is_partial: bool = Field(False, description="True if this is a mid-utterance partial")

    model_config = {"extra": "forbid"}


class EndOfTurnPayload(BaseModel):
    """Marks the end of a user turn — triggers slow-path reasoning."""
    utterance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    final_text: str | None = Field(None, description="Complete assembled utterance if available")

    model_config = {"extra": "forbid"}


class AudioWavPayload(BaseModel):
    """Raw audio chunk (bytes encoded as base64 string for JSON transport)."""
    audio_b64: str = Field(..., description="Base64-encoded WAV bytes")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    duration_ms: float = Field(..., description="Duration of this chunk in ms")
    channel: int = Field(0)

    model_config = {"extra": "forbid"}


class VideoFramePayload(BaseModel):
    """Single video frame (PNG encoded as base64 string)."""
    frame_b64: str = Field(..., description="Base64-encoded PNG bytes")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    frame_index: int = Field(..., ge=0)

    model_config = {"extra": "forbid"}


class InterruptionPayload(BaseModel):
    """
    User interruption signal.

    The runtime uses this to trigger generation increment and
    cancellation of superseded tasks.

    Inspired by: DuplexOmni async collaboration — slow work must not
    continue after the user redirects the conversation.
    """
    reason: str = Field("user_interruption")
    text: str | None = Field(None, description="Optional new text from the user at interrupt time")
    supersedes_generation: int | None = Field(
        None, description="If set, explicitly supersedes this generation"
    )

    model_config = {"extra": "forbid"}


class ToolResultPayload(BaseModel):
    """
    Result arriving from an asynchronous tool call.

    CRITICAL: call_id and task_id are required so the runtime can
    unambiguously route the result. If the task has been cancelled or
    the generation has advanced, the result will be marked STALE.
    """
    call_id: str = Field(..., description="Must match the call_id of the originating TOOL_CALL output")
    task_id: str = Field(..., description="Must match the task_id that issued the call")
    result: Any = Field(None, description="Tool output — schema depends on the tool")
    success: bool = Field(True)
    error: str | None = Field(None, description="Error message if success=False")
    latency_ms: float | None = Field(None, description="Tool-reported execution latency")

    model_config = {"extra": "forbid"}


class ToolManifestPayload(BaseModel):
    """Tool capability advertisement — scenario-scoped list of available tools."""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    scenario_id: str | None = Field(None)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Factory functions — the public API for creating input events
# ---------------------------------------------------------------------------

def _make_input(
    event_type: InputEventType,
    session_id: str,
    timestamp_ms: float,
    payload: BaseModel,
    **kwargs: Any,
) -> BaseEvent:
    return BaseEvent(
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        event_type=event_type,
        category=EventCategory.INPUT,
        payload=payload.model_dump(),
        **kwargs,
    )


def make_text_chunk(
    session_id: str,
    timestamp_ms: float,
    text: str,
    is_partial: bool = False,
    **kwargs: Any,
) -> BaseEvent:
    return _make_input(
        InputEventType.TEXT_CHUNK,
        session_id,
        timestamp_ms,
        TextChunkPayload(text=text, is_partial=is_partial),
        **kwargs,
    )


def make_end_of_turn(
    session_id: str,
    timestamp_ms: float,
    final_text: str | None = None,
    utterance_id: str | None = None,
    **kwargs: Any,
) -> BaseEvent:
    p = EndOfTurnPayload(
        utterance_id=utterance_id or str(uuid.uuid4()),
        final_text=final_text,
    )
    return _make_input(InputEventType.END_OF_TURN, session_id, timestamp_ms, p, **kwargs)


def make_audio_wav(
    session_id: str,
    timestamp_ms: float,
    audio_b64: str,
    duration_ms: float,
    sample_rate: int = 16000,
    **kwargs: Any,
) -> BaseEvent:
    p = AudioWavPayload(audio_b64=audio_b64, sample_rate=sample_rate, duration_ms=duration_ms)
    return _make_input(InputEventType.AUDIO_WAV, session_id, timestamp_ms, p, **kwargs)


def make_video_frame(
    session_id: str,
    timestamp_ms: float,
    frame_b64: str,
    width: int,
    height: int,
    frame_index: int,
    **kwargs: Any,
) -> BaseEvent:
    p = VideoFramePayload(frame_b64=frame_b64, width=width, height=height, frame_index=frame_index)
    return _make_input(InputEventType.VIDEO_FRAME, session_id, timestamp_ms, p, **kwargs)


def make_interruption(
    session_id: str,
    timestamp_ms: float,
    text: str | None = None,
    reason: str = "user_interruption",
    supersedes_generation: int | None = None,
    **kwargs: Any,
) -> BaseEvent:
    p = InterruptionPayload(
        reason=reason,
        text=text,
        supersedes_generation=supersedes_generation,
    )
    return _make_input(InputEventType.INTERRUPTION, session_id, timestamp_ms, p, **kwargs)


def make_tool_result(
    session_id: str,
    timestamp_ms: float,
    call_id: str,
    task_id: str,
    result: Any = None,
    success: bool = True,
    error: str | None = None,
    latency_ms: float | None = None,
    **kwargs: Any,
) -> BaseEvent:
    p = ToolResultPayload(
        call_id=call_id,
        task_id=task_id,
        result=result,
        success=success,
        error=error,
        latency_ms=latency_ms,
    )
    return _make_input(
        InputEventType.TOOL_RESULT,
        session_id,
        timestamp_ms,
        p,
        call_id=call_id,
        task_id=task_id,
        **kwargs,
    )


def make_tool_manifest(
    session_id: str,
    timestamp_ms: float,
    tools: list[dict[str, Any]],
    scenario_id: str | None = None,
    **kwargs: Any,
) -> BaseEvent:
    p = ToolManifestPayload(tools=tools, scenario_id=scenario_id)
    return _make_input(InputEventType.TOOL_MANIFEST, session_id, timestamp_ms, p, **kwargs)
