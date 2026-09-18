"""
module3/runtime/events/base.py

Base event model for the PRISM runtime.

Every event — input, output, or internal — shares this common envelope.
This ensures every event is strongly typed, validated, and carries
enough identity to be traced back to its session, task, and call.

Design: Pydantic v2 models for zero-ambiguity schema enforcement.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EventCategory(str, Enum):
    """High-level categorization of event visibility."""
    INPUT = "INPUT"           # Externally injected into the runtime
    OUTPUT = "OUTPUT"         # Externally observable from the runtime
    INTERNAL = "INTERNAL"     # Runtime lifecycle events, not sent to callers


class InputEventType(str, Enum):
    """All supported input event types from the challenge protocol."""
    TEXT_CHUNK = "TEXT_CHUNK"
    END_OF_TURN = "END_OF_TURN"
    AUDIO_WAV = "AUDIO_WAV"
    VIDEO_FRAME = "VIDEO_FRAME"
    INTERRUPTION = "INTERRUPTION"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_MANIFEST = "TOOL_MANIFEST"


class OutputEventType(str, Enum):
    """All supported output event types from the challenge protocol."""
    FILLER = "FILLER"
    TOOL_CALL = "TOOL_CALL"
    CANCELLATION = "CANCELLATION"
    CLARIFICATION = "CLARIFICATION"
    FINAL_RESPONSE = "FINAL_RESPONSE"


class InternalEventType(str, Enum):
    """Internal runtime lifecycle events, never surfaced externally."""
    TASK_CREATED = "TASK_CREATED"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_CANCEL_REQUESTED = "TASK_CANCEL_REQUESTED"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_FAILED = "TASK_FAILED"
    STALE_RESULT_DISCARDED = "STALE_RESULT_DISCARDED"
    GENERATION_INVALIDATED = "GENERATION_INVALIDATED"
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_CLOSED = "SESSION_CLOSED"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    DUPLICATE_CALL_ID = "DUPLICATE_CALL_ID"


# Union of all event type enums for type hints
AnyEventType = InputEventType | OutputEventType | InternalEventType


def _new_event_id() -> str:
    return str(uuid.uuid4())


class BaseEvent(BaseModel):
    """
    Universal event envelope.

    All events flowing through the PRISM runtime carry this structure.
    The event_id + session_id pair uniquely identifies any event.
    The timestamp_ms is the authoritative ordering key (virtual or real).

    Inspired by:
    - DuplexOmni: timestamped async streams between interaction/thinking layers
    - Gaia2: reproducible event injection with explicit timestamps
    - Sema Code: FIFO event handling with explicit ordering
    """

    event_id: str = Field(default_factory=_new_event_id, description="Unique event identifier")
    session_id: str = Field(..., description="Session this event belongs to")
    timestamp_ms: float = Field(..., description="Event timestamp in virtual/wall-clock milliseconds")
    event_type: AnyEventType = Field(..., description="Discriminated event type")
    category: EventCategory = Field(..., description="INPUT / OUTPUT / INTERNAL")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event-specific data")

    # Optional correlation fields — not all events need all of these
    task_id: str | None = Field(None, description="Associated task ID")
    call_id: str | None = Field(None, description="Tool call ID for correlation")
    generation: int | None = Field(None, description="Generation counter at time of event")
    correlation_id: str | None = Field(None, description="Arbitrary cross-event correlation")
    parent_task_id: str | None = Field(None, description="Parent task ID if hierarchical")
    source: str | None = Field(None, description="Emitting subsystem label")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary runtime metadata")

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("timestamp_ms")
    @classmethod
    def timestamp_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"timestamp_ms must be >= 0, got {v}")
        return v

    def to_trace_dict(self) -> dict[str, Any]:
        """Compact representation suitable for trace logs."""
        d = {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp_ms": self.timestamp_ms,
            "event_type": str(self.event_type.value),
            "category": self.category.value,
        }
        for k in ("task_id", "call_id", "generation", "correlation_id", "source"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.payload:
            d["payload"] = self.payload
        if self.metadata:
            d["metadata"] = self.metadata
        return d
