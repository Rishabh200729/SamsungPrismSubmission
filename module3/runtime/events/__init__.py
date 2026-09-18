"""
module3/runtime/events/__init__.py

Public re-exports for the events package.
"""

from .base import (
    AnyEventType,
    BaseEvent,
    EventCategory,
    InputEventType,
    InternalEventType,
    OutputEventType,
)
from .input_events import (
    AudioWavPayload,
    EndOfTurnPayload,
    InterruptionPayload,
    TextChunkPayload,
    ToolManifestPayload,
    ToolResultPayload,
    VideoFramePayload,
    make_audio_wav,
    make_end_of_turn,
    make_interruption,
    make_text_chunk,
    make_tool_manifest,
    make_tool_result,
    make_video_frame,
)
from .internal_events import (
    make_duplicate_call_id,
    make_generation_invalidated,
    make_queue_overflow,
    make_session_closed,
    make_session_created,
    make_stale_result_discarded,
    make_task_cancel_requested,
    make_task_cancelled,
    make_task_completed,
    make_task_created,
    make_task_failed,
    make_task_started,
)
from .output_events import (
    CancellationPayload,
    ClarificationPayload,
    FillerPayload,
    FinalResponsePayload,
    StateSnapshot,
    ToolCallPayload,
    make_cancellation,
    make_clarification,
    make_filler,
    make_final_response,
    make_tool_call,
)

__all__ = [
    # base
    "AnyEventType",
    "BaseEvent",
    "EventCategory",
    "InputEventType",
    "InternalEventType",
    "OutputEventType",
    # input payloads
    "AudioWavPayload",
    "EndOfTurnPayload",
    "InterruptionPayload",
    "TextChunkPayload",
    "ToolManifestPayload",
    "ToolResultPayload",
    "VideoFramePayload",
    # input factories
    "make_audio_wav",
    "make_end_of_turn",
    "make_interruption",
    "make_text_chunk",
    "make_tool_manifest",
    "make_tool_result",
    "make_video_frame",
    # output payloads
    "CancellationPayload",
    "ClarificationPayload",
    "FillerPayload",
    "FinalResponsePayload",
    "StateSnapshot",
    "ToolCallPayload",
    # output factories
    "make_cancellation",
    "make_clarification",
    "make_filler",
    "make_final_response",
    "make_tool_call",
    # internal factories
    "make_duplicate_call_id",
    "make_generation_invalidated",
    "make_queue_overflow",
    "make_session_closed",
    "make_session_created",
    "make_stale_result_discarded",
    "make_task_cancel_requested",
    "make_task_cancelled",
    "make_task_completed",
    "make_task_created",
    "make_task_failed",
    "make_task_started",
]
