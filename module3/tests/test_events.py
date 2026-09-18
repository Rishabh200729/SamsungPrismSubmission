"""
module3/tests/test_events.py

Tests for event schema validation and factory functions.
"""
import pytest
from pydantic import ValidationError

from module3.runtime.events.base import BaseEvent, EventCategory, InputEventType, OutputEventType
from module3.runtime.events import (
    make_text_chunk, make_end_of_turn, make_interruption,
    make_tool_result, make_tool_manifest, make_audio_wav, make_video_frame,
    make_filler, make_tool_call, make_cancellation, make_clarification, make_final_response,
    StateSnapshot,
)


SESSION = "test-session-001"
TS = 100.0


class TestBaseEvent:
    def test_event_requires_session_id(self):
        with pytest.raises(ValidationError):
            BaseEvent(
                timestamp_ms=0.0,
                event_type=InputEventType.TEXT_CHUNK,
                category=EventCategory.INPUT,
            )

    def test_event_requires_non_negative_timestamp(self):
        with pytest.raises(ValidationError):
            BaseEvent(
                session_id=SESSION,
                timestamp_ms=-1.0,
                event_type=InputEventType.TEXT_CHUNK,
                category=EventCategory.INPUT,
            )

    def test_event_auto_generates_event_id(self):
        e = make_text_chunk(SESSION, 0.0, "hello")
        assert e.event_id
        assert len(e.event_id) == 36  # UUID format

    def test_two_events_have_different_ids(self):
        e1 = make_text_chunk(SESSION, 0.0, "hello")
        e2 = make_text_chunk(SESSION, 0.0, "hello")
        assert e1.event_id != e2.event_id

    def test_to_trace_dict(self):
        e = make_text_chunk(SESSION, 42.5, "hello")
        d = e.to_trace_dict()
        assert d["session_id"] == SESSION
        assert d["timestamp_ms"] == 42.5
        assert d["event_type"] == "TEXT_CHUNK"
        assert d["category"] == "INPUT"


class TestInputEventFactories:
    def test_text_chunk(self):
        e = make_text_chunk(SESSION, TS, "Book a flight", is_partial=True)
        assert e.event_type == InputEventType.TEXT_CHUNK
        assert e.category == EventCategory.INPUT
        assert e.payload["text"] == "Book a flight"
        assert e.payload["is_partial"] is True

    def test_text_chunk_requires_text(self):
        with pytest.raises(ValidationError):
            make_text_chunk(SESSION, TS, "")  # min_length=1

    def test_end_of_turn(self):
        e = make_end_of_turn(SESSION, TS, final_text="Done")
        assert e.event_type == InputEventType.END_OF_TURN
        assert e.payload["final_text"] == "Done"
        assert e.payload["utterance_id"]  # auto-generated

    def test_audio_wav(self):
        e = make_audio_wav(SESSION, TS, audio_b64="base64data", duration_ms=500.0)
        assert e.event_type == InputEventType.AUDIO_WAV
        assert e.payload["duration_ms"] == 500.0

    def test_video_frame(self):
        e = make_video_frame(SESSION, TS, frame_b64="base64frame", width=640, height=480, frame_index=3)
        assert e.event_type == InputEventType.VIDEO_FRAME
        assert e.payload["frame_index"] == 3

    def test_interruption(self):
        e = make_interruption(SESSION, TS, text="Stop!", reason="user_interruption")
        assert e.event_type == InputEventType.INTERRUPTION
        assert e.payload["reason"] == "user_interruption"
        assert e.payload["text"] == "Stop!"

    def test_tool_result_sets_call_id_and_task_id(self):
        e = make_tool_result(SESSION, TS, call_id="c1", task_id="t1", result={"data": 42})
        assert e.event_type == InputEventType.TOOL_RESULT
        assert e.call_id == "c1"
        assert e.task_id == "t1"
        assert e.payload["result"] == {"data": 42}

    def test_tool_manifest(self):
        tools = [{"name": "search", "description": "web search"}]
        e = make_tool_manifest(SESSION, TS, tools=tools)
        assert e.event_type == InputEventType.TOOL_MANIFEST
        assert len(e.payload["tools"]) == 1


class TestOutputEventFactories:
    def test_filler(self):
        e = make_filler(SESSION, TS, text="One moment...", filler_type="progress")
        assert e.event_type == OutputEventType.FILLER
        assert e.category == EventCategory.OUTPUT
        assert e.payload["filler_type"] == "progress"

    def test_tool_call_returns_event_and_call_id(self):
        evt, cid = make_tool_call(SESSION, TS, tool_name="search", arguments={"q": "test"})
        assert evt.event_type == OutputEventType.TOOL_CALL
        assert cid
        assert evt.call_id == cid
        assert evt.payload["tool_name"] == "search"

    def test_tool_call_with_explicit_call_id(self):
        evt, cid = make_tool_call(SESSION, TS, tool_name="search", call_id="explicit-id")
        assert cid == "explicit-id"
        assert evt.call_id == "explicit-id"

    def test_cancellation(self):
        e = make_cancellation(SESSION, TS, cancelled_task_id="t1", reason="interrupted")
        assert e.event_type == OutputEventType.CANCELLATION
        assert e.payload["cancelled_task_id"] == "t1"

    def test_clarification(self):
        e = make_clarification(SESSION, TS, text="Which city?", missing_slots=["destination"])
        assert e.event_type == OutputEventType.CLARIFICATION
        assert e.payload["missing_slots"] == ["destination"]

    def test_final_response_with_state_snapshot(self):
        snap = StateSnapshot(
            intent="book_flight",
            slots={"origin": "Seoul", "destination": "Tokyo"},
            completed_actions=["search", "book"],
        )
        e = make_final_response(SESSION, TS, text="Your flight is booked!", state_snapshot=snap)
        assert e.event_type == OutputEventType.FINAL_RESPONSE
        assert e.payload["state_snapshot"]["intent"] == "book_flight"
        assert e.payload["state_snapshot"]["slots"]["origin"] == "Seoul"
