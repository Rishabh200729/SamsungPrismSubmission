"""module3/__init__.py — PRISM Module 3: Shared Runtime Architecture"""
from .runtime import Runtime, RuntimeConfig, TEST_CONFIG, PRODUCTION_CONFIG
from .runtime.events import (
    BaseEvent, InputEventType, OutputEventType, InternalEventType,
    make_text_chunk, make_end_of_turn, make_interruption, make_tool_result,
    make_filler, make_tool_call, make_final_response, make_cancellation,
)

__version__ = "0.1.0"
__all__ = [
    "Runtime", "RuntimeConfig", "TEST_CONFIG", "PRODUCTION_CONFIG",
    "BaseEvent", "InputEventType", "OutputEventType", "InternalEventType",
    "make_text_chunk", "make_end_of_turn", "make_interruption", "make_tool_result",
    "make_filler", "make_tool_call", "make_final_response", "make_cancellation",
]
