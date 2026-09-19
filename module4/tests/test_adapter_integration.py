"""
Integration tests against the REAL module3.runtime.Runtime — not a fake/mock runtime.

These prove three things:
  1. Module4Adapter correctly buffers VIDEO_FRAME events and reads TOOL_MANIFEST events
     through the real event factories and real dispatcher.
  2. The direct-call path (`evaluate_ambiguity`) is race-free: Module 2's handler can
     check ambiguity in-line before deciding whether to emit a tool call.
  3. The DEFENSIVE END_OF_TURN handler, used alone (without the direct-call path), really
     does race against a concurrently-registered Module 2 handler and CAN produce both a
     CLARIFICATION and a TOOL_CALL for the same turn — this is demonstrated, not just
     asserted in a docstring, so nobody has to take my word for it.

Run: python -m pytest module4/tests/test_adapter_integration.py -v
"""

import asyncio

import pytest

from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.events import (
    InputEventType,
    make_end_of_turn,
    make_final_response,
    make_tool_call,
    make_tool_manifest,
    make_video_frame,
)

from module4.adapter import Module4Adapter

TOOL_SET_POWER = {
    "name": "set_power",
    "input_schema": {
        "required": ["device"],
        "properties": {"device": {"description": "device name"}},
    },
    "side_effect": "STATE_MODIFYING",
}


async def fake_caption_fn(frame_b64: str, frame_index: int):
    """Deterministic stand-in for the real/mock captioner, keyed off frame_b64 content
    so tests can simulate specific vision outcomes without touching module3.mocks."""
    if frame_b64 == "two_lamps":
        return "two lamps on a shelf", ["left lamp", "right lamp"], 0.9
    if frame_b64 == "one_button":
        return "a single red button", ["red button"], 0.9
    return "unrecognized", [], 0.1


@pytest.mark.asyncio
async def test_video_frame_and_manifest_flow_through_real_dispatcher():
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    m4 = Module4Adapter(rt, caption_fn_async=fake_caption_fn)
    m4.install()
    sid = rt.create_session()

    await rt.submit_event(make_tool_manifest(sid, rt.clock.now(), tools=[TOOL_SET_POWER]))
    await rt.submit_event(make_video_frame(sid, rt.clock.now(), "one_button", 640, 480, 0))
    await rt.input_queue.join()
    await asyncio.sleep(0)

    assert m4._active_tool[sid]["name"] == "set_power"
    latest = m4.buffer.latest_frame(sid)
    assert latest is not None and latest.salient_objects == ["red button"]

    await rt.stop()


@pytest.mark.asyncio
async def test_direct_call_path_is_race_free():
    """This is the integration pattern Module 2 should actually adopt."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    m4 = Module4Adapter(rt, caption_fn_async=fake_caption_fn)
    # Deliberately do NOT call m4.install() here — we only want the direct-call surface,
    # not the defensive handler, to prove the direct path alone is sufficient and clean.
    sid = rt.create_session()
    await rt.submit_event(make_tool_manifest(sid, rt.clock.now(), tools=[TOOL_SET_POWER]))
    await rt.input_queue.join()

    emitted = {"tool_call": False, "clarification": False}

    async def module2_end_of_turn(event):
        final_text = event.payload.get("final_text") or ""
        verdict = m4.evaluate_ambiguity(event.session_id, final_text)
        if verdict.ambiguous:
            from module3.runtime.events import make_clarification
            emitted["clarification"] = True
            await rt.emit_output(make_clarification(
                event.session_id, rt.clock.now(), **verdict.to_clarification_kwargs()
            ))
            return  # correctly refrains from also emitting a tool call
        evt, _ = make_tool_call(event.session_id, rt.clock.now(), "set_power",
                                 arguments={"device": "lamp"})
        emitted["tool_call"] = True
        await rt.emit_output(evt)

    rt.register_handler(InputEventType.END_OF_TURN, module2_end_of_turn)

    await rt.submit_event(make_end_of_turn(sid, rt.clock.now(), final_text="turn it on"))
    await rt.input_queue.join()
    await asyncio.sleep(0)

    # Missing "device" slot -> ambiguous -> clarification only, no tool call.
    assert emitted["clarification"] is True
    assert emitted["tool_call"] is False

    await rt.stop()


@pytest.mark.asyncio
async def test_defensive_handler_alone_can_race_with_module2():
    """Demonstrates the failure mode described in adapter.py's module docstring: with
    ONLY the fan-out handler pattern (no direct call), Module 4's clarification and
    Module 2's tool call can both fire for the same ambiguous turn."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    m4 = Module4Adapter(rt, caption_fn_async=fake_caption_fn)
    m4.install()  # registers the DEFENSIVE end-of-turn handler
    sid = rt.create_session()
    await rt.submit_event(make_tool_manifest(sid, rt.clock.now(), tools=[TOOL_SET_POWER]))
    await rt.input_queue.join()

    # Module 2 stand-in that does NOT know about Module 4 at all — the realistic case if
    # the two modules are built independently without the coordination this file argues for.
    async def naive_module2_end_of_turn(event):
        evt, _ = make_tool_call(event.session_id, rt.clock.now(), "set_power",
                                 arguments={"device": "lamp"})
        await rt.emit_output(evt)

    rt.register_handler(InputEventType.END_OF_TURN, naive_module2_end_of_turn)

    await rt.submit_event(make_end_of_turn(sid, rt.clock.now(), final_text="turn it on"))
    await rt.input_queue.join()
    await asyncio.sleep(0)

    outputs = await rt.drain_outputs()
    event_types = [o.event_type.value for o in outputs]

    # Both fired for the SAME ambiguous turn — this is the bug, reproduced, not asserted
    # from a docstring. If your team ships only the defensive handler, this is what your
    # judges' trace log will look like too.
    assert "CLARIFICATION" in event_types
    assert "TOOL_CALL" in event_types

    await rt.stop()
