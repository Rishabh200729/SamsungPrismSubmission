"""
module3/tests/test_priority_queue.py

Unit tests for PriorityInputQueue — the two-lane control/data queue.
"""

from __future__ import annotations

import asyncio
import pytest

from module3.runtime.queues.priority_input_queue import PriorityInputQueue, _CONTROL_TYPES
from module3.runtime.events import (
    make_interruption, make_text_chunk, make_video_frame, make_audio_wav,
    make_tool_result,
)
from module3.runtime.events.base import InputEventType

SID = "piq-test-session"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interruption(ts=0.0):
    return make_interruption(SID, ts, text="Stop")


def _text(ts=0.0):
    return make_text_chunk(SID, ts, "Hello")


def _video(ts=0.0, idx=0):
    return make_video_frame(SID, ts, "AA==", 1, 1, idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_piq_control_types_constant():
    assert InputEventType.INTERRUPTION in _CONTROL_TYPES


@pytest.mark.asyncio
async def test_piq_single_data_event():
    q = PriorityInputQueue()
    evt = _text(0.0)
    await q.put(evt)
    got = await q.get()
    assert got.event_type == InputEventType.TEXT_CHUNK
    await q.close()


@pytest.mark.asyncio
async def test_piq_single_control_event():
    q = PriorityInputQueue()
    evt = _interruption(0.0)
    await q.put(evt)
    got = await q.get()
    assert got.event_type == InputEventType.INTERRUPTION
    await q.close()


@pytest.mark.asyncio
async def test_piq_control_beats_data_regardless_of_order():
    """Interruption submitted after 5 data events must come out first."""
    q = PriorityInputQueue()
    for i in range(5):
        await q.put(_video(float(i * 10), idx=i))
    await q.put(_interruption(100.0))

    first = await q.get()
    assert first.event_type == InputEventType.INTERRUPTION, (
        f"Expected INTERRUPTION first, got {first.event_type.value}"
    )
    for _ in range(5):
        evt = await q.get()
        assert evt.event_type == InputEventType.VIDEO_FRAME

    await q.close()


@pytest.mark.asyncio
async def test_piq_multiple_control_events_in_order():
    """Multiple control events are yielded before any data, in FIFO order."""
    q = PriorityInputQueue()
    await q.put(_text(0.0))
    i1 = _interruption(1.0)
    i2 = _interruption(2.0)
    await q.put(i1)
    await q.put(i2)

    got1 = await q.get()
    got2 = await q.get()
    assert got1.event_type == InputEventType.INTERRUPTION
    assert got2.event_type == InputEventType.INTERRUPTION
    # Both interruptions come before the text
    got3 = await q.get()
    assert got3.event_type == InputEventType.TEXT_CHUNK

    await q.close()


@pytest.mark.asyncio
async def test_piq_close_sentinel_after_control():
    """Close sentinel must only appear after all control events are drained."""
    q = PriorityInputQueue()
    await q.put(_interruption(0.0))
    await q.close()

    first = await q.get()
    assert first.event_type == InputEventType.INTERRUPTION

    sentinel = await q.get()
    assert sentinel is None


@pytest.mark.asyncio
async def test_piq_qsize_reflects_both_lanes():
    q = PriorityInputQueue()
    await q.put(_text(0.0))
    await q.put(_video(1.0))
    await q.put(_interruption(2.0))

    assert q.qsize() == 3
    assert q.control_qsize() == 1
    assert q.data_qsize() == 2

    await q.close()


@pytest.mark.asyncio
async def test_piq_drop_oldest_data_under_overflow():
    """With drop_oldest_data=True and maxsize=2, oldest data event is dropped."""
    q = PriorityInputQueue(data_maxsize=2, drop_oldest_data=True)
    await q.put(_text(0.0))    # slot 1
    await q.put(_video(1.0))   # slot 2
    await q.put(_video(2.0, idx=1))  # slot 3 — must drop oldest data (text)

    assert q.dropped_count == 1
    # Control events are never dropped even at overflow
    await q.put(_interruption(3.0))
    assert q.dropped_count == 1  # No extra drop for control

    first = await q.get()
    assert first.event_type == InputEventType.INTERRUPTION

    await q.close()


@pytest.mark.asyncio
async def test_piq_raises_when_closed():
    q = PriorityInputQueue()
    await q.close()
    with pytest.raises(RuntimeError, match="closed"):
        await q.put(_text(10.0))


@pytest.mark.asyncio
async def test_piq_empty_initially():
    q = PriorityInputQueue()
    assert q.empty()
    assert q.qsize() == 0
    await q.close()
