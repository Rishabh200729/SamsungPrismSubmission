"""Deterministic tests for the Module 3 mock tool library."""

import asyncio

import pytest

from module3.mocks import (
    mock_booking,
    mock_failing_tool,
    mock_frame_lookup,
    mock_instant_tool,
    mock_retry_tool,
    mock_search,
    mock_slow_tool,
    mock_ticket_creation,
)
from module3.runtime.clock.virtual_clock import VirtualClock


async def _complete(coro, clock: VirtualClock, milliseconds: float):
    task = asyncio.create_task(coro)
    await asyncio.sleep(0)
    await clock.advance_to_async(milliseconds)
    return await task


async def test_search_success_failure_and_virtual_latency():
    clock = VirtualClock()
    result = await _complete(mock_search("Seoul", latency_ms=25, clock=clock), clock, 25)
    assert result.success and result.data["query"] == "Seoul"
    assert result.data["total"] == 2 and result.latency_ms == 25

    failed = asyncio.create_task(mock_search("Seoul", latency_ms=10, clock=clock, fail=True))
    await asyncio.sleep(0)
    await clock.advance_to_async(35)
    with pytest.raises(RuntimeError, match="simulated failure"):
        await failed


async def test_booking_ticket_and_frame_results():
    clock = VirtualClock()
    booking = await _complete(mock_booking("ICN", "NRT", "2030-01-01", latency_ms=30, clock=clock), clock, 30)
    assert booking.success and booking.data["booking_id"].startswith("BK-")

    failed_booking = await _complete(
        mock_booking("ICN", "NRT", "2030-01-01", latency_ms=10, clock=clock, fail=True, fail_reason="sold_out"),
        clock,
        40,
    )
    assert not failed_booking.success and failed_booking.error == "sold_out"

    ticket = await _complete(mock_ticket_creation(booking.data["booking_id"], "Ari", latency_ms=15, clock=clock), clock, 55)
    frame = await _complete(mock_frame_lookup(4, "describe", latency_ms=5, clock=clock), clock, 60)
    assert ticket.data["booking_id"] == booking.data["booking_id"]
    assert frame.data["frame_index"] == 4 and frame.data["confidence"] == 0.95


async def test_failing_retry_instant_and_slow_cancellation():
    clock = VirtualClock()
    failing = asyncio.create_task(mock_failing_tool("boom", latency_ms=10, clock=clock))
    await asyncio.sleep(0)
    await clock.advance_to_async(10)
    with pytest.raises(RuntimeError, match="boom"):
        await failing

    state = [0]
    for target in (20, 30):
        retry = asyncio.create_task(mock_retry_tool(fail_times=2, latency_ms=10, clock=clock, _attempt_state=state))
        await asyncio.sleep(0)
        await clock.advance_to_async(target)
        with pytest.raises(RuntimeError, match="attempt"):
            await retry
    result = await _complete(mock_retry_tool(fail_times=2, latency_ms=10, clock=clock, _attempt_state=state), clock, 40)
    assert result.success and result.data["attempt"] == 3 and result.latency_ms == 30

    instant = await mock_instant_tool("now")
    assert instant.latency_ms == 0 and instant.data["label"] == "now"

    slow = asyncio.create_task(mock_slow_tool(latency_ms=100, clock=clock, label="cancel-me"))
    await asyncio.sleep(0)
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
