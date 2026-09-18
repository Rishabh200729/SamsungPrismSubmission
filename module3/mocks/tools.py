"""
module3/mocks/tools.py

Deterministic mock async tools for scenario testing.

These mocks simulate real tool behavior (latency, failures, retries, chaining)
without connecting to any real network APIs. Their purpose is to stress the
runtime's cancellation and stale-result machinery.

All tools accept a Clock instance so they can use virtual time instead of
real asyncio.sleep() in deterministic mode.

Available mocks:
    mock_search()          — simulated search with configurable latency
    mock_booking()         — simulated booking with optional failure
    mock_ticket_creation() — simulated ticket creation (chained after booking)
    mock_frame_lookup()    — simulated video frame analysis
    mock_failing_tool()    — always fails after delay
    mock_retry_tool()      — fails N times then succeeds
    mock_slow_tool()       — very slow, designed to be interrupted
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ..runtime.clock.base import Clock
from ..runtime.clock.real_clock import RealClock

logger = logging.getLogger(__name__)

_DEFAULT_CLOCK = RealClock()


@dataclass
class ToolResult:
    """Structured result from any mock tool."""
    tool_name: str
    success: bool
    data: dict[str, Any]
    latency_ms: float
    error: str | None = None


async def mock_search(
    query: str,
    latency_ms: float = 200.0,
    clock: Clock = _DEFAULT_CLOCK,
    fail: bool = False,
) -> ToolResult:
    """
    Simulate a search tool with configurable latency.

    Args:
        query:      The search query string.
        latency_ms: Simulated execution time.
        clock:      Use VirtualClock in tests for deterministic timing.
        fail:       If True, raises a simulated error.
    """
    logger.debug("mock_search: starting query=%r latency=%.0fms", query, latency_ms)
    await clock.sleep(latency_ms)

    if fail:
        raise RuntimeError(f"mock_search: simulated failure for query={query!r}")

    result = ToolResult(
        tool_name="search",
        success=True,
        data={
            "query": query,
            "results": [
                {"title": f"Result 1 for '{query}'", "url": "https://example.com/1"},
                {"title": f"Result 2 for '{query}'", "url": "https://example.com/2"},
            ],
            "total": 2,
        },
        latency_ms=latency_ms,
    )
    logger.debug("mock_search: completed query=%r", query)
    return result


async def mock_booking(
    origin: str,
    destination: str,
    date: str,
    latency_ms: float = 300.0,
    clock: Clock = _DEFAULT_CLOCK,
    fail: bool = False,
    fail_reason: str = "no_availability",
) -> ToolResult:
    """
    Simulate a flight booking tool.

    Designed to take long enough (300ms+) to be interrupted by the
    stale-result test scenarios.
    """
    logger.debug("mock_booking: %s -> %s on %s", origin, destination, date)
    await clock.sleep(latency_ms)

    if fail:
        return ToolResult(
            tool_name="booking",
            success=False,
            data={"origin": origin, "destination": destination, "date": date},
            latency_ms=latency_ms,
            error=fail_reason,
        )

    return ToolResult(
        tool_name="booking",
        success=True,
        data={
            "booking_id": f"BK-{hash((origin, destination, date)) % 100000:05d}",
            "origin": origin,
            "destination": destination,
            "date": date,
            "status": "confirmed",
            "price_usd": 450.00,
        },
        latency_ms=latency_ms,
    )


async def mock_ticket_creation(
    booking_id: str,
    passenger_name: str,
    latency_ms: float = 150.0,
    clock: Clock = _DEFAULT_CLOCK,
) -> ToolResult:
    """
    Simulate ticket creation — typically chained after a successful booking.
    Tests chained tool call correlation.
    """
    logger.debug("mock_ticket_creation: booking_id=%s", booking_id)
    await clock.sleep(latency_ms)

    return ToolResult(
        tool_name="ticket_creation",
        success=True,
        data={
            "ticket_id": f"TKT-{hash(booking_id) % 100000:05d}",
            "booking_id": booking_id,
            "passenger": passenger_name,
            "status": "issued",
        },
        latency_ms=latency_ms,
    )


async def mock_frame_lookup(
    frame_index: int,
    query: str = "describe",
    latency_ms: float = 100.0,
    clock: Clock = _DEFAULT_CLOCK,
) -> ToolResult:
    """
    Simulate video frame analysis (Module 4 multimodal path).
    """
    logger.debug("mock_frame_lookup: frame=%d query=%r", frame_index, query)
    await clock.sleep(latency_ms)

    return ToolResult(
        tool_name="frame_lookup",
        success=True,
        data={
            "frame_index": frame_index,
            "description": f"Frame {frame_index}: Simulated scene analysis for '{query}'",
            "objects_detected": ["person", "background"],
            "confidence": 0.95,
        },
        latency_ms=latency_ms,
    )


async def mock_failing_tool(
    reason: str = "tool_error",
    latency_ms: float = 100.0,
    clock: Clock = _DEFAULT_CLOCK,
) -> ToolResult:
    """Always fails after the given latency. Tests TASK_FAILED path."""
    await clock.sleep(latency_ms)
    raise RuntimeError(f"mock_failing_tool: {reason}")


async def mock_retry_tool(
    fail_times: int = 2,
    latency_ms: float = 100.0,
    clock: Clock = _DEFAULT_CLOCK,
    _attempt_state: list[int] | None = None,
) -> ToolResult:
    """
    Fails for the first `fail_times` calls, then succeeds.
    Use _attempt_state=[0] to share state across retries.
    """
    state = _attempt_state if _attempt_state is not None else [0]
    state[0] += 1
    attempt = state[0]

    await clock.sleep(latency_ms)

    if attempt <= fail_times:
        raise RuntimeError(f"mock_retry_tool: attempt {attempt}/{fail_times} failed (simulated)")

    return ToolResult(
        tool_name="retry_tool",
        success=True,
        data={"attempt": attempt, "message": "Succeeded after retries"},
        latency_ms=latency_ms * attempt,
    )


async def mock_slow_tool(
    latency_ms: float = 800.0,
    clock: Clock = _DEFAULT_CLOCK,
    label: str = "slow",
) -> ToolResult:
    """
    Very slow tool — designed to be interrupted by scenario tests.
    Any result from this tool arriving after an interruption should
    be marked STALE by the cancellation coordinator.
    """
    logger.debug("mock_slow_tool: starting label=%s latency=%.0fms", label, latency_ms)
    await clock.sleep(latency_ms)
    logger.debug("mock_slow_tool: completed label=%s", label)

    return ToolResult(
        tool_name="slow_tool",
        success=True,
        data={"label": label, "message": "Slow tool completed"},
        latency_ms=latency_ms,
    )


async def mock_instant_tool(label: str = "instant") -> ToolResult:
    """Zero-latency tool for testing immediate results."""
    await asyncio.sleep(0)  # Yield to event loop
    return ToolResult(
        tool_name="instant_tool",
        success=True,
        data={"label": label, "message": "Instant result"},
        latency_ms=0.0,
    )
