"""
module3/runtime/clock/virtual_clock.py

Deterministic VirtualClock for scenario testing.

This is the key enabler for reproducible evaluation (inspired by Gaia2).

The VirtualClock does NOT use real asyncio.sleep(). Instead it:
1. Tracks the current virtual time in milliseconds.
2. Maintains a priority queue of scheduled callbacks/events.
3. Advances time explicitly via advance() or advance_to().
4. Fires scheduled callbacks in timestamp order.
5. Provides run_until_idle() to flush all pending work.

The ScenarioRunner uses the VirtualClock to inject events at deterministic
timestamps, making timing-sensitive tests fully reproducible:

    clock = VirtualClock()
    clock.schedule(lambda: inject_text_chunk(), at_ms=0)
    clock.schedule(lambda: inject_interruption(), at_ms=450)
    clock.schedule(lambda: inject_tool_result(), at_ms=700)
    await clock.run_until_idle()

Design: cooperative — awaiting tasks must also check the clock's time
rather than using asyncio.sleep(). Mock tools use the VirtualClock
directly for their simulated delays.

The VirtualClock also provides a sleep() method compatible with the
Clock interface, so tools and background coroutines can use it:

    await clock.sleep(200)  # advances virtual time by 200ms

IMPORTANT: VirtualClock is single-threaded (asyncio). Do NOT call
advance() from multiple coroutines concurrently without external locking.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from typing import Any, Awaitable, Callable

from .base import Clock

logger = logging.getLogger(__name__)

# Callback type: either sync or async callable with no args
Callback = Callable[[], Any]


class _ScheduledItem:
    """A callback scheduled to fire at a specific virtual time."""

    def __init__(self, at_ms: float, seq: int, callback: Callback) -> None:
        self.at_ms = at_ms
        self.seq = seq  # tiebreaker for same timestamp
        self.callback = callback

    def __lt__(self, other: "_ScheduledItem") -> bool:
        return (self.at_ms, self.seq) < (other.at_ms, other.seq)


class VirtualClock(Clock):
    """
    Deterministic virtual clock for scenario testing.

    Maintains a priority queue of scheduled callbacks and fires them
    in timestamp order as virtual time advances.
    """

    def __init__(self, start_ms: float = 0.0) -> None:
        self._current_ms: float = start_ms
        self._heap: list[_ScheduledItem] = []
        self._seq: int = 0
        # asyncio.Event for unblocking sleepers when time advances past their target
        self._sleepers: list[tuple[float, asyncio.Event]] = []

    # ------------------------------------------------------------------
    # Clock interface
    # ------------------------------------------------------------------

    def now(self) -> float:
        """Current virtual time in milliseconds."""
        return self._current_ms

    async def sleep(self, duration_ms: float) -> None:
        """
        Virtual sleep: registers the caller as a sleeper and suspends
        until virtual time has advanced by at least duration_ms.

        Compatible with the Clock interface — background tasks can
        call this instead of asyncio.sleep().
        """
        target = self._current_ms + duration_ms
        event = asyncio.Event()
        heapq.heappush(self._sleepers, (target, event))  # type: ignore[misc]
        await event.wait()

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def advance(self, delta_ms: float) -> None:
        """
        Advance virtual time by delta_ms and fire all callbacks
        scheduled in [current_ms, current_ms + delta_ms].
        """
        if delta_ms < 0:
            raise ValueError(f"Cannot advance by negative delta: {delta_ms}")
        target = self._current_ms + delta_ms
        self.advance_to(target)

    def advance_to(self, target_ms: float) -> None:
        """
        Advance virtual time to target_ms (absolute).
        Fires all scheduled callbacks up to and including target_ms.
        """
        if target_ms < self._current_ms:
            logger.warning(
                "VirtualClock.advance_to: target %.1f < current %.1f — no-op",
                target_ms, self._current_ms,
            )
            return

        # Fire scheduled callbacks in order
        while self._heap and self._heap[0].at_ms <= target_ms:
            item = heapq.heappop(self._heap)
            self._current_ms = item.at_ms
            logger.debug("VirtualClock: firing scheduled item at %.1fms", item.at_ms)
            try:
                result = item.callback()
                # If the callback returns a coroutine, we can't await here
                # (advance_to is sync). Callers that need async should use
                # schedule_async() + await run_until_idle().
                if asyncio.iscoroutine(result):
                    logger.warning(
                        "VirtualClock: async callback scheduled via advance_to() — "
                        "use run_until_idle() for async callbacks. Scheduling as task."
                    )
                    asyncio.ensure_future(result)
            except Exception as exc:
                logger.error("VirtualClock: callback raised %r", exc)

        self._current_ms = target_ms
        self._wake_sleepers()

    async def advance_async(self, delta_ms: float) -> None:
        """
        Async version of advance that yields to the event loop after
        firing each callback, allowing awaited background coroutines to run.
        """
        target = self._current_ms + delta_ms
        await self.advance_to_async(target)

    async def advance_to_async(self, target_ms: float) -> None:
        """
        Advance virtual time asynchronously, yielding between callbacks
        so that background asyncio tasks can run.
        """
        if target_ms < self._current_ms:
            return

        while self._heap and self._heap[0].at_ms <= target_ms:
            item = heapq.heappop(self._heap)
            self._current_ms = item.at_ms
            self._wake_sleepers()
            try:
                result = item.callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("VirtualClock: async callback raised %r", exc)
            # Yield to event loop so background tasks can progress
            await asyncio.sleep(0)

        self._current_ms = target_ms
        self._wake_sleepers()
        await asyncio.sleep(0)  # Final yield

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(self, callback: Callback, at_ms: float) -> int:
        """
        Schedule callback to fire when virtual time reaches at_ms.

        Returns the sequence number (useful for cancellation in future).
        """
        seq = self._next_seq()
        heapq.heappush(self._heap, _ScheduledItem(at_ms=at_ms, seq=seq, callback=callback))
        logger.debug("VirtualClock: scheduled callback at %.1fms seq=%d", at_ms, seq)
        return seq

    def schedule_coro(self, coro_fn: Callable[[], Awaitable[Any]], at_ms: float) -> int:
        """
        Schedule an async callable to fire at at_ms.
        The coroutine is wrapped so it can be awaited when the item fires.
        """
        def _wrapper() -> Any:
            return coro_fn()
        return self.schedule(_wrapper, at_ms)

    # ------------------------------------------------------------------
    # Idle detection and run loop
    # ------------------------------------------------------------------

    async def run_until_idle(self) -> None:
        """
        Fire all remaining scheduled items and then return.

        After each item fires, yields to the event loop to allow
        background coroutines to run. Stops when the heap is empty
        and no sleepers are pending.
        """
        while self._heap or self._sleepers:
            if self._heap:
                next_ms = self._heap[0].at_ms
                await self.advance_to_async(next_ms)
            else:
                # Only sleepers left — no scheduled callbacks to advance to
                break
            await asyncio.sleep(0)

    def is_idle(self) -> bool:
        """True when no scheduled callbacks and no sleepers are pending."""
        return not self._heap and not self._sleepers

    def pending_count(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _wake_sleepers(self) -> None:
        """Wake all sleepers whose target time has been reached."""
        remaining = []
        for target, event in self._sleepers:
            if self._current_ms >= target:
                event.set()
            else:
                remaining.append((target, event))
        self._sleepers = remaining
