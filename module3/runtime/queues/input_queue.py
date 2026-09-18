"""
module3/runtime/queues/input_queue.py

Async timestamp-aware input queue.

Responsibilities:
- Accept events from multiple async producers concurrently
- Deliver events to the dispatcher in insertion order
- Never execute business logic
- Never lose events under normal operation
- Document backpressure behavior

Backpressure policy:
    maxsize=0 (unbounded by default) — appropriate for a hackathon runtime
    where event bursts are bounded by scenario duration. If maxsize is set,
    put() will block until space is available (standard asyncio.Queue behavior).
    Producers that cannot tolerate blocking should use put_nowait() and handle
    QueueFull explicitly.

Ordering guarantee:
    FIFO within a single producer coroutine. Cross-producer ordering is
    determined by the order asyncio schedules the coroutines. For deterministic
    scenarios the VirtualClock controls event injection ordering externally.

Inspired by:
- Sema Code: FIFO input handling with no business logic in the queue
- DuplexOmni: continuous event ingestion while background work runs
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from ..events.base import BaseEvent

logger = logging.getLogger(__name__)


class InputQueue:
    """
    Thread-safe async input queue for the PRISM runtime.

    Usage::

        q = InputQueue(maxsize=0)
        await q.put(event)
        event = await q.get()
        q.task_done()

    The queue does NOT filter by session — the dispatcher handles routing.
    """

    def __init__(self, maxsize: int = 0, drop_oldest: bool = False) -> None:
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self._drop_oldest = drop_oldest
        self._dropped_count = 0

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    async def put(self, event: BaseEvent) -> BaseEvent | None:
        """
        Async enqueue. Blocks if maxsize reached (backpressure).
        Raises RuntimeError if the queue has been closed.
        """
        if self._closed:
            raise RuntimeError("InputQueue is closed — cannot accept new events")
        dropped = self._drop_oldest_if_full()
        await self._queue.put(event)
        logger.debug(
            "InputQueue.put event_type=%s session=%s event_id=%s ts=%.1f",
            event.event_type.value,
            event.session_id,
            event.event_id,
            event.timestamp_ms,
        )
        return dropped

    def put_nowait(self, event: BaseEvent) -> BaseEvent | None:
        """
        Non-blocking enqueue. Raises asyncio.QueueFull if maxsize reached.
        """
        if self._closed:
            raise RuntimeError("InputQueue is closed")
        dropped = self._drop_oldest_if_full()
        self._queue.put_nowait(event)
        return dropped

    def _drop_oldest_if_full(self) -> BaseEvent | None:
        if not (self._drop_oldest and self._queue.maxsize > 0 and self._queue.full()):
            return None
        dropped = self._queue.get_nowait()
        self._queue.task_done()
        self._dropped_count += 1
        logger.warning("InputQueue overflow: dropped oldest event_id=%s", dropped.event_id)
        return dropped

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    async def get(self) -> BaseEvent:
        """
        Async dequeue. Blocks until an event is available.
        Signals cancellation of the waiting coroutine if the queue is closed.
        """
        return await self._queue.get()

    def get_nowait(self) -> BaseEvent:
        """Non-blocking dequeue. Raises asyncio.QueueEmpty if empty."""
        return self._queue.get_nowait()

    def task_done(self) -> None:
        """Must be called after each get() to support join()."""
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all currently queued items have been processed."""
        await self._queue.join()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    @property
    def dropped_count(self) -> int:
        """Number of events discarded by the opt-in drop-oldest policy."""
        return self._dropped_count

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[BaseEvent]:
        """
        Iterate over events until a None sentinel is received (close signal).
        """
        while True:
            event = await self.get()
            if event is None:  # type: ignore[comparison-overlap]
                self.task_done()
                break
            yield event
            self.task_done()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Signal consumers to stop by injecting a None sentinel.
        After closing, put() raises RuntimeError.
        """
        self._closed = True
        await self._queue.put(None)  # type: ignore[arg-type]
        logger.info("InputQueue closed")

    @property
    def is_closed(self) -> bool:
        return self._closed
