"""
module3/runtime/queues/priority_input_queue.py

Two-lane priority input queue: control events are always delivered before
data events, regardless of insertion order.

Responsibilities:
- Accept events from multiple async producers concurrently
- Deliver control events (e.g. INTERRUPTION) before data events (FIFO within
  each lane)
- Support optional bounded data lane with drop-oldest overflow policy
- Never drop control events due to overflow
- Signal close only after all pending control events are drained

Lane definitions:
    Control lane  — event types in _CONTROL_TYPES (currently INTERRUPTION)
    Data lane     — all other InputEventType values

Backpressure policy:
    Control lane is always unbounded.
    Data lane respects data_maxsize (0 = unbounded). If data_maxsize is set
    and drop_oldest_data is False, put() blocks until space is available.
    If drop_oldest_data is True, the oldest data event is silently discarded
    to make room, and dropped_count is incremented.

Ordering guarantee:
    FIFO within each lane. Control events always precede data events in get().
    Close sentinel is returned only after all control events are consumed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from ..events.base import BaseEvent, InputEventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lane membership — extend this frozenset to promote more event types to the
# control lane without touching any other code.
# ---------------------------------------------------------------------------

_CONTROL_TYPES: frozenset[InputEventType] = frozenset({
    InputEventType.INTERRUPTION,
})


class PriorityInputQueue:
    """
    Async two-lane input queue for the PRISM runtime.

    Control events (INTERRUPTION, …) are always dequeued before data events.

    Usage::

        q = PriorityInputQueue()
        await q.put(event)          # routes automatically to the correct lane
        event = await q.get()       # control lane first, then data lane
        await q.close()             # drains control lane before sentinel

    Constructor arguments:
        data_maxsize      int  – max capacity of the data lane (0 = unbounded)
        drop_oldest_data  bool – if True, oldest data event is dropped on
                                 overflow instead of blocking
    """

    def __init__(
        self,
        *,
        data_maxsize: int = 0,
        drop_oldest_data: bool = False,
    ) -> None:
        # Control lane is always unbounded — we never want to block/drop INTERRUPTION
        self._control: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self._data: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=data_maxsize)
        self._data_maxsize = data_maxsize
        self._drop_oldest_data = drop_oldest_data
        self._dropped_count = 0
        self._closed = False

        # Condition used to wake get() waiters whenever either lane gains
        # an item or the queue is closed.
        self._ready = asyncio.Condition()

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    async def put(self, event: BaseEvent) -> None:
        """
        Route and enqueue an event.

        Raises RuntimeError if the queue has been closed.
        Blocks on the data lane if data_maxsize is set and drop_oldest_data
        is False and the data lane is full.
        """
        if self._closed:
            raise RuntimeError("PriorityInputQueue is closed — cannot accept new events")

        if event.event_type in _CONTROL_TYPES:
            await self._control.put(event)
            logger.debug(
                "PriorityInputQueue.put [CONTROL] event_type=%s session=%s event_id=%s",
                event.event_type.value,
                event.session_id,
                event.event_id,
            )
        else:
            # Apply drop-oldest policy if needed before blocking
            self._maybe_drop_oldest_data()
            await self._data.put(event)
            logger.debug(
                "PriorityInputQueue.put [DATA] event_type=%s session=%s event_id=%s",
                event.event_type.value,
                event.session_id,
                event.event_id,
            )

        async with self._ready:
            self._ready.notify_all()

    def _maybe_drop_oldest_data(self) -> None:
        """Drop the oldest data event if the lane is full and policy allows it."""
        if (
            self._drop_oldest_data
            and self._data_maxsize > 0
            and self._data.full()
        ):
            try:
                dropped = self._data.get_nowait()
                self._data.task_done()
                self._dropped_count += 1
                logger.warning(
                    "PriorityInputQueue data overflow: dropped oldest event_id=%s",
                    dropped.event_id,
                )
            except asyncio.QueueEmpty:
                pass  # Race: queue was drained between full() check and get_nowait()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    async def get(self) -> BaseEvent:
        """
        Dequeue the next event, preferring the control lane.

        Blocks until an event (or the close sentinel) is available.
        Returns None as the close sentinel once the queue is closed and both
        lanes are empty.
        """
        async with self._ready:
            while True:
                # Prefer control lane
                if not self._control.empty():
                    event = self._control.get_nowait()
                    self._control.task_done()
                    return event

                # Fall back to data lane
                if not self._data.empty():
                    event = self._data.get_nowait()
                    self._data.task_done()
                    return event

                # Both lanes empty: if closed, return sentinel
                if self._closed:
                    return None  # type: ignore[return-value]

                # Wait for a producer to signal
                await self._ready.wait()

    def task_done(self) -> None:
        """No-op: task accounting is handled inside get()."""

    async def join(self) -> None:
        """Block until both lanes are fully drained."""
        await self._control.join()
        await self._data.join()

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[BaseEvent]:
        """
        Iterate over events until the close sentinel (None) is received.
        """
        while True:
            event = await self.get()
            if event is None:
                break
            yield event

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Signal consumers to stop.

        The sentinel (None) is returned by get() only after all pending
        control events have been consumed, so late INTERRUPTION events are
        never silently discarded.
        """
        self._closed = True
        async with self._ready:
            self._ready.notify_all()
        logger.info("PriorityInputQueue closed")

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def qsize(self) -> int:
        """Total number of events waiting across both lanes."""
        return self._control.qsize() + self._data.qsize()

    def control_qsize(self) -> int:
        """Number of events waiting in the control lane."""
        return self._control.qsize()

    def data_qsize(self) -> int:
        """Number of events waiting in the data lane."""
        return self._data.qsize()

    def empty(self) -> bool:
        return self._control.empty() and self._data.empty()

    def full(self) -> bool:
        """True only when the data lane is full (control lane is always unbounded)."""
        return self._data_maxsize > 0 and self._data.full()

    @property
    def dropped_count(self) -> int:
        """Number of data events discarded by the drop-oldest-data policy."""
        return self._dropped_count
