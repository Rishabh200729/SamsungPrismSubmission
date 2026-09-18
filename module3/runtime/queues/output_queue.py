"""Priority-aware async output queue for externally observable events."""

from __future__ import annotations

import asyncio
import heapq
import logging
from typing import Any, AsyncIterator

from ..events.base import BaseEvent, EventCategory

logger = logging.getLogger(__name__)


class OutputQueue:
    """Stable priority queue; higher priority is delivered first.

    Filtering does not discard non-matching events. Among matching events the
    normal priority/FIFO ordering remains intact.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._maxsize = maxsize
        self._heap: list[tuple[int, int, BaseEvent]] = []
        self._condition = asyncio.Condition()
        self._closed = False
        self._total_emitted = 0
        self._sequence = 0

    async def emit(self, event: BaseEvent, *, priority: int = 0) -> None:
        if event.category != EventCategory.OUTPUT:
            logger.warning(
                "OutputQueue received non-OUTPUT event category=%s type=%s — routing anyway",
                event.category.value,
                event.event_type.value,
            )
        async with self._condition:
            if self._closed:
                raise RuntimeError("OutputQueue is closed — cannot emit")
            while self._maxsize > 0 and len(self._heap) >= self._maxsize:
                await self._condition.wait()
            self._push(event, priority)
            self._condition.notify_all()

    def emit_nowait(self, event: BaseEvent, *, priority: int = 0) -> None:
        if self._closed:
            raise RuntimeError("OutputQueue is closed")
        if self._maxsize > 0 and len(self._heap) >= self._maxsize:
            raise asyncio.QueueFull
        self._push(event, priority)
        try:
            asyncio.get_running_loop().create_task(self._notify_waiters())
        except RuntimeError:
            # No running loop means there cannot be an async consumer waiting.
            pass

    async def _notify_waiters(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def _push(self, event: BaseEvent, priority: int) -> None:
        self._sequence += 1
        heapq.heappush(self._heap, (-priority, self._sequence, event))
        self._total_emitted += 1

    async def next_output(
        self,
        event_type: Any | None = None,
        session_id: str | None = None,
        timeout_ms: float | None = None,
    ) -> BaseEvent:
        async def _get() -> BaseEvent:
            async with self._condition:
                while True:
                    index = self._matching_index(event_type, session_id)
                    if index is not None:
                        _, _, event = self._heap[index]
                        self._heap[index] = self._heap[-1]
                        self._heap.pop()
                        if self._heap:
                            heapq.heapify(self._heap)
                        self._condition.notify_all()
                        return event
                    if self._closed:
                        return None  # type: ignore[return-value]
                    await self._condition.wait()

        if timeout_ms is None:
            return await _get()
        return await asyncio.wait_for(_get(), timeout=timeout_ms / 1000.0)

    def _matching_index(self, event_type: Any | None, session_id: str | None) -> int | None:
        wanted_type = getattr(event_type, "value", event_type)
        candidates = [
            (item, index)
            for index, item in enumerate(self._heap)
            if (wanted_type is None or item[2].event_type.value == wanted_type)
            and (session_id is None or item[2].session_id == session_id)
        ]
        return min(candidates, default=(None, None))[1]

    def next_output_nowait(self) -> BaseEvent:
        index = self._matching_index(None, None)
        if index is None:
            raise asyncio.QueueEmpty
        _, _, event = self._heap[index]
        self._heap[index] = self._heap[-1]
        self._heap.pop()
        if self._heap:
            heapq.heapify(self._heap)
        return event

    def task_done(self) -> None:
        """Compatibility no-op: consumption is accounted for during dequeue."""

    async def drain_all(self) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        while True:
            try:
                events.append(self.next_output_nowait())
            except asyncio.QueueEmpty:
                return events

    async def collect_until_idle(self, timeout_ms: float = 100.0) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        while True:
            try:
                event = await self.next_output(timeout_ms=timeout_ms)
            except asyncio.TimeoutError:
                return events
            if event is None:
                return events
            events.append(event)

    async def __aiter__(self) -> AsyncIterator[BaseEvent]:
        while True:
            event = await self.next_output()
            if event is None:
                return
            yield event

    def qsize(self) -> int:
        return len(self._heap)

    def empty(self) -> bool:
        return not self._heap

    @property
    def total_emitted(self) -> int:
        return self._total_emitted

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        logger.info("OutputQueue closed (total emitted: %d)", self._total_emitted)

    @property
    def is_closed(self) -> bool:
        return self._closed
