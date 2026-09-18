"""
module3/runtime/tasks/cancellation.py

Generation-based cancellation coordinator.

This is the heart of the stale-result protection mechanism.

The core problem (from DuplexOmni-inspired architecture):
    1. Slow task A starts at generation=1, issues a tool call.
    2. User interrupts at T=450.
    3. Generation increments to 2.
    4. Tool result for task A arrives at T=700.
    5. This result is from generation=1, which is now superseded.
    6. The result must NOT be used to drive the response.
    7. A STALE_RESULT_DISCARDED internal event must be recorded.

The CancellationCoordinator:
- Maintains the current generation per session
- Provides increment_generation() — called on interruption
- Provides is_stale(task) — true if task's generation < current
- Provides cancel_task() — logical + optional physical cancellation
- Works with the TaskRegistry to update task states

Design rule: physical asyncio.Task.cancel() is a best-effort mechanism.
Logical stale-result rejection based on generation is the authoritative guard.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from .lifecycle import ExecutionPhase, TaskStatus
from .registry import TaskRecord, TaskRegistry

logger = logging.getLogger(__name__)

# Optional trace callback type — allows runtime to record internal events
# without creating a circular dependency on TraceRecorder
TraceCallback = Callable[..., Awaitable[None]]


class CancellationCoordinator:
    """
    Generation-based cancellation coordinator for a single session.

    Manages:
    - Current generation counter
    - Logical cancellation requests
    - Stale-result detection
    - Optional physical asyncio task cancellation

    Usage::

        coord = CancellationCoordinator(session_id="s1", registry=registry)

        # On interruption:
        old_gen, new_gen = coord.increment_generation(reason="user_interruption")

        # Before publishing a task result:
        if coord.is_stale(task_record):
            # discard result
            ...

        # To explicitly cancel a specific task:
        await coord.cancel_task(task_id, timestamp_ms=452.0)
    """

    def __init__(
        self,
        session_id: str,
        registry: TaskRegistry,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        self._session_id = session_id
        self._registry = registry
        self._generation: int = 1
        self._trace_callback = trace_callback

    # ------------------------------------------------------------------
    # Generation management
    # ------------------------------------------------------------------

    @property
    def current_generation(self) -> int:
        return self._generation

    def increment_generation(self, reason: str = "interruption") -> tuple[int, int]:
        """
        Advance the generation counter. Returns (old_generation, new_generation).

        Called by the interruption handler when the user interrupts the agent.
        All tasks from the old generation become candidates for stale rejection.
        """
        old = self._generation
        self._generation += 1
        logger.info(
            "CancellationCoordinator: generation %d -> %d (%s) session=%s",
            old, self._generation, reason, self._session_id,
        )
        return old, self._generation

    # ------------------------------------------------------------------
    # Stale detection
    # ------------------------------------------------------------------

    def is_stale(self, task: TaskRecord) -> bool:
        """
        True if the task's generation is less than the current generation.
        A stale task's results should be discarded without being published.
        """
        return task.generation < self._generation

    def is_stale_by_id(self, task_id: str) -> bool:
        record = self._registry.get(task_id)
        return record is not None and self.is_stale(record)

    def is_stale_call(self, call_id: str) -> bool:
        """True if the tool call belongs to a stale generation."""
        record = self._registry.get_by_call_id(call_id)
        return record is not None and self.is_stale(record)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel_task(
        self,
        task_id: str,
        timestamp_ms: float = 0.0,
        reason: str = "interrupted",
    ) -> bool:
        """
        Request logical cancellation of a task.

        1. Transitions task to CANCELLATION_REQUESTED (if not terminal).
        2. Optionally physically cancels the asyncio.Task.
        3. Returns True if cancellation was applied.

        Note: physical cancellation is cooperative — the coroutine
        must handle asyncio.CancelledError to respond promptly.
        """
        record = self._registry.get(task_id)
        if record is None:
            logger.warning("CancellationCoordinator: cancel_task unknown task_id=%s", task_id)
            return False

        if record.status.is_terminal():
            logger.debug(
                "CancellationCoordinator: task_id=%s already terminal (%s), skip cancel",
                task_id, record.status.value,
            )
            return False

        try:
            self._registry.transition(
                task_id,
                TaskStatus.CANCELLATION_REQUESTED,
                timestamp_ms=timestamp_ms,
                phase=record.phase,
            )
        except Exception as exc:
            logger.error(
                "CancellationCoordinator: transition to CANCELLATION_REQUESTED failed for %s: %r",
                task_id, exc,
            )
            return False

        # Physical cancellation — best effort
        if record.asyncio_task and not record.asyncio_task.done():
            record.asyncio_task.cancel()
            logger.debug("CancellationCoordinator: physically cancelled asyncio task %s", task_id)

        logger.info(
            "CancellationCoordinator: cancel_requested task_id=%s gen=%d reason=%s ts=%.1f",
            task_id, record.generation, reason, timestamp_ms,
        )
        return True

    async def confirm_cancelled(
        self,
        task_id: str,
        timestamp_ms: float = 0.0,
        phase: ExecutionPhase | None = None,
    ) -> None:
        """
        Confirm that a task has been cancelled (transition to CANCELLED).
        Called after the asyncio task exits or cancellation is acknowledged.
        """
        record = self._registry.get(task_id)
        if record is None:
            return
        if record.status == TaskStatus.CANCELLATION_REQUESTED:
            self._registry.transition(
                task_id,
                TaskStatus.CANCELLED,
                timestamp_ms=timestamp_ms,
                phase=phase,
            )
            logger.info(
                "CancellationCoordinator: confirmed cancelled task_id=%s gen=%d ts=%.1f",
                task_id, record.generation, timestamp_ms,
            )

    async def cancel_all_active(
        self,
        timestamp_ms: float = 0.0,
        reason: str = "session_interrupt",
    ) -> list[str]:
        """
        Cancel all non-terminal tasks in this session's registry.
        Returns list of task_ids that were cancelled.
        """
        cancelled = []
        for record in self._registry.active_tasks():
            if await self.cancel_task(record.task_id, timestamp_ms=timestamp_ms, reason=reason):
                cancelled.append(record.task_id)
        return cancelled

    # ------------------------------------------------------------------
    # Stale result handling
    # ------------------------------------------------------------------

    async def handle_stale_result(
        self,
        task_id: str,
        call_id: str | None,
        timestamp_ms: float = 0.0,
    ) -> None:
        """
        Mark task as STALE and trigger the trace callback so the
        STALE_RESULT_DISCARDED internal event is recorded.

        This is the authoritative path for stale-result rejection.
        Every discarded result MUST go through here to maintain trace integrity.
        """
        record = self._registry.get(task_id)
        if record is None:
            logger.warning(
                "CancellationCoordinator: handle_stale_result for unknown task_id=%s", task_id
            )
            return

        # A late result can be delivered more than once by an external tool.
        # Record the authoritative discard once; repeats are safe no-ops.
        if record.stale_result_discarded:
            logger.debug(
                "CancellationCoordinator: stale result already discarded task_id=%s", task_id
            )
            return

        task_generation = record.generation
        current_generation = self._generation

        logger.warning(
            "CancellationCoordinator: STALE_RESULT_DISCARDED task_id=%s "
            "call_id=%s task_gen=%d current_gen=%d ts=%.1f",
            task_id, call_id, task_generation, current_generation, timestamp_ms,
        )

        # Transition task to STALE if it hasn't been terminated yet
        if not record.status.is_terminal():
            try:
                self._registry.transition(task_id, TaskStatus.STALE, timestamp_ms=timestamp_ms)
            except Exception as exc:
                logger.error("CancellationCoordinator: stale transition failed: %r", exc)

        record.stale_result_discarded = True

        # BUG-5 FIX: pass session_id directly so the runtime callback doesn't
        # need to do an O(N×M) reverse lookup across all sessions and tasks.
        if self._trace_callback:
            await self._trace_callback(
                session_id=self._session_id,
                task_id=task_id,
                call_id=call_id,
                task_generation=task_generation,
                current_generation=current_generation,
                timestamp_ms=timestamp_ms,
            )
