"""
module3/runtime/tasks/registry.py

TaskRegistry — tracks every background task in the runtime.

Every background execution unit has a unique task_id and a TaskRecord
that tracks its full lifecycle from creation to terminal state.

The registry is session-scoped (one per SessionContext) to ensure
session isolation. Cross-session task lookup is not supported.

Key design decisions:
1. All state mutations go through transition() which validates legal moves.
2. asyncio.Task handles are stored for physical cancellation support.
3. Generation field ties tasks to the causal epoch at creation time.
4. call_id field ties tool-calling tasks to their output TOOL_CALL events.

Inspired by:
- Structured Graph Harness: explicit task lifecycle with observable states
- Sema Code: background tasks must be observable
- DuplexOmni: tasks from old generations must be detectable and suppressible
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from .lifecycle import ExecutionPhase, InvalidTransitionError, TaskStatus, validate_transition

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    """
    Complete record of a single background task.

    Immutable fields (set at creation) vs mutable fields (updated during lifecycle).
    """
    # Immutable identity
    task_id: str
    session_id: str
    task_type: str          # e.g. "slow_reasoning", "tool_call", "multimodal"
    generation: int         # epoch counter at time of creation
    parent_task_id: str | None = None
    call_id: str | None = None          # tool call correlation
    correlation_id: str | None = None

    # Mutable lifecycle fields
    status: TaskStatus = field(default=TaskStatus.PENDING)
    phase: ExecutionPhase = field(default=ExecutionPhase.IDLE)

    # Timestamps (virtual or wall-clock milliseconds)
    created_ms: float = field(default=0.0)
    started_ms: float | None = field(default=None)
    completed_ms: float | None = field(default=None)
    cancelled_ms: float | None = field(default=None)
    failed_ms: float | None = field(default=None)
    stale_result_discarded: bool = field(default=False)

    # Physical asyncio task handle (not serialized)
    asyncio_task: asyncio.Task[Any] | None = field(default=None, repr=False)

    # Arbitrary metadata for module-specific info
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "task_type": self.task_type,
            "generation": self.generation,
            "parent_task_id": self.parent_task_id,
            "call_id": self.call_id,
            "correlation_id": self.correlation_id,
            "status": self.status.value,
            "phase": self.phase.value,
            "created_ms": self.created_ms,
            "started_ms": self.started_ms,
            "completed_ms": self.completed_ms,
            "cancelled_ms": self.cancelled_ms,
            "failed_ms": self.failed_ms,
            "stale_result_discarded": self.stale_result_discarded,
            "metadata": self.metadata,
        }


class TaskRegistry:
    """
    Tracks all tasks within a session.

    The registry enforces legal state transitions and provides
    lookup by task_id and call_id.

    Usage::

        registry = TaskRegistry(session_id="s1")
        record = registry.create_task(
            task_type="slow_reasoning",
            generation=1,
            created_ms=0.0,
        )
        registry.transition(record.task_id, TaskStatus.RUNNING, started_ms=10.0)
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._tasks: dict[str, TaskRecord] = {}
        # call_id -> task_id for fast lookup when tool results arrive
        self._call_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_task(
        self,
        task_type: str,
        generation: int,
        created_ms: float = 0.0,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        call_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        """
        Register a new task. Returns the TaskRecord.

        The task starts in PENDING status.
        """
        tid = task_id or str(uuid.uuid4())
        record = TaskRecord(
            task_id=tid,
            session_id=self._session_id,
            task_type=task_type,
            generation=generation,
            parent_task_id=parent_task_id,
            call_id=call_id,
            correlation_id=correlation_id,
            status=TaskStatus.PENDING,
            phase=ExecutionPhase.IDLE,
            created_ms=created_ms,
            metadata=metadata or {},
        )
        self._tasks[tid] = record
        if call_id:
            self._call_index[call_id] = tid
        logger.debug(
            "TaskRegistry: created task_id=%s type=%s gen=%d session=%s",
            tid, task_type, generation, self._session_id,
        )
        return record

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        task_id: str,
        new_status: TaskStatus,
        timestamp_ms: float = 0.0,
        phase: ExecutionPhase | None = None,
    ) -> TaskRecord:
        """
        Transition a task to new_status.

        Raises:
            KeyError: if task_id is not registered
            InvalidTransitionError: if the transition is illegal
        """
        record = self._get_or_raise(task_id)
        validate_transition(task_id, record.status, new_status)

        old_status = record.status
        record.status = new_status
        if phase is not None:
            record.phase = phase

        # Update relevant timestamp
        if new_status == TaskStatus.RUNNING and record.started_ms is None:
            record.started_ms = timestamp_ms
        elif new_status in (TaskStatus.COMPLETED, TaskStatus.STALE):
            record.completed_ms = timestamp_ms
        elif new_status == TaskStatus.CANCELLED:
            record.cancelled_ms = timestamp_ms
        elif new_status == TaskStatus.FAILED:
            record.failed_ms = timestamp_ms

        logger.debug(
            "TaskRegistry: task_id=%s %s -> %s (gen=%d ts=%.1f)",
            task_id, old_status.value, new_status.value, record.generation, timestamp_ms,
        )
        return record

    def set_phase(self, task_id: str, phase: ExecutionPhase) -> None:
        """Update the execution phase without changing status."""
        record = self._get_or_raise(task_id)
        record.phase = phase

    def attach_asyncio_task(self, task_id: str, asyncio_task: asyncio.Task[Any]) -> None:
        """Store the asyncio.Task handle for physical cancellation."""
        self._get_or_raise(task_id).asyncio_task = asyncio_task

    def register_call_id(self, task_id: str, call_id: str) -> None:
        """Register a call_id → task_id mapping after task creation."""
        self._get_or_raise(task_id).call_id = call_id
        self._call_index[call_id] = task_id

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def get_by_call_id(self, call_id: str) -> TaskRecord | None:
        task_id = self._call_index.get(call_id)
        return self._tasks.get(task_id) if task_id else None

    def all_tasks(self) -> list[TaskRecord]:
        return list(self._tasks.values())

    def active_tasks(self) -> list[TaskRecord]:
        """Tasks that are not in a terminal state."""
        return [t for t in self._tasks.values() if not t.status.is_terminal()]

    def tasks_by_generation(self, generation: int) -> list[TaskRecord]:
        return [t for t in self._tasks.values() if t.generation == generation]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, task_id: str) -> TaskRecord:
        record = self._tasks.get(task_id)
        if record is None:
            raise KeyError(f"Task {task_id!r} not found in registry for session {self._session_id!r}")
        return record

    def __len__(self) -> int:
        return len(self._tasks)
