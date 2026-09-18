"""
module3/runtime/tasks/lifecycle.py

Explicit task state machine.

Inspired by the structured graph harness / scheduler-theoretic paper:
"Do not turn the whole project into a giant implicit agent loop.
Prefer explicit runtime state and explicit task lifecycle."

States and valid transitions are defined here. The registry enforces them.
No task can skip states or transition backwards without explicit allowance.

State meanings:
    PENDING               - Created, waiting for execution slot
    RUNNING               - Actively executing
    WAITING               - Blocked on external result (tool call, etc.)
    CANCELLATION_REQUESTED - Cancellation requested but not yet confirmed
    COMPLETED             - Finished successfully
    CANCELLED             - Cancellation confirmed
    FAILED                - Terminated due to error
    STALE                 - Completed but result discarded (old generation)

Execution phase meanings (for phase-aware interrupt handling per spec §12):
    PLANNING              - Agent forming a plan, no tool calls issued
    TOOL_RUNNING          - A tool call has been issued and is in flight
    WAITING_FOR_RESULT    - Waiting for async tool result
    RESPONSE_EMISSION     - Assembling and emitting response
    IDLE                  - Between tasks
"""

from __future__ import annotations

from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle state of a background task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    STALE = "STALE"

    def is_terminal(self) -> bool:
        """True if no further transitions are possible."""
        return self in _TERMINAL_STATES

    def can_transition_to(self, target: "TaskStatus") -> bool:
        """Check whether a transition from self → target is legal."""
        return target in _TRANSITIONS.get(self, set())


class ExecutionPhase(str, Enum):
    """Fine-grained execution phase within a running task."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    TOOL_RUNNING = "TOOL_RUNNING"
    WAITING_FOR_RESULT = "WAITING_FOR_RESULT"
    RESPONSE_EMISSION = "RESPONSE_EMISSION"


# Terminal states — no further transitions allowed
_TERMINAL_STATES: frozenset[TaskStatus] = frozenset({
    TaskStatus.COMPLETED,
    TaskStatus.CANCELLED,
    TaskStatus.FAILED,
    TaskStatus.STALE,
})

# Legal transition table
# From state -> set of allowable next states
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {
        TaskStatus.RUNNING,
        TaskStatus.CANCELLATION_REQUESTED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RUNNING: {
        TaskStatus.WAITING,
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLATION_REQUESTED,
        TaskStatus.FAILED,
        TaskStatus.STALE,    # Direct stale: result arrived from old generation
    },
    TaskStatus.WAITING: {
        TaskStatus.RUNNING,
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLATION_REQUESTED,
        TaskStatus.FAILED,
        TaskStatus.STALE,    # Waiting task's result is from old generation
    },
    TaskStatus.CANCELLATION_REQUESTED: {
        TaskStatus.CANCELLED,
        TaskStatus.STALE,
        # BUG-7 FIX: COMPLETED removed. A task that receives a cancellation
        # request must not appear as COMPLETED in the trace \u2014 it masks the
        # cancellation from metrics. The task must transition to CANCELLED.
    },
    # Terminal states have empty sets (no transitions allowed)
    TaskStatus.COMPLETED: set(),
    TaskStatus.CANCELLED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.STALE: set(),
}


class InvalidTransitionError(Exception):
    """Raised when an invalid task state transition is attempted."""

    def __init__(self, task_id: str, from_state: TaskStatus, to_state: TaskStatus) -> None:
        super().__init__(
            f"Invalid transition for task {task_id}: {from_state.value} -> {to_state.value}. "
            f"Allowed from {from_state.value}: {[s.value for s in _TRANSITIONS.get(from_state, set())]}"
        )
        self.task_id = task_id
        self.from_state = from_state
        self.to_state = to_state


def validate_transition(task_id: str, from_state: TaskStatus, to_state: TaskStatus) -> None:
    """
    Raise InvalidTransitionError if the transition is illegal.
    Call this before changing task status in the registry.
    """
    if from_state.is_terminal():
        raise InvalidTransitionError(task_id, from_state, to_state)
    if not from_state.can_transition_to(to_state):
        raise InvalidTransitionError(task_id, from_state, to_state)
