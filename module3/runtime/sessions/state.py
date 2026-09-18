"""Authoritative, versioned session state for Theme 05."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FloorState(str, Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"


# Legal floor-state transitions: from -> {allowed destinations}
_LEGAL_TRANSITIONS: dict[FloorState, frozenset[FloorState]] = {
    FloorState.LISTENING: frozenset({
        FloorState.THINKING,
        FloorState.INTERRUPTED,
        FloorState.WAITING_FOR_USER,
    }),
    FloorState.THINKING: frozenset({
        FloorState.SPEAKING,
        FloorState.INTERRUPTED,
        FloorState.WAITING_FOR_TOOL,
        FloorState.WAITING_FOR_USER,
        FloorState.LISTENING,
    }),
    FloorState.SPEAKING: frozenset({
        FloorState.LISTENING,
        FloorState.INTERRUPTED,
        FloorState.WAITING_FOR_USER,
    }),
    FloorState.INTERRUPTED: frozenset({
        FloorState.LISTENING,
        FloorState.THINKING,
        FloorState.WAITING_FOR_USER,
    }),
    FloorState.WAITING_FOR_USER: frozenset({
        FloorState.LISTENING,
        FloorState.THINKING,
        FloorState.INTERRUPTED,
    }),
    FloorState.WAITING_FOR_TOOL: frozenset({
        FloorState.THINKING,
        FloorState.SPEAKING,
        FloorState.INTERRUPTED,
        FloorState.LISTENING,
    }),
}


@dataclass(frozen=True)
class SlotValue:
    value: Any
    source_event_id: str | None = None
    updated_at_ms: float = 0.0
    confidence: float | None = None


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    generation: int = 1
    snapshot_version: int = 1
    intent: str | None = None
    slots: dict[str, SlotValue] = field(default_factory=dict)
    floor_state: FloorState = FloorState.LISTENING
    pending_call_ids: tuple[str, ...] = ()
    completed_actions: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "slots": {k: v.value for k, v in self.slots.items()},
            "completed_actions": list(self.completed_actions),
            "pending_actions": list(self.pending_call_ids),
            "snapshot_version": self.snapshot_version,
            "generation": self.generation,
            "floor_state": self.floor_state.value,
        }


class SessionStateStore:
    """
    Single-writer state container. Callers must supply expected version.

    Floor-state transitions are validated against the legal transition table.
    An interruption always forces SPEAKING -> INTERRUPTED regardless of
    normal transition rules (handled by invalidate()).

    Filler throttling: at most one FILLER may be emitted per filler_window_ms.
    Callers must call can_emit_filler() before emitting a filler.
    """

    def __init__(
        self,
        session_id: str,
        filler_window_ms: float = 3000.0,
    ) -> None:
        self._snapshot = SessionSnapshot(session_id=session_id)
        self._filler_window_ms = filler_window_ms
        self._last_filler_ms: float | None = None

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot

    def patch_slots(
        self,
        expected_version: int,
        patch: dict[str, Any],
        *,
        source_event_id: str | None,
        timestamp_ms: float,
        intent: str | None = None,
    ) -> SessionSnapshot:
        if expected_version != self._snapshot.snapshot_version:
            raise ValueError("stale snapshot version")
        slots = dict(self._snapshot.slots)
        for key, value in patch.items():
            slots[key] = SlotValue(value, source_event_id, timestamp_ms)
        self._snapshot = replace(
            self._snapshot,
            slots=slots,
            intent=intent if intent is not None else self._snapshot.intent,
            snapshot_version=self._snapshot.snapshot_version + 1,
        )
        return self._snapshot

    def invalidate(self, *, generation: int, reason: str) -> SessionSnapshot:
        """Interrupt: atomically bump generation and force INTERRUPTED state."""
        self._snapshot = replace(
            self._snapshot,
            generation=generation,
            snapshot_version=self._snapshot.snapshot_version + 1,
            floor_state=FloorState.INTERRUPTED,
            pending_call_ids=(),
        )
        return self._snapshot

    def update_floor(
        self,
        state: FloorState,
        *,
        force: bool = False,
    ) -> SessionSnapshot:
        """
        Transition the floor to *state*, enforcing the legal transition table.

        If *force* is True, skip validation (used internally by invalidate).
        Raises ValueError for illegal transitions.
        """
        current = self._snapshot.floor_state
        if not force and state not in _LEGAL_TRANSITIONS.get(current, frozenset()):
            raise ValueError(
                f"Illegal floor transition {current.value} -> {state.value}"
            )
        self._snapshot = replace(
            self._snapshot,
            floor_state=state,
            snapshot_version=self._snapshot.snapshot_version + 1,
        )
        return self._snapshot

    def can_emit_filler(self, now_ms: float) -> bool:
        """
        True if enough time has elapsed since the last filler to allow a new one.
        Filler spam is throttled to at most once per filler_window_ms.
        """
        if self._last_filler_ms is None:
            return True
        return (now_ms - self._last_filler_ms) >= self._filler_window_ms

    def record_filler_emitted(self, now_ms: float) -> None:
        """Mark that a filler was emitted at *now_ms*."""
        self._last_filler_ms = now_ms

    def add_pending_call(self, call_id: str) -> SessionSnapshot:
        self._snapshot = replace(
            self._snapshot,
            pending_call_ids=(*self._snapshot.pending_call_ids, call_id),
            snapshot_version=self._snapshot.snapshot_version + 1,
        )
        return self._snapshot

    def finish_call(self, call_id: str, action: str | None = None) -> SessionSnapshot:
        pending = tuple(x for x in self._snapshot.pending_call_ids if x != call_id)
        completed = (
            self._snapshot.completed_actions
            if action is None
            else (*self._snapshot.completed_actions, action)
        )
        self._snapshot = replace(
            self._snapshot,
            pending_call_ids=pending,
            completed_actions=completed,
            snapshot_version=self._snapshot.snapshot_version + 1,
        )
        return self._snapshot
