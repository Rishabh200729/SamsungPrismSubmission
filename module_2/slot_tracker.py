"""
slot_tracker.py
Per-session slot accumulation, readiness gating, and state-snapshot building.

This is the missing piece between NLU extraction (nlu.py) and tool dispatch
(orchestrator.py). Every turn, the orchestrator calls:

    update_kind = tracker.update(intent, extracted_slots, turn_index)

If `is_ready()` is True, all required slots for the current intent are filled
and the orchestrator fires a tool call. If not, it emits a CLARIFICATION
listing `missing_slots()`.

Design decisions:
- NEW vs PATCH vs REPLACE: switching intents wipes all slots (REPLACE) because
  a book_hotel's "city" slot has nothing to do with book_flight's "destination".
  Same intent + new slots = PATCH (additive merge). First update = NEW.
- `snapshot()` returns the dict shape that `make_final_response(state_snapshot=...)`
  expects — see module3's `StateSnapshot` model for the canonical schema.
- `retry_count` feeds into `make_idempotency_key(attempt=...)` so a deliberate
  user-requested retry gets a fresh key distinct from the original failed attempt.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tool_manifest import ToolManifest


class UpdateKind(Enum):
    """Returned by SlotTracker.update() so the caller knows what happened."""
    NEW = "NEW"            # First update for this tracker (no prior intent)
    PATCH = "PATCH"        # Same intent, slots merged additively
    REPLACE = "REPLACE"    # Intent changed — old slots wiped, new ones installed


@dataclass
class Slot:
    """One filled slot value with provenance metadata."""
    value: Any
    source_turn: int = 0
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"Slot({self.value!r}, turn={self.source_turn})"


class SlotTracker:
    """
    Per-session, per-intent slot accumulator.

    Parameters
    ----------
    manifest : ToolManifest
        Used to look up which slots are required for the current intent,
        so `is_ready()` and `missing_slots()` work without hardcoding
        tool schemas here.
    """

    def __init__(self, manifest: ToolManifest) -> None:
        self._manifest = manifest
        self._active_intent: Optional[str] = None
        self._slots: Dict[str, Slot] = {}
        self._completed_actions: List[str] = []
        self._result_data: Dict[str, Any] = {}
        self._call_ids: List[str] = []
        self._retry_count: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        intent: str,
        extracted_slots: Dict[str, Any],
        turn_index: int = 0,
    ) -> UpdateKind:
        """
        Merge newly extracted intent + slots into the tracked state.

        Returns UpdateKind so callers (orchestrator, telemetry) can
        distinguish first-fill from correction from intent-switch without
        inspecting internal state.
        """
        if self._active_intent is None:
            # Very first update — no prior state to merge or wipe.
            kind = UpdateKind.NEW
            self._active_intent = intent
        elif intent != self._active_intent:
            # Intent changed — wipe everything. A book_hotel's "city" slot
            # is not transferable to book_flight's "destination".
            kind = UpdateKind.REPLACE
            self._slots.clear()
            self._retry_count = 0
            self._active_intent = intent
        else:
            kind = UpdateKind.PATCH

        # Merge extracted slots (overwrite per-key, preserving untouched ones).
        for name, value in extracted_slots.items():
            if value is not None:
                self._slots[name] = Slot(
                    value=value,
                    source_turn=turn_index,
                )

        return kind

    # ------------------------------------------------------------------
    # Readiness gating
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True when every required slot for the active intent is filled."""
        if self._active_intent is None:
            return False
        for slot_name in self._required_slot_names():
            if slot_name not in self._slots:
                return False
        return True

    def missing_slots(self) -> List[str]:
        """Return names of required slots that haven't been filled yet."""
        if self._active_intent is None:
            return []
        required = self._required_slot_names()
        return [name for name in required if name not in self._slots]

    def _required_slot_names(self) -> List[str]:
        """Look up required slots from the manifest for the active intent."""
        if self._active_intent is None:
            return []
        try:
            return self._manifest.required_slots_for(self._active_intent)
        except KeyError:
            # Tool not in manifest (e.g. unseen tool scenario) — no required
            # slots known, so treat as "ready" (let the tool call proceed;
            # Module 3's output gate will validate further).
            return []

    # ------------------------------------------------------------------
    # Post-call lifecycle
    # ------------------------------------------------------------------

    def record_tool_call(self, call_id: str) -> None:
        """Record that a tool call was emitted for audit/tracing."""
        self._call_ids.append(call_id)

    def mark_completed(self, tool_name: str) -> None:
        """Record a successfully completed tool action."""
        if tool_name not in self._completed_actions:
            self._completed_actions.append(tool_name)

    def record_result_data(self, result_data: Dict[str, Any]) -> None:
        """Store the tool's result payload for inclusion in the state snapshot."""
        self._result_data.update(result_data)

    def increment_retry(self) -> None:
        """Bump retry counter so the next idempotency key is distinct."""
        self._retry_count += 1

    def reset_after_completion(self) -> None:
        """
        Clear transient state after a successful tool completion cycle.

        Keeps `_completed_actions` and `_result_data` intact (they're part
        of the session history) but resets the slot tracker to accept a
        fresh intent on the next turn.
        """
        self._active_intent = None
        self._slots.clear()
        self._retry_count = 0
        self._call_ids.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_intent(self) -> Optional[str]:
        return self._active_intent

    @property
    def slots(self) -> Dict[str, Slot]:
        """Mutable view of filled slots. The orchestrator reads .value off each."""
        return self._slots

    @property
    def retry_count(self) -> int:
        return self._retry_count

    # ------------------------------------------------------------------
    # Snapshot for FINAL_RESPONSE
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Build the state_snapshot dict that `make_final_response()` expects.

        Shape matches Module 3's `StateSnapshot` model:
            {
                "intent": str,
                "slots": {name: value, ...},
                "completed_actions": [str, ...],
                "result_data": {...},
            }
        """
        return {
            "intent": self._active_intent or "",
            "slots": {name: slot.value for name, slot in self._slots.items()},
            "completed_actions": list(self._completed_actions),
            "result_data": dict(self._result_data),
        }
