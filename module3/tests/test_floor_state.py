"""
module3/tests/test_floor_state.py

Unit tests for SessionStateStore floor-state machine (§6).
Validates transition enforcement, interruption shortcut, and filler throttling.
"""

from __future__ import annotations

import pytest
from module3.runtime.sessions.state import (
    FloorState, SessionStateStore, SessionSnapshot, SlotValue,
)


SID = "floor-test-session"


# ---------------------------------------------------------------------------
# FloorState transition validation
# ---------------------------------------------------------------------------

def test_legal_transition_listening_to_thinking():
    store = SessionStateStore(SID)
    assert store.snapshot.floor_state == FloorState.LISTENING
    snap = store.update_floor(FloorState.THINKING)
    assert snap.floor_state == FloorState.THINKING


def test_legal_transition_thinking_to_speaking():
    store = SessionStateStore(SID)
    store.update_floor(FloorState.THINKING)
    snap = store.update_floor(FloorState.SPEAKING)
    assert snap.floor_state == FloorState.SPEAKING


def test_legal_transition_speaking_to_interrupted():
    store = SessionStateStore(SID)
    store.update_floor(FloorState.THINKING)
    store.update_floor(FloorState.SPEAKING)
    snap = store.update_floor(FloorState.INTERRUPTED)
    assert snap.floor_state == FloorState.INTERRUPTED


def test_illegal_transition_listening_to_speaking_raises():
    store = SessionStateStore(SID)
    with pytest.raises(ValueError, match="Illegal floor transition"):
        store.update_floor(FloorState.SPEAKING)


def test_illegal_transition_speaking_to_thinking_raises():
    store = SessionStateStore(SID)
    store.update_floor(FloorState.THINKING)
    store.update_floor(FloorState.SPEAKING)
    with pytest.raises(ValueError, match="Illegal floor transition"):
        store.update_floor(FloorState.THINKING)


def test_illegal_transition_waiting_for_tool_to_speaking_then_tool():
    store = SessionStateStore(SID)
    store.update_floor(FloorState.THINKING)
    store.update_floor(FloorState.WAITING_FOR_TOOL)
    snap = store.update_floor(FloorState.SPEAKING)
    assert snap.floor_state == FloorState.SPEAKING


def test_force_bypass_validation():
    """force=True allows any transition including normally illegal ones."""
    store = SessionStateStore(SID)
    # LISTENING -> SPEAKING is illegal normally
    snap = store.update_floor(FloorState.SPEAKING, force=True)
    assert snap.floor_state == FloorState.SPEAKING


def test_invalidate_always_forces_interrupted():
    """invalidate() must set INTERRUPTED regardless of current state."""
    store = SessionStateStore(SID)
    store.update_floor(FloorState.THINKING)
    store.update_floor(FloorState.SPEAKING)
    snap = store.invalidate(generation=2, reason="barge-in")
    assert snap.floor_state == FloorState.INTERRUPTED
    assert snap.generation == 2


def test_snapshot_version_increments_on_floor_change():
    store = SessionStateStore(SID)
    v0 = store.snapshot.snapshot_version
    store.update_floor(FloorState.THINKING)
    assert store.snapshot.snapshot_version == v0 + 1


# ---------------------------------------------------------------------------
# Filler throttling
# ---------------------------------------------------------------------------

def test_filler_allowed_initially():
    store = SessionStateStore(SID, filler_window_ms=2000.0)
    assert store.can_emit_filler(now_ms=0.0) is True


def test_filler_blocked_within_window():
    store = SessionStateStore(SID, filler_window_ms=2000.0)
    store.record_filler_emitted(now_ms=1000.0)
    assert store.can_emit_filler(now_ms=2999.0) is False  # only 1999 ms elapsed


def test_filler_allowed_after_window():
    store = SessionStateStore(SID, filler_window_ms=2000.0)
    store.record_filler_emitted(now_ms=1000.0)
    assert store.can_emit_filler(now_ms=3000.0) is True  # exactly 2000 ms elapsed


def test_filler_window_resets_on_each_emit():
    store = SessionStateStore(SID, filler_window_ms=500.0)
    store.record_filler_emitted(now_ms=100.0)
    store.record_filler_emitted(now_ms=700.0)  # new baseline
    assert store.can_emit_filler(now_ms=1100.0) is False  # only 400 ms since last
    assert store.can_emit_filler(now_ms=1200.0) is True   # 500 ms since last


# ---------------------------------------------------------------------------
# Slot patches preserve unrelated slots
# ---------------------------------------------------------------------------

def test_slot_patch_localized():
    store = SessionStateStore(SID)
    v0 = store.snapshot.snapshot_version
    snap1 = store.patch_slots(
        v0,
        {"destination": "Paris", "date": "2026-10-01", "passengers": 2},
        source_event_id="e1",
        timestamp_ms=10.0,
    )
    snap2 = store.patch_slots(
        snap1.snapshot_version,
        {"destination": "Tokyo"},  # only destination changes
        source_event_id="e2",
        timestamp_ms=20.0,
    )
    assert snap2.slots["destination"].value == "Tokyo"
    assert snap2.slots["date"].value == "2026-10-01"
    assert snap2.slots["passengers"].value == 2


def test_stale_version_patch_raises():
    store = SessionStateStore(SID)
    v0 = store.snapshot.snapshot_version
    store.patch_slots(v0, {"x": 1}, source_event_id=None, timestamp_ms=0.0)
    with pytest.raises(ValueError, match="stale"):
        store.patch_slots(v0, {"x": 2}, source_event_id=None, timestamp_ms=1.0)
