"""
module3/runtime/tracing/recorder.py

First-class TraceRecorder.

Records ALL important runtime transitions in an append-only log.
Each entry is a TraceEntry with enough information to reconstruct
the causal chain of events without replaying the actual logic.

The trace is the primary artifact for evaluation. The evaluation
harness reads it to compute all scoring metrics.

Design: append-only list in memory (JSONL export available).
Inspired by Gaia2: reproducible traces for deterministic evaluation.

Trace entry example:

    {
        "seq": 7,
        "timestamp_ms": 451.0,
        "event_id": "...",
        "session_id": "s1",
        "task_id": "t2",
        "call_id": "c1",
        "entry_type": "STALE_RESULT_DISCARDED",
        "status_before": "CANCELLATION_REQUESTED",
        "status_after": "STALE",
        "phase": "WAITING_FOR_RESULT",
        "generation": 2,
        "source": "cancellation_coordinator",
        "metadata": {"task_generation": 1, "current_generation": 2}
    }
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..events.base import BaseEvent

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """One trace record."""
    seq: int                           # Monotonic sequence number
    timestamp_ms: float                # Virtual or wall-clock time
    entry_type: str                    # Event type or internal lifecycle label
    session_id: str

    # Optional correlation fields
    event_id: str | None = None
    task_id: str | None = None
    call_id: str | None = None
    generation: int | None = None
    correlation_id: str | None = None

    # State transition context
    status_before: str | None = None
    status_after: str | None = None
    phase: str | None = None

    # Attribution
    source: str | None = None

    # Wall-clock time (for latency audit in real-time mode)
    wall_clock_ms: float | None = None

    # Arbitrary extra data
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Remove None values for compact output
        return {k: v for k, v in d.items() if v is not None}


class TraceRecorder:
    """
    Append-only trace recorder for a single runtime instance.

    Multiple sessions share one recorder (entries are tagged with session_id).
    The evaluator reads from one recorder to assess the full scenario.

    Usage::

        recorder = TraceRecorder()
        recorder.record_event(event)
        recorder.record_transition(
            session_id="s1",
            timestamp_ms=452.0,
            entry_type="TASK_CANCELLED",
            task_id="t1",
            status_before="CANCELLATION_REQUESTED",
            status_after="CANCELLED",
        )
        recorder.export_jsonl("trace.jsonl")
    """

    def __init__(self, record_wall_clock: bool = False) -> None:
        self._entries: list[TraceEntry] = []
        self._seq = 0
        self._record_wall_clock = record_wall_clock

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_event(
        self,
        event: BaseEvent,
        source: str | None = None,
        status_before: str | None = None,
        status_after: str | None = None,
        phase: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """
        Record a BaseEvent as a trace entry.
        Used for both input and output events flowing through the runtime.
        """
        metadata = dict(event.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        if event.payload:
            # Include payload summary (not always the full payload for brevity)
            metadata["payload_keys"] = list(event.payload.keys())

        entry = TraceEntry(
            seq=self._next_seq(),
            timestamp_ms=event.timestamp_ms,
            entry_type=event.event_type.value,
            session_id=event.session_id,
            event_id=event.event_id,
            task_id=event.task_id,
            call_id=event.call_id,
            generation=event.generation,
            correlation_id=event.correlation_id,
            status_before=status_before,
            status_after=status_after,
            phase=phase,
            source=source or event.source,
            wall_clock_ms=self._wall_now() if self._record_wall_clock else None,
            metadata=metadata,
        )
        self._append(entry)
        return entry

    def record_transition(
        self,
        session_id: str,
        timestamp_ms: float,
        entry_type: str,
        task_id: str | None = None,
        call_id: str | None = None,
        generation: int | None = None,
        status_before: str | None = None,
        status_after: str | None = None,
        phase: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """
        Record an internal state transition (not tied to an external event).
        Used for task lifecycle changes, generation invalidation, etc.
        """
        entry = TraceEntry(
            seq=self._next_seq(),
            timestamp_ms=timestamp_ms,
            entry_type=entry_type,
            session_id=session_id,
            task_id=task_id,
            call_id=call_id,
            generation=generation,
            status_before=status_before,
            status_after=status_after,
            phase=phase,
            source=source or "runtime",
            wall_clock_ms=self._wall_now() if self._record_wall_clock else None,
            metadata=metadata or {},
        )
        self._append(entry)
        return entry

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def entries(self) -> list[TraceEntry]:
        """All trace entries in sequence order."""
        return list(self._entries)

    def entries_for_session(self, session_id: str) -> list[TraceEntry]:
        return [e for e in self._entries if e.session_id == session_id]

    def entries_for_task(self, task_id: str) -> list[TraceEntry]:
        return [e for e in self._entries if e.task_id == task_id]

    def entries_of_type(self, entry_type: str) -> list[TraceEntry]:
        return [e for e in self._entries if e.entry_type == entry_type]

    def first_of_type(self, entry_type: str, session_id: str | None = None) -> TraceEntry | None:
        for e in self._entries:
            if e.entry_type == entry_type:
                if session_id is None or e.session_id == session_id:
                    return e
        return None

    def count_of_type(self, entry_type: str) -> int:
        return sum(1 for e in self._entries if e.entry_type == entry_type)

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        """Export all entries as a list of dicts."""
        return [e.to_dict() for e in self._entries]

    def export_jsonl(self, path: str | Path) -> None:
        """Write all entries to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info("TraceRecorder: exported %d entries to %s", len(self._entries), p)

    def export_json(self, path: str | Path) -> None:
        """Write all entries as a JSON array."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2)
        logger.info("TraceRecorder: exported %d entries to %s", len(self._entries), p)

    def print_summary(self) -> None:
        """Print a human-readable trace to stdout (for demos and debugging)."""
        print(f"\n{'='*70}")
        print(f"TRACE SUMMARY ({len(self._entries)} entries)")
        print(f"{'='*70}")
        for e in self._entries:
            parts = [f"T={e.timestamp_ms:>8.1f}ms", f"[{e.entry_type:<30}]"]
            if e.task_id:
                parts.append(f"task={e.task_id[:8]}")
            if e.call_id:
                parts.append(f"call={e.call_id[:8]}")
            if e.generation is not None:
                parts.append(f"gen={e.generation}")
            if e.status_before and e.status_after:
                parts.append(f"{e.status_before}->{e.status_after}")
            if e.metadata.get("task_generation") is not None:
                parts.append(f"(task_gen={e.metadata['task_generation']})")
            print("  " + "  ".join(parts))
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _append(self, entry: TraceEntry) -> None:
        self._entries.append(entry)
        logger.debug(
            "Trace[%d] %s session=%s task=%s gen=%s ts=%.1f",
            entry.seq,
            entry.entry_type,
            entry.session_id,
            entry.task_id or "-",
            entry.generation or "-",
            entry.timestamp_ms,
        )

    @staticmethod
    def _wall_now() -> float:
        return time.monotonic() * 1000.0

    def clear(self) -> None:
        """Reset the recorder (useful between test scenarios)."""
        self._entries.clear()
        self._seq = 0
