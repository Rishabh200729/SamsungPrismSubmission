"""
module3/evaluation/metrics.py

Scoring metrics for PRISM evaluation.

Computes all scoring categories from a ScenarioResult trace:
A) Response Latency
B) Interruption Cancellation Delay
C) Stale Result Count
D) Duplicate / Invalid Actions
E) Task Completion
F) Protocol Validity
G) Session Isolation
H) Trace Completeness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runner import ScenarioResult


@dataclass
class Metrics:
    """All computed metrics for one scenario run."""

    # A) Response Latency
    first_output_latency_ms: float | None = None   # First output - first input
    first_filler_latency_ms: float | None = None   # First FILLER - first input
    first_final_response_latency_ms: float | None = None

    # B) Interruption Cancellation Delay
    interruption_to_cancel_delay_ms: float | None = None  # CANCEL_REQUESTED - INTERRUPTION

    # C) Stale Results
    stale_result_count: int = 0

    # D) Duplicate / Invalid Actions
    duplicate_call_ids: list[str] = field(default_factory=list)
    invalid_actions: list[str] = field(default_factory=list)

    # E) Task Completion
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_cancelled: int = 0
    tasks_failed: int = 0
    tasks_stale: int = 0

    # F) Protocol Validity
    events_missing_session_id: int = 0
    events_missing_event_id: int = 0
    protocol_violations: list[str] = field(default_factory=list)

    # G) Session Isolation
    cross_session_events: int = 0

    # H) Trace Completeness
    has_complete_lifecycle: bool = False  # At least one: CREATED->STARTED->COMPLETED chain
    trace_entry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_latency": {
                "first_output_ms": self.first_output_latency_ms,
                "first_filler_ms": self.first_filler_latency_ms,
                "first_final_response_ms": self.first_final_response_latency_ms,
            },
            "interruption": {
                "cancellation_delay_ms": self.interruption_to_cancel_delay_ms,
            },
            "stale_results": {
                "count": self.stale_result_count,
            },
            "task_completion": {
                "created": self.tasks_created,
                "completed": self.tasks_completed,
                "cancelled": self.tasks_cancelled,
                "failed": self.tasks_failed,
                "stale": self.tasks_stale,
            },
            "protocol_validity": {
                "missing_session_id": self.events_missing_session_id,
                "missing_event_id": self.events_missing_event_id,
                "violations": self.protocol_violations,
                "duplicate_call_ids": self.duplicate_call_ids,
            },
            "session_isolation": {
                "cross_session_events": self.cross_session_events,
            },
            "trace_completeness": {
                "has_complete_lifecycle": self.has_complete_lifecycle,
                "total_entries": self.trace_entry_count,
            },
        }


def compute_metrics(result: ScenarioResult) -> Metrics:
    """Compute all metrics from a ScenarioResult."""
    m = Metrics()
    trace = result.trace
    outputs = result.outputs
    m.trace_entry_count = len(trace)

    # ------------------------------------------------------------------
    # A) Response Latency
    # ------------------------------------------------------------------
    first_input_ts = None
    for e in trace:
        if e.entry_type in ("TEXT_CHUNK", "END_OF_TURN", "INTERRUPTION"):
            first_input_ts = e.timestamp_ms
            break

    if first_input_ts is not None and outputs:
        m.first_output_latency_ms = outputs[0].timestamp_ms - first_input_ts

    # FILLER latency
    for o in outputs:
        if o.event_type.value == "FILLER":
            m.first_filler_latency_ms = o.timestamp_ms - (first_input_ts or 0)
            break

    # FINAL_RESPONSE latency
    for o in outputs:
        if o.event_type.value == "FINAL_RESPONSE":
            m.first_final_response_latency_ms = o.timestamp_ms - (first_input_ts or 0)
            break

    # ------------------------------------------------------------------
    # B) Interruption Cancellation Delay
    # ------------------------------------------------------------------
    interruption_ts = None
    cancel_ts = None
    for e in trace:
        if e.entry_type == "INTERRUPTION" and interruption_ts is None:
            interruption_ts = e.timestamp_ms
        if e.entry_type == "TASK_CANCEL_REQUESTED" and cancel_ts is None:
            cancel_ts = e.timestamp_ms

    if interruption_ts is not None and cancel_ts is not None:
        m.interruption_to_cancel_delay_ms = cancel_ts - interruption_ts

    # ------------------------------------------------------------------
    # C) Stale Results
    # ------------------------------------------------------------------
    m.stale_result_count = sum(
        1 for e in trace if e.entry_type == "STALE_RESULT_DISCARDED"
    )

    # ------------------------------------------------------------------
    # D) Duplicate Call IDs
    # ------------------------------------------------------------------
    seen_call_ids: set[str] = set()
    for o in outputs:
        cid = o.call_id
        if cid:
            if cid in seen_call_ids:
                m.duplicate_call_ids.append(cid)
            else:
                seen_call_ids.add(cid)
    # Also capture duplicates that were soft-rejected by the output gate
    # (they never reach the output queue but ARE recorded in the trace).
    for e in trace:
        if e.entry_type == "DUPLICATE_CALL_ID" and e.call_id:
            if e.call_id not in m.duplicate_call_ids:
                m.duplicate_call_ids.append(e.call_id)

    # ------------------------------------------------------------------
    # E) Task Completion
    # ------------------------------------------------------------------
    m.tasks_created = sum(1 for e in trace if e.entry_type == "TASK_CREATED")
    m.tasks_completed = sum(1 for e in trace if e.entry_type == "TASK_COMPLETED")
    m.tasks_cancelled = sum(1 for e in trace if e.entry_type == "TASK_CANCELLED")
    m.tasks_failed = sum(1 for e in trace if e.entry_type == "TASK_FAILED")
    m.tasks_stale = sum(1 for e in trace if e.entry_type == "STALE_RESULT_DISCARDED")

    # ------------------------------------------------------------------
    # F) Protocol Validity
    # ------------------------------------------------------------------
    for e in trace:
        if not e.session_id:
            m.events_missing_session_id += 1
        if not e.event_id:
            m.events_missing_event_id += 1

    # ------------------------------------------------------------------
    # G) Session Isolation
    # ------------------------------------------------------------------
    # Each trace entry's session_id must be one of the known sessions
    known_sessions = set(result.session_ids.values())
    for e in trace:
        if e.session_id and e.session_id not in known_sessions and e.session_id != "unknown":
            m.cross_session_events += 1

    # ------------------------------------------------------------------
    # H) Trace Completeness
    # ------------------------------------------------------------------
    # Check for at least one full CREATED -> STARTED -> COMPLETED chain
    created_task_ids = {e.task_id for e in trace if e.entry_type == "TASK_CREATED" and e.task_id}
    started_task_ids = {e.task_id for e in trace if e.entry_type == "TASK_STARTED" and e.task_id}
    completed_task_ids = {e.task_id for e in trace if e.entry_type == "TASK_COMPLETED" and e.task_id}
    m.has_complete_lifecycle = bool(
        created_task_ids & started_task_ids & completed_task_ids
    )

    return m
