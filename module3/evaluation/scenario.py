"""
module3/evaluation/scenario.py

Scenario model for the PRISM evaluation harness.

A scenario defines:
- A sequence of input events with virtual timestamps
- Expected output event types and ordering
- Assertions on the trace
- Pass/fail criteria

Scenarios can be loaded from YAML or constructed programmatically.

Inspired by Gaia2: scenarios are environment specifications that drive
deterministic evaluation without real network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScenarioEvent:
    """One event to inject into the runtime at a specific virtual time."""
    at_ms: float
    event_type: str              # InputEventType string value
    payload: dict[str, Any] = field(default_factory=dict)
    session_key: str = "default" # Maps to a session_id assigned at runtime
    call_id: str | None = None
    task_id: str | None = None
    generation: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceAssertion:
    """
    An assertion on the trace after scenario completion.

    entry_type: The trace entry type to look for (e.g. "STALE_RESULT_DISCARDED")
    session_key: Which session to check
    min_count:  Minimum occurrences required
    max_count:  Maximum occurrences allowed (None = no limit)
    before_ms:  Entry must appear before this virtual timestamp
    after_ms:   Entry must appear after this virtual timestamp
    metadata_contains: Key-value pairs that must be present in entry metadata
    """
    entry_type: str
    session_key: str = "default"
    min_count: int = 1
    max_count: int | None = None
    before_ms: float | None = None
    after_ms: float | None = None
    metadata_contains: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class OutputAssertion:
    """
    An assertion on the output event sequence.

    event_type: The OutputEventType string value to expect
    min_count:  Minimum required occurrences
    must_precede: Another event_type that must come after this one
    """
    event_type: str
    min_count: int = 1
    max_count: int | None = None
    must_precede: str | None = None
    description: str = ""


@dataclass
class Scenario:
    """
    Complete scenario specification.

    Attributes:
        name:           Human-readable scenario name
        description:    What this scenario tests
        events:         Ordered list of events to inject
        trace_assertions: Assertions on the internal trace
        output_assertions: Assertions on output events
        max_duration_ms: Scenario must complete within this virtual time
        sessions:        Named session configuration (key -> metadata)
        tags:            Categorization tags for test filtering
    """
    name: str
    description: str = ""
    events: list[ScenarioEvent] = field(default_factory=list)
    trace_assertions: list[TraceAssertion] = field(default_factory=list)
    output_assertions: list[OutputAssertion] = field(default_factory=list)
    max_duration_ms: float = 10_000.0
    sessions: dict[str, dict[str, Any]] = field(default_factory=lambda: {"default": {}})
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        """Load a scenario from a YAML file."""
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        """Construct a Scenario from a plain dictionary."""
        scenario_data = data.get("scenario", data)

        events = [
            ScenarioEvent(
                at_ms=float(e["at_ms"]),
                event_type=e["type"],
                payload=e.get("payload", {}),
                session_key=e.get("session_key", "default"),
                call_id=e.get("call_id"),
                task_id=e.get("task_id"),
                generation=e.get("generation"),
                metadata=e.get("metadata", {}),
            )
            for e in scenario_data.get("events", [])
        ]

        trace_assertions = [
            TraceAssertion(
                entry_type=a["entry_type"],
                session_key=a.get("session_key", "default"),
                min_count=a.get("min_count", 1),
                max_count=a.get("max_count"),
                before_ms=a.get("before_ms"),
                after_ms=a.get("after_ms"),
                metadata_contains=a.get("metadata_contains", {}),
                description=a.get("description", ""),
            )
            for a in scenario_data.get("trace_assertions", [])
        ]

        output_assertions = [
            OutputAssertion(
                event_type=a["event_type"],
                min_count=a.get("min_count", 1),
                max_count=a.get("max_count"),
                must_precede=a.get("must_precede"),
                description=a.get("description", ""),
            )
            for a in scenario_data.get("output_assertions", [])
        ]

        return cls(
            name=scenario_data.get("name", "unnamed"),
            description=scenario_data.get("description", ""),
            events=events,
            trace_assertions=trace_assertions,
            output_assertions=output_assertions,
            max_duration_ms=float(scenario_data.get("max_duration_ms", 10_000.0)),
            sessions=scenario_data.get("sessions", {"default": {}}),
            tags=scenario_data.get("tags", []),
        )
