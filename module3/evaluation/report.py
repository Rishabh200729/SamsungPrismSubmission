"""
module3/evaluation/report.py

Evaluation report generator.

Produces both:
- Human-readable text report (for console output and demos)
- Machine-readable JSON report (for automated scoring pipelines)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assertions import AssertionResult
from .metrics import Metrics
from .runner import ScenarioResult


@dataclass
class EvaluationReport:
    """Complete evaluation report for a scenario run."""
    scenario_name: str
    passed: bool
    metrics: Metrics
    assertion_results: list[AssertionResult]
    trace_entry_count: int
    output_count: int
    session_ids: dict[str, str]
    final_time_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "passed": self.passed,
            "error": self.error,
            "summary": {
                "trace_entries": self.trace_entry_count,
                "outputs": self.output_count,
                "final_time_ms": self.final_time_ms,
                "assertions_passed": sum(1 for a in self.assertion_results if a.passed),
                "assertions_total": len(self.assertion_results),
            },
            "metrics": self.metrics.to_dict(),
            "assertions": [
                {
                    "description": a.description,
                    "passed": a.passed,
                    "actual": a.actual,
                    "expected": a.expected,
                    "detail": a.detail,
                }
                for a in self.assertion_results
            ],
            "sessions": self.session_ids,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    def print_human_readable(self) -> None:
        """Print a formatted evaluation report to stdout."""
        width = 70
        status = "[PASSED]" if self.passed else "[FAILED]"
        print(f"\n{'='*width}")
        print(f"  PRISM Evaluation Report -- {self.scenario_name}")
        print(f"  Status: {status}")
        if self.error:
            print(f"  Error: {self.error}")
        print(f"{'='*width}")

        # Metrics summary
        m = self.metrics
        print("\n[METRICS]")
        print(f"  Response Latency:")
        print(f"    First output:         {_fmt_ms(m.first_output_latency_ms)}")
        print(f"    First filler:         {_fmt_ms(m.first_filler_latency_ms)}")
        print(f"    First final response: {_fmt_ms(m.first_final_response_latency_ms)}")
        print(f"  Interruption:")
        print(f"    Cancellation delay:   {_fmt_ms(m.interruption_to_cancel_delay_ms)}")
        print(f"  Stale results:          {m.stale_result_count}")
        print(f"  Duplicate call IDs:     {len(m.duplicate_call_ids)}")
        print(f"  Tasks: created={m.tasks_created} completed={m.tasks_completed} "
              f"cancelled={m.tasks_cancelled} failed={m.tasks_failed} stale={m.tasks_stale}")
        print(f"  Protocol violations:    {len(m.protocol_violations)}")
        print(f"  Cross-session events:   {m.cross_session_events}")
        print(f"  Complete lifecycle:     {'Yes' if m.has_complete_lifecycle else 'No'}")
        print(f"  Trace entries:          {m.trace_entry_count}")
        print(f"  Output events:          {self.output_count}")

        # Assertions
        print(f"\n[ASSERTIONS] ({len(self.assertion_results)} total)")
        for ar in self.assertion_results:
            icon = "  [OK]" if ar.passed else "  [FAIL]"
            print(f"{icon} {ar.description}")
            if not ar.passed:
                print(f"       expected={ar.expected} actual={ar.actual}")
                if ar.detail:
                    print(f"       {ar.detail}")

        passed_count = sum(1 for a in self.assertion_results if a.passed)
        print(f"\n  {passed_count}/{len(self.assertion_results)} assertions passed")
        print(f"{'='*width}\n")


def build_report(
    scenario_name: str,
    result: ScenarioResult,
    metrics: Metrics,
    assertion_results: list[AssertionResult],
) -> EvaluationReport:
    """Assemble a complete EvaluationReport."""
    passed = all(a.passed for a in assertion_results) and result.error is None
    return EvaluationReport(
        scenario_name=scenario_name,
        passed=passed,
        metrics=metrics,
        assertion_results=assertion_results,
        trace_entry_count=len(result.trace),
        output_count=len(result.outputs),
        session_ids=result.session_ids,
        final_time_ms=result.final_virtual_time_ms,
        error=result.error,
    )


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f}ms"
