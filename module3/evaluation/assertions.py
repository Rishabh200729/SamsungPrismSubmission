"""
module3/evaluation/assertions.py

Trace-based assertion engine.

Reads a ScenarioResult and verifies trace + output assertions.
Returns a list of AssertionResult objects with pass/fail status and details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runner import ScenarioResult
from .scenario import OutputAssertion, Scenario, TraceAssertion


@dataclass
class AssertionResult:
    """Result of a single assertion check."""
    description: str
    passed: bool
    actual: Any = None
    expected: Any = None
    detail: str = ""


def run_assertions(scenario: Scenario, result: ScenarioResult) -> list[AssertionResult]:
    """
    Run all assertions defined in the scenario against the ScenarioResult.
    Returns list of AssertionResult (one per assertion).
    """
    assertion_results: list[AssertionResult] = []

    # Trace assertions
    for assertion in scenario.trace_assertions:
        ar = _check_trace_assertion(assertion, result)
        assertion_results.append(ar)

    # Output assertions
    for assertion in scenario.output_assertions:
        ar = _check_output_assertion(assertion, result)
        assertion_results.append(ar)

    return assertion_results


def _check_trace_assertion(
    assertion: TraceAssertion, result: ScenarioResult
) -> AssertionResult:
    desc = assertion.description or f"trace has {assertion.min_count}+ {assertion.entry_type}"

    # Get session_id for this assertion
    session_id = result.session_ids.get(assertion.session_key)

    # Filter trace entries
    matching = [
        e for e in result.trace
        if e.entry_type == assertion.entry_type
        and (session_id is None or e.session_id == session_id)
    ]

    # Time filters
    if assertion.after_ms is not None:
        matching = [e for e in matching if e.timestamp_ms > assertion.after_ms]
    if assertion.before_ms is not None:
        matching = [e for e in matching if e.timestamp_ms < assertion.before_ms]

    # Metadata filters
    if assertion.metadata_contains:
        def _meta_match(entry: Any) -> bool:
            for k, v in assertion.metadata_contains.items():
                if entry.metadata.get(k) != v:
                    return False
            return True
        matching = [e for e in matching if _meta_match(e)]

    count = len(matching)

    if count < assertion.min_count:
        return AssertionResult(
            description=desc,
            passed=False,
            actual=count,
            expected=f">= {assertion.min_count}",
            detail=f"Only {count} matching trace entries for {assertion.entry_type}",
        )

    if assertion.max_count is not None and count > assertion.max_count:
        return AssertionResult(
            description=desc,
            passed=False,
            actual=count,
            expected=f"<= {assertion.max_count}",
            detail=f"Too many ({count}) trace entries for {assertion.entry_type}",
        )

    return AssertionResult(
        description=desc,
        passed=True,
        actual=count,
        expected=f">= {assertion.min_count}",
    )


def _check_output_assertion(
    assertion: OutputAssertion, result: ScenarioResult
) -> AssertionResult:
    desc = assertion.description or f"output has {assertion.min_count}+ {assertion.event_type}"

    matching = [
        e for e in result.outputs
        if e.event_type.value == assertion.event_type
    ]
    count = len(matching)

    if count < assertion.min_count:
        return AssertionResult(
            description=desc,
            passed=False,
            actual=count,
            expected=f">= {assertion.min_count}",
            detail=f"Only {count} output events of type {assertion.event_type}",
        )

    if assertion.max_count is not None and count > assertion.max_count:
        return AssertionResult(
            description=desc,
            passed=False,
            actual=count,
            expected=f"<= {assertion.max_count}",
        )

    # Order assertion: this event type must precede another
    if assertion.must_precede:
        output_types = [e.event_type.value for e in result.outputs]
        try:
            first_idx = output_types.index(assertion.event_type)
            try:
                other_idx = output_types.index(assertion.must_precede)
                if first_idx >= other_idx:
                    return AssertionResult(
                        description=desc,
                        passed=False,
                        detail=f"{assertion.event_type} must precede {assertion.must_precede} but appeared after",
                    )
            except ValueError:
                # must_precede type not in outputs — that's a separate assertion
                pass
        except ValueError:
            pass  # Already caught by min_count check above

    return AssertionResult(description=desc, passed=True, actual=count)


def all_passed(assertion_results: list[AssertionResult]) -> bool:
    return all(a.passed for a in assertion_results)
