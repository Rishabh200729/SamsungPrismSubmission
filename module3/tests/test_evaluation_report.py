"""Evaluation report and assertion-engine regression tests."""

import json

from module3.evaluation.assertions import all_passed, run_assertions
from module3.evaluation.metrics import Metrics
from module3.evaluation.report import _fmt_ms, build_report
from module3.evaluation.runner import ScenarioResult
from module3.evaluation.scenario import OutputAssertion, Scenario, TraceAssertion
from module3.runtime.events import make_filler, make_final_response
from module3.runtime.tracing.recorder import TraceEntry


def _result(trace=None, outputs=None):
    return ScenarioResult(
        scenario_name="report-case",
        session_ids={"default": "s1"},
        outputs=outputs or [],
        trace=trace or [],
        final_virtual_time_ms=99.0,
    )


def test_assertions_cover_filters_failures_ordering_and_session_scope():
    trace = [
        TraceEntry(1, 5, "X", "s1", metadata={"tag": "yes"}),
        TraceEntry(2, 15, "X", "s2", metadata={"tag": "yes"}),
        TraceEntry(3, 25, "X", "s1", metadata={"tag": "no"}),
    ]
    outputs = [make_final_response("s1", 20, "done"), make_filler("s1", 10, "wait")]
    scenario = Scenario(
        name="assertions",
        trace_assertions=[
            TraceAssertion("MISSING", min_count=1),
            TraceAssertion("X", min_count=2, max_count=2, session_key="default"),
            TraceAssertion("X", min_count=1, after_ms=0, before_ms=10, metadata_contains={"tag": "yes"}),
        ],
        output_assertions=[
            OutputAssertion("FILLER", min_count=2),
            OutputAssertion("FINAL_RESPONSE", max_count=0),
            OutputAssertion("FILLER", must_precede="FINAL_RESPONSE"),
        ],
    )
    results = run_assertions(scenario, _result(trace, outputs))
    assert [result.passed for result in results] == [False, True, True, False, False, False]
    assert "appeared after" in results[-1].detail
    assert not all_passed(results)


def test_report_serializes_saves_and_prints_passing_and_failing_cases(tmp_path, capsys):
    passing_result = _result([TraceEntry(1, 0, "TEXT_CHUNK", "s1")], [make_filler("s1", 5, "wait")])
    passing_assertions = run_assertions(Scenario(name="pass", trace_assertions=[TraceAssertion("TEXT_CHUNK")]), passing_result)
    report = build_report("pass", passing_result, Metrics(trace_entry_count=1), passing_assertions)
    data = report.to_dict()
    assert data["scenario"] == "pass" and data["summary"]["assertions_passed"] == 1
    assert json.loads(report.to_json())["sessions"] == {"default": "s1"}

    path = tmp_path / "reports" / "result.json"
    report.save_json(path)
    assert json.loads(path.read_text())["passed"] is True
    report.print_human_readable()
    assert "[PASSED]" in capsys.readouterr().out

    failed_assertions = run_assertions(Scenario(name="fail", trace_assertions=[TraceAssertion("MISSING")]), _result())
    failed_assertions[0].detail = "missing trace"
    failed = build_report("fail", _result(), Metrics(), failed_assertions)
    failed.print_human_readable()
    text = capsys.readouterr().out
    assert "[FAILED]" in text and "missing trace" in text
    assert _fmt_ms(None) == "N/A" and _fmt_ms(1.25) == "1.2ms"
