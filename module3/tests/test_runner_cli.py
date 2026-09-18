"""Black-box tests for ``python -m module3.evaluation.runner``."""

import json
import subprocess
import sys


def _write_scenario(path, assertion_type="SESSION_CREATED"):
    path.write_text(
        "scenario:\n"
        "  name: cli_case\n"
        "  sessions:\n"
        "    default: {}\n"
        "  events: []\n"
        "  trace_assertions:\n"
        f"    - entry_type: {assertion_type}\n"
    )


def test_runner_cli_prints_and_saves_passing_report(tmp_path):
    scenario = tmp_path / "pass.yaml"
    report = tmp_path / "out" / "report.json"
    _write_scenario(scenario)
    completed = subprocess.run(
        [sys.executable, "-m", "module3.evaluation.runner", "--scenario", str(scenario), "--output", str(report)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "[PASSED]" in completed.stdout
    assert json.loads(report.read_text())["passed"] is True


def test_runner_cli_returns_nonzero_for_failed_assertion_and_bad_path(tmp_path):
    scenario = tmp_path / "fail.yaml"
    _write_scenario(scenario, "MISSING")
    failed = subprocess.run(
        [sys.executable, "-m", "module3.evaluation.runner", "--scenario", str(scenario)],
        capture_output=True,
        text=True,
        check=False,
    )
    missing = subprocess.run(
        [sys.executable, "-m", "module3.evaluation.runner", "--scenario", str(tmp_path / "missing.yaml")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1 and "[FAILED]" in failed.stdout
    assert missing.returncode == 2 and "PRISM evaluation failed" in missing.stderr
