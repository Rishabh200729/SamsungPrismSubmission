"""Additional runner paths that are not exercised by bundled YAML files."""

import asyncio

from module3.evaluation import runner as runner_module
from module3.evaluation.runner import ScenarioRunner
from module3.evaluation.scenario import Scenario, ScenarioEvent
from module3.runtime.config import TEST_CONFIG
from module3.runtime.runtime import Runtime


async def test_runner_injects_all_supported_protocol_event_types_and_ignores_unknown():
    runtime = Runtime(config=TEST_CONFIG)
    await runtime.start()
    sid = runtime.create_session()
    runner = ScenarioRunner()
    events = [
        ScenarioEvent(1, "TOOL_RESULT", payload={"call_id": "c", "task_id": "t", "result": {}}),
        ScenarioEvent(2, "TOOL_MANIFEST", payload={"tools": []}),
        ScenarioEvent(3, "AUDIO_WAV", payload={"audio_b64": "", "duration_ms": 2}),
        ScenarioEvent(4, "VIDEO_FRAME", payload={"frame_b64": "", "width": 1, "height": 1}),
        ScenarioEvent(5, "UNKNOWN"),
    ]
    for event in events:
        await runner._inject_event(runtime, event, sid)
    await runtime.input_queue.join()
    types = [entry.entry_type for entry in runtime.tracer.entries()]
    assert {"TOOL_RESULT", "TOOL_MANIFEST", "AUDIO_WAV", "VIDEO_FRAME"}.issubset(types)
    await runtime.stop()


async def test_runner_returns_error_result_when_scenario_cannot_schedule():
    result = await ScenarioRunner().run(Scenario(name="bad", sessions={}, events=[ScenarioEvent(0, "TEXT_CHUNK")]))
    assert result.error and not result.passed


def test_cli_helpers_are_directly_covered(tmp_path, capsys):
    path = tmp_path / "scenario.yaml"
    path.write_text("scenario:\n  name: direct_cli\n  sessions:\n    default: {}\n")
    assert runner_module.main(["--scenario", str(path)]) == 0
    assert "[PASSED]" in capsys.readouterr().out
