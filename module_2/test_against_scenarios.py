"""
test_against_scenarios.py
Runs YOUR Module2ReasoningAdapter against Rishabh's real scenario YAML
files (basic.yaml, interruption.yaml, concurrency.yaml, stale_result.yaml)
instead of the generic fast/slow handlers his own test_scenarios.py uses.

This is the actual dress rehearsal: does your module survive being driven
by the real scenario files the harness will eventually grade against,
not just the events you hand-crafted yourself.

IMPORTANT CAVEAT: none of these YAML files submit a TOOL_MANIFEST event —
they were written to test Module 3's own runtime behavior, not Module 2's
tool orchestration. So this test injects a manifest itself before running
each scenario's events, using SAMPLE_MANIFEST below. Replace it with
whatever manifest the real harness will actually declare once that's
known — this is a best-effort stand-in, not a guarantee your module will
be fed exactly these tools in the graded run.

Run: python3 -m unittest test_against_scenarios -v
"""

import asyncio
import unittest
from pathlib import Path

try:
    import module3
    from module3.runtime.runtime import Runtime
    from module3.runtime.config import TEST_CONFIG
    from module3.runtime.clock.virtual_clock import VirtualClock
    from module3.runtime.events import make_tool_manifest
    from module3.runtime.events.base import InputEventType
    from module3.evaluation.scenario import Scenario
    from module3.evaluation.runner import ScenarioRunner, ScenarioResult
    from module3.evaluation.assertions import run_assertions, all_passed
    HAVE_MODULE3 = True
except ImportError:
    HAVE_MODULE3 = False

from orchestrator import Module2ReasoningAdapter

SAMPLE_MANIFEST = [
    {"name": "book_flight", "side_effect": "STATE_MODIFYING",
     "parameters": {"destination": {"required": True}, "date": {"required": True}}},
    {"name": "book_hotel", "side_effect": "STATE_MODIFYING",
     "parameters": {"city": {"required": True}}},
    {"name": "get_flight_status", "side_effect": "READ_ONLY",
     "parameters": {"flight_id": {"required": True}}},
]


async def run_scenario_with_module2(yaml_path: Path, manifest_payload=SAMPLE_MANIFEST):
    """
    Mirrors what Rishabh's own test_yaml_scenario_interruption does, but
    wires YOUR adapter instead of his generic fast/interrupt handlers, and
    also wires the generic interruption handler he uses everywhere (since
    that's Module 1's job, not yours — Module 2 doesn't own interruption
    handling, only reacting correctly once Module 3 has already fenced it).
    """
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    adapter = Module2ReasoningAdapter(rt)
    adapter.install()

    async def interrupt_handler(event):
        if event.session_id in rt._session_manager.all_session_ids():
            await rt.request_cancellation(event.session_id)

    rt.register_handler(InputEventType.INTERRUPTION, interrupt_handler)
    await rt.start()

    scenario = Scenario.from_yaml(yaml_path)
    session_ids = {}
    for key, meta in scenario.sessions.items():
        sid = rt.create_session(metadata=meta)
        session_ids[key] = sid
        # Inject the manifest before any scenario events run, so your
        # tool_manifest.py has something to classify against.
        await rt.submit_event(make_tool_manifest(sid, clock.now(), tools=manifest_payload))

    runner = ScenarioRunner(config=TEST_CONFIG)
    sorted_events = sorted(scenario.events, key=lambda e: e.at_ms)
    for se in sorted_events:
        sid = session_ids.get(se.session_key, list(session_ids.values())[0])
        await runner._inject_event(rt, se, sid)
        for _ in range(3):
            await asyncio.sleep(0)

    await asyncio.sleep(0.05)
    await rt.stop()

    outputs = await rt.output_queue.drain_all()
    result = ScenarioResult(
        scenario_name=scenario.name,
        session_ids=session_ids,
        outputs=outputs,
        trace=rt.tracer.entries(),
        final_virtual_time_ms=clock.now(),
    )
    return scenario, result


@unittest.skipUnless(HAVE_MODULE3, "module3 package not on path yet")
class TestModule2AgainstRealScenarios(unittest.TestCase):
    SCENARIOS_DIR = Path(module3.__file__).parent / "scenarios" if HAVE_MODULE3 else None

    def _run(self, filename):
        scenario, result = asyncio.run(
            run_scenario_with_module2(self.SCENARIOS_DIR / filename)
        )
        # His own scenario assertions must still pass with your adapter
        # wired in — if they don't, your module broke behavior that has
        # nothing to do with you, which is itself worth knowing.
        assertions = run_assertions(scenario, result)
        failed = [a for a in assertions if not a.passed]
        self.assertTrue(all_passed(assertions), failed)
        return result

    def test_basic_scenario_survives_module2(self):
        self._run("basic.yaml")

    def test_interruption_scenario_survives_module2(self):
        result = self._run("interruption.yaml")
        # Soft check: your adapter should have produced SOME output
        # (a CLARIFICATION, since "book a flight" alone is missing the
        # date slot) rather than silently doing nothing.
        event_types = {o.event_type.value for o in result.outputs}
        self.assertTrue(
            "CLARIFICATION" in event_types or "TOOL_CALL" in event_types,
            f"Expected Module 2 to react somehow; got only {event_types}",
        )

    def test_concurrency_scenario_survives_module2(self):
        self._run("concurrency.yaml")

    def test_stale_result_scenario_survives_module2(self):
        # This one deliberately injects a TOOL_RESULT for a placeholder
        # call_id/task_id that was never actually created by your adapter
        # — proves on_tool_result's is_stale_result check doesn't crash
        # on a completely unknown task_id (Module 3 treats unknown as
        # stale per its own test_scenario_13).
        self._run("stale_result.yaml")


if __name__ == "__main__":
    unittest.main()
