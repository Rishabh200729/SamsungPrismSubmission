"""
module3/evaluation/runner.py

ScenarioRunner — executes a scenario against a Runtime instance.

The runner:
1. Creates sessions (one per scenario session key)
2. Schedules events on the VirtualClock
3. Starts the runtime dispatch loop
4. Advances virtual time event-by-event
5. Collects all outputs and trace entries
6. Returns a ScenarioResult for evaluation

The runner is model-agnostic — it injects protocol events and observes
the runtime's behavior. Any registered handlers (from Modules 1/2/4)
process the events; the runner only injects and collects.

Inspired by Gaia2: scenario = environment spec driving reproducible evaluation.
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from ..runtime.clock.virtual_clock import VirtualClock
from ..runtime.config import TEST_CONFIG, RuntimeConfig
from ..runtime.events.base import BaseEvent, EventCategory, InputEventType
from ..runtime.events import (
    make_text_chunk, make_end_of_turn, make_interruption,
    make_tool_result, make_tool_manifest, make_audio_wav, make_video_frame,
)
from ..runtime.runtime import Runtime
from ..runtime.tracing.recorder import TraceEntry
from .scenario import Scenario, ScenarioEvent

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario_name: str
    session_ids: dict[str, str]     # session_key -> session_id
    outputs: list[BaseEvent]
    trace: list[TraceEntry]
    final_virtual_time_ms: float
    passed: bool = False
    assertion_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class ScenarioRunner:
    """
    Executes a scenario deterministically using VirtualClock.

    Usage::

        runner = ScenarioRunner()
        result = await runner.run(scenario)
        # or with custom handlers:
        runner.register_handler(InputEventType.TEXT_CHUNK, my_handler)
        result = await runner.run(scenario)
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        extra_handlers: dict[Any, Any] | None = None,
    ) -> None:
        self._config = config or TEST_CONFIG
        self._extra_handlers = extra_handlers or {}

    async def run(self, scenario: Scenario) -> ScenarioResult:
        """
        Execute a scenario and return the result.

        Creates a fresh Runtime for each scenario run.
        """
        clock = VirtualClock(start_ms=0.0)
        runtime = Runtime(config=self._config, clock=clock)

        # Register any extra handlers provided by other modules
        for event_type, handler in self._extra_handlers.items():
            runtime.register_handler(event_type, handler)

        await runtime.start()

        # Create sessions
        session_ids: dict[str, str] = {}
        for key, metadata in scenario.sessions.items():
            sid = runtime.create_session(metadata=metadata)
            session_ids[key] = sid

        try:
            # Sort events by timestamp
            sorted_events = sorted(scenario.events, key=lambda e: e.at_ms)

            # Schedule all events on the virtual clock
            for scenario_event in sorted_events:
                sid = session_ids.get(scenario_event.session_key, list(session_ids.values())[0])
                clock.schedule(
                    lambda se=scenario_event, s=sid: asyncio.ensure_future(
                        self._inject_event(runtime, se, s)
                    ),
                    at_ms=scenario_event.at_ms,
                )

            # Advance clock to max_duration, firing events in order
            await clock.advance_to_async(scenario.max_duration_ms)

            # Let all background tasks settle
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        except Exception as exc:
            logger.error("ScenarioRunner: error during scenario %s: %r", scenario.name, exc)
            return ScenarioResult(
                scenario_name=scenario.name,
                session_ids=session_ids,
                outputs=[],
                trace=runtime.tracer.entries(),
                final_virtual_time_ms=clock.now(),
                passed=False,
                error=str(exc),
            )
        finally:
            await runtime.stop()

        outputs = await runtime.output_queue.drain_all()
        trace = runtime.tracer.entries()

        return ScenarioResult(
            scenario_name=scenario.name,
            session_ids=session_ids,
            outputs=outputs,
            trace=trace,
            final_virtual_time_ms=clock.now(),
            passed=False,  # Populated by assertions
        )

    async def _inject_event(
        self,
        runtime: Runtime,
        scenario_event: ScenarioEvent,
        session_id: str,
    ) -> None:
        """Build and submit a BaseEvent from a ScenarioEvent spec."""
        ts = scenario_event.at_ms
        et = scenario_event.event_type
        p = scenario_event.payload

        common_kwargs: dict[str, Any] = {}
        if scenario_event.call_id:
            common_kwargs["call_id"] = scenario_event.call_id
        if scenario_event.task_id:
            common_kwargs["task_id"] = scenario_event.task_id
        if scenario_event.generation is not None:
            common_kwargs["generation"] = scenario_event.generation

        event: BaseEvent | None = None

        if et == InputEventType.TEXT_CHUNK.value:
            event = make_text_chunk(
                session_id=session_id,
                timestamp_ms=ts,
                text=p.get("text", ""),
                is_partial=p.get("is_partial", False),
                **common_kwargs,
            )
        elif et == InputEventType.END_OF_TURN.value:
            event = make_end_of_turn(
                session_id=session_id,
                timestamp_ms=ts,
                final_text=p.get("final_text"),
                **common_kwargs,
            )
        elif et == InputEventType.INTERRUPTION.value:
            event = make_interruption(
                session_id=session_id,
                timestamp_ms=ts,
                text=p.get("text"),
                reason=p.get("reason", "user_interruption"),
                **common_kwargs,
            )
        elif et == InputEventType.TOOL_RESULT.value:
            event = make_tool_result(
                session_id=session_id,
                timestamp_ms=ts,
                call_id=p.get("call_id", scenario_event.call_id or "unknown"),
                task_id=p.get("task_id", scenario_event.task_id or "unknown"),
                result=p.get("result"),
                success=p.get("success", True),
                error=p.get("error"),
                **{k: v for k, v in common_kwargs.items() if k not in ("call_id", "task_id")},
            )
        elif et == InputEventType.TOOL_MANIFEST.value:
            event = make_tool_manifest(
                session_id=session_id,
                timestamp_ms=ts,
                tools=p.get("tools", []),
                **common_kwargs,
            )
        elif et == InputEventType.AUDIO_WAV.value:
            event = make_audio_wav(
                session_id=session_id,
                timestamp_ms=ts,
                audio_b64=p.get("audio_b64", ""),
                duration_ms=p.get("duration_ms", 100.0),
                **common_kwargs,
            )
        elif et == InputEventType.VIDEO_FRAME.value:
            event = make_video_frame(
                session_id=session_id,
                timestamp_ms=ts,
                frame_b64=p.get("frame_b64", ""),
                width=p.get("width", 640),
                height=p.get("height", 480),
                frame_index=p.get("frame_index", 0),
                **common_kwargs,
            )
        else:
            logger.warning("ScenarioRunner: unknown event_type=%s at %.1fms", et, ts)
            return

        await runtime.submit_event(event)
        logger.debug("ScenarioRunner: injected %s at %.1fms session=%s", et, ts, session_id)


async def _run_cli(scenario_path: str, output_path: str | None) -> int:
    """Run a declarative scenario and print/save its evaluation report."""
    from .assertions import run_assertions
    from .metrics import compute_metrics
    from .report import build_report
    from .scenario import Scenario

    try:
        scenario = Scenario.from_yaml(scenario_path)
        result = await ScenarioRunner().run(scenario)
        metrics = compute_metrics(result)
        assertions = run_assertions(scenario, result)
        report = build_report(scenario.name, result, metrics, assertions)
        report.print_human_readable()
        if output_path:
            report.save_json(output_path)
        return 0 if report.passed else 1
    except Exception as exc:
        print(f"PRISM evaluation failed: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Module 3 YAML evaluation scenario.")
    parser.add_argument("--scenario", required=True, help="Path to the YAML scenario")
    parser.add_argument("--output", help="Optional path for the JSON evaluation report")
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args.scenario, args.output))


if __name__ == "__main__":
    raise SystemExit(main())
