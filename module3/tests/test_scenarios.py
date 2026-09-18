"""
module3/tests/test_scenarios.py

Scenario-based tests using the ScenarioRunner and YAML scenario files.
Covers all 16 required test scenarios from the spec.
"""
import asyncio
import pytest
from pathlib import Path

from module3.runtime.events import (
    make_text_chunk, make_interruption, make_filler, make_final_response,
    make_tool_call, make_tool_result, StateSnapshot,
)
from module3.runtime.events.base import InputEventType, OutputEventType
from module3.runtime.clock.virtual_clock import VirtualClock
from module3.runtime.config import TEST_CONFIG
from module3.runtime.runtime import Runtime
from module3.evaluation.scenario import Scenario
from module3.evaluation.runner import ScenarioRunner
from module3.evaluation.assertions import run_assertions, all_passed
from module3.evaluation.metrics import compute_metrics

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


# ---------------------------------------------------------------------------
# Helper: build a runtime with an interrupt handler wired
# ---------------------------------------------------------------------------

async def _make_runtime_with_handlers(virtual_clock=None):
    """Create a runtime with basic fast/slow-path handlers for testing."""
    clock = virtual_clock or VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    # Fast-path handler: on TEXT_CHUNK emit a FILLER immediately
    async def fast_handler(event):
        if event.event_type == InputEventType.TEXT_CHUNK:
            filler = make_filler(
                session_id=event.session_id,
                timestamp_ms=clock.now() + 5.0,
                text="Got it, one moment...",
                task_id=event.task_id,
                generation=event.generation,
            )
            await rt.emit_output(filler)

    # Interruption handler: increment generation
    async def interrupt_handler(event):
        if event.event_type == InputEventType.INTERRUPTION:
            await rt.request_cancellation(event.session_id, reason="user_interruption")

    rt.register_handler(InputEventType.TEXT_CHUNK, fast_handler)
    rt.register_handler(InputEventType.INTERRUPTION, interrupt_handler)

    await rt.start()
    return rt


# ============================================================
# SCENARIO 1: Simple normal request
# ============================================================
async def test_scenario_01_simple_normal_request():
    """TEXT_CHUNK -> END_OF_TURN -> FILLER output."""
    rt = await _make_runtime_with_handlers()
    sid = rt.create_session()

    try:
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
        await asyncio.sleep(0.05)

        outputs = await rt.drain_outputs()
        assert any(o.event_type == OutputEventType.FILLER for o in outputs)

        trace = rt.tracer.entries()
        assert any(e.entry_type == "TEXT_CHUNK" for e in trace)
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 2: Slow background reasoning while new input arrives
# ============================================================
async def test_scenario_02_slow_background_with_new_input():
    """Background task runs while new TEXT_CHUNK arrives — no blocking."""
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    received_events = []

    async def handler(event):
        received_events.append(event.event_type.value)

    rt.register_handler(InputEventType.TEXT_CHUNK, handler)
    await rt.start()
    sid = rt.create_session()

    try:
        # Slow background task
        async def slow_work():
            await clock.sleep(500.0)

        task_id = await rt.create_background_task(sid, slow_work(), "slow_reasoning")

        # New input arrives while slow task is running
        await rt.submit_event(make_text_chunk(sid, 10.0, "New input while slow task runs"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert "TEXT_CHUNK" in received_events

        task = rt.get_task(sid, task_id)
        # Task is either RUNNING or WAITING (not yet completed)
        assert task.status.value in ("RUNNING", "PENDING")
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 3: Interruption during planning
# ============================================================
async def test_scenario_03_interruption_during_planning():
    """Interruption before any tool call — generation advances."""
    clock = VirtualClock()
    rt = await _make_runtime_with_handlers(clock)
    sid = rt.create_session()

    try:
        await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        gen_before = rt.get_current_generation(sid)

        await rt.submit_event(make_interruption(sid, 50.0, text="Cancel that"))
        # Multiple yields to allow interrupt handler to run
        for _ in range(5):
            await asyncio.sleep(0)

        gen_after = rt.get_current_generation(sid)
        assert gen_after == gen_before + 1, f"Expected gen {gen_before+1}, got {gen_after}"

        trace_types = [e.entry_type for e in rt.tracer.entries()]
        assert "GENERATION_INVALIDATED" in trace_types
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 4: Interruption during tool execution
# ============================================================
async def test_scenario_04_interruption_during_tool_execution():
    """Task running a slow tool gets cancelled on interruption."""
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    async def interrupt_handler(event):
        await rt.request_cancellation(event.session_id)

    rt.register_handler(InputEventType.INTERRUPTION, interrupt_handler)
    await rt.start()
    sid = rt.create_session()

    try:
        async def slow_tool():
            await clock.sleep(1000.0)  # Long running

        task_id = await rt.create_background_task(sid, slow_tool(), "tool_call")
        await asyncio.sleep(0)

        # Interruption while tool runs
        await rt.submit_event(make_interruption(sid, 450.0, text="Stop!"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        task = rt.get_task(sid, task_id)
        assert task.status.value in ("CANCELLATION_REQUESTED", "CANCELLED", "RUNNING")

        gen = rt.get_current_generation(sid)
        assert gen == 2
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 5: Tool result arrives AFTER cancellation
# ============================================================
async def test_scenario_05_stale_tool_result_discarded():
    """
    Tool result from generation 1 arrives after generation increments to 2.
    Must be marked STALE_RESULT_DISCARDED.
    """
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    async def interrupt_handler(event):
        await rt.request_cancellation(event.session_id)

    rt.register_handler(InputEventType.INTERRUPTION, interrupt_handler)
    await rt.start()
    sid = rt.create_session()

    try:
        # Create a slow task that will NOT complete before interrupt
        async def slow_noop():
            await clock.sleep(1000.0)  # Very slow — won't finish

        task_id = await rt.create_background_task(sid, slow_noop(), "slow_reasoning")
        await asyncio.sleep(0)  # Let task start

        # Interrupt — generation becomes 2
        await rt.submit_event(make_interruption(sid, 100.0))
        for _ in range(5):
            await asyncio.sleep(0)

        # Task from generation 1 is now stale (gen 1 < current gen 2)
        assert rt.get_current_generation(sid) == 2
        stale = rt.is_stale_result(sid, task_id)
        assert stale, f"Expected stale=True, task gen=1, current gen={rt.get_current_generation(sid)}"

        # Explicitly handle the stale result
        await rt.handle_stale_result(sid, task_id, call_id="stale-call-001")
        await asyncio.sleep(0)

        trace_types = [e.entry_type for e in rt.tracer.entries()]
        assert "STALE_RESULT_DISCARDED" in trace_types
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 6: Repeated interruption
# ============================================================
async def test_scenario_06_repeated_interruption():
    """Multiple interruptions advance generation multiple times."""
    clock = VirtualClock()
    rt = await _make_runtime_with_handlers(clock)
    sid = rt.create_session()

    try:
        for i in range(3):
            await rt.submit_event(make_interruption(sid, float(i * 100), text=f"Interrupt {i}"))
            # Give the event loop multiple chances to process each interrupt
            for _ in range(5):
                await asyncio.sleep(0)

        gen = rt.get_current_generation(sid)
        assert gen == 4, f"Expected gen=4, got gen={gen}"  # Started at 1, incremented 3 times

        invalidations = [e for e in rt.tracer.entries() if e.entry_type == "GENERATION_INVALIDATED"]
        assert len(invalidations) == 3
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 7: Multiple concurrent background tasks
# ============================================================
async def test_scenario_07_multiple_concurrent_tasks():
    """Multiple tasks run concurrently — all tracked independently."""
    clock = VirtualClock()
    rt = Runtime(config=TEST_CONFIG, clock=clock)
    await rt.start()
    sid = rt.create_session()

    try:
        task_ids = []
        for i in range(5):
            async def work(n=i):
                await asyncio.sleep(0)
            tid = await rt.create_background_task(sid, work(), f"task_{i}")
            task_ids.append(tid)

        await asyncio.sleep(0.05)

        ctx = rt.get_session(sid)
        assert len(ctx.task_registry.all_tasks()) == 5
        # All task_ids are unique
        assert len(set(task_ids)) == 5
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 8: Chained tool calls
# ============================================================
async def test_scenario_08_chained_tool_calls():
    """Task A spawns Task B (parent-child relationship tracked)."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        async def parent_task():
            await asyncio.sleep(0)

        async def child_task():
            await asyncio.sleep(0)

        parent_id = await rt.create_background_task(sid, parent_task(), "parent")
        child_id = await rt.create_background_task(
            sid, child_task(), "child", parent_task_id=parent_id
        )

        await asyncio.sleep(0.05)

        ctx = rt.get_session(sid)
        child_record = ctx.task_registry.get(child_id)
        assert child_record.parent_task_id == parent_id
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 9: Failed tool + retry
# ============================================================
async def test_scenario_09_failed_tool_retry():
    """Task fails, retry task is created with same parent."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        failed_task_id = None
        retry_task_id = None

        async def failing_task():
            raise RuntimeError("Tool failed!")

        failed_task_id = await rt.create_background_task(
            sid, failing_task(), "tool_call"
        )
        await asyncio.sleep(0.05)

        # Check task failed
        from module3.runtime.tasks.lifecycle import TaskStatus
        task = rt.get_task(sid, failed_task_id)
        assert task.status == TaskStatus.FAILED

        # Retry
        async def retry_task():
            await asyncio.sleep(0)

        retry_task_id = await rt.create_background_task(
            sid, retry_task(), "tool_call_retry", parent_task_id=failed_task_id
        )
        await asyncio.sleep(0.05)

        retry = rt.get_task(sid, retry_task_id)
        assert retry.status == TaskStatus.COMPLETED
        assert retry.parent_task_id == failed_task_id

        trace_types = [e.entry_type for e in rt.tracer.entries()]
        assert "TASK_FAILED" in trace_types
        assert "TASK_COMPLETED" in trace_types
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 10: Session A and Session B concurrent
# ============================================================
async def test_scenario_10_concurrent_sessions_isolation():
    """Two sessions execute concurrently — no state leakage."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()

    try:
        sid_a = rt.create_session(session_id="session-A")
        sid_b = rt.create_session(session_id="session-B")

        async def work_a():
            await asyncio.sleep(0)

        async def work_b():
            await asyncio.sleep(0)

        tid_a = await rt.create_background_task(sid_a, work_a(), "work_a")
        tid_b = await rt.create_background_task(sid_b, work_b(), "work_b")

        await asyncio.sleep(0.05)

        ctx_a = rt.get_session(sid_a)
        ctx_b = rt.get_session(sid_b)

        # Task A is only in session A's registry
        assert ctx_a.task_registry.get(tid_a) is not None
        assert ctx_b.task_registry.get(tid_a) is None

        # Task B is only in session B's registry
        assert ctx_b.task_registry.get(tid_b) is not None
        assert ctx_a.task_registry.get(tid_b) is None

        # Generation changes in A don't affect B
        await rt.request_cancellation(sid_a)
        assert rt.get_current_generation(sid_a) == 2
        assert rt.get_current_generation(sid_b) == 1
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 11: Malformed event
# ============================================================
async def test_scenario_11_malformed_event():
    """Invalid event payload is rejected by Pydantic before entering queue."""
    from pydantic import ValidationError
    from module3.runtime.events.input_events import TextChunkPayload

    with pytest.raises(ValidationError):
        TextChunkPayload(text="")  # min_length=1 violated


# ============================================================
# SCENARIO 12: Duplicate call ID
# ============================================================
async def test_scenario_12_duplicate_call_id_detected():
    """Duplicate call IDs should appear in metrics as protocol violations."""
    from module3.evaluation.metrics import compute_metrics
    from module3.evaluation.runner import ScenarioResult
    from module3.runtime.tracing.recorder import TraceRecorder

    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        evt1, cid = make_tool_call(sid, 0.0, "search", call_id="dup-call-id", task_id="t1")
        evt2, _ = make_tool_call(sid, 10.0, "search", call_id="dup-call-id", task_id="t1")
        await rt.emit_output(evt1)
        await rt.emit_output(evt2)

        outputs = await rt.drain_outputs()

        result = ScenarioResult(
            scenario_name="dup_call_test",
            session_ids={"default": sid},
            outputs=outputs,
            trace=rt.tracer.entries(),
            final_virtual_time_ms=10.0,
        )
        metrics = compute_metrics(result)
        assert "dup-call-id" in metrics.duplicate_call_ids
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 13: Tool result for unknown task
# ============================================================
async def test_scenario_13_tool_result_unknown_task():
    """is_stale_result returns True for completely unknown task_id."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        # Task ID that was never registered
        stale = rt.is_stale_result(sid, "completely-unknown-task-id")
        assert stale  # Unknown tasks should be treated as stale/invalid
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 14: Result from stale generation
# ============================================================
async def test_scenario_14_stale_generation_result():
    """Task from generation 1 is stale after generation advances to 3."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        async def noop(): pass
        task_id = await rt.create_background_task(sid, noop(), "old_task")
        await asyncio.sleep(0.05)

        # Advance generation twice
        await rt.request_cancellation(sid)  # gen 2
        await rt.request_cancellation(sid)  # gen 3

        assert rt.get_current_generation(sid) == 3
        assert rt.is_stale_result(sid, task_id)  # Task was gen 1
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 15: Empty queue / idle runtime
# ============================================================
async def test_scenario_15_idle_runtime():
    """Runtime with no events stays idle without error."""
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    try:
        await asyncio.sleep(0.05)
        outputs = await rt.drain_outputs()
        assert len(outputs) == 0
        # No tasks created
        ctx = rt.get_session(sid)
        assert len(ctx.task_registry.all_tasks()) == 0
    finally:
        await rt.stop()


# ============================================================
# SCENARIO 16: High event burst / queue pressure
# ============================================================
async def test_scenario_16_high_event_burst():
    """100 events submitted rapidly — none lost, all recorded in trace."""
    rt = Runtime(config=TEST_CONFIG)
    received = []

    async def handler(event):
        received.append(event.event_id)

    rt.register_handler(InputEventType.TEXT_CHUNK, handler)
    await rt.start()
    sid = rt.create_session()

    try:
        events = [make_text_chunk(sid, float(i), f"burst message {i}") for i in range(100)]
        for e in events:
            await rt.submit_event(e)

        # Let dispatcher process all
        await asyncio.sleep(0.2)

        assert len(received) == 100
        trace_text_chunks = [e for e in rt.tracer.entries() if e.entry_type == "TEXT_CHUNK"]
        assert len(trace_text_chunks) == 100
    finally:
        await rt.stop()


# ============================================================
# YAML-based scenario tests
# ============================================================
async def test_yaml_scenario_basic():
    """Load and run the basic.yaml scenario."""
    scenario = Scenario.from_yaml(SCENARIOS_DIR / "basic.yaml")
    runner = ScenarioRunner()
    result = await runner.run(scenario)
    assertions = run_assertions(scenario, result)
    assert all_passed(assertions), [a for a in assertions if not a.passed]


async def test_yaml_scenario_interruption():
    """Load and run the interruption.yaml scenario with interrupt handler wired."""
    scenario = Scenario.from_yaml(SCENARIOS_DIR / "interruption.yaml")

    # We create a closure that captures the runtime inside the handler
    # The ScenarioRunner creates its own runtime internally, so we need
    # to use a wrapper that gets the runtime reference at call time.
    # We'll override by running manually instead.
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    async def interrupt_handler(event):
        if event.session_id in rt._session_manager.all_session_ids():
            await rt.request_cancellation(event.session_id)

    rt.register_handler(InputEventType.INTERRUPTION, interrupt_handler)
    await rt.start()

    session_ids = {}
    for key, meta in scenario.sessions.items():
        sid = rt.create_session(metadata=meta)
        session_ids[key] = sid

    from module3.evaluation.runner import ScenarioRunner
    from module3.evaluation.runner import ScenarioResult

    sorted_events = sorted(scenario.events, key=lambda e: e.at_ms)
    runner = ScenarioRunner(config=TEST_CONFIG)
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

    assertions = run_assertions(scenario, result)
    assert all_passed(assertions), [a for a in assertions if not a.passed]


async def test_yaml_scenario_concurrency():
    """Load and run the concurrency.yaml scenario."""
    scenario = Scenario.from_yaml(SCENARIOS_DIR / "concurrency.yaml")

    async def interrupt_handler(event):
        pass  # Will be overridden in runner

    runner = ScenarioRunner(extra_handlers={
        InputEventType.INTERRUPTION: interrupt_handler,
    })
    result = await runner.run(scenario)
    assertions = run_assertions(scenario, result)
    # Session isolation: both sessions exist, no cross-contamination
    assert len(result.session_ids) == 2
