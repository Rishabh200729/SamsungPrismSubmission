"""Stress realistic interrupt/tool integration through the public contract."""

import asyncio
import uuid

import pytest

# Fake modules deliberately import only public Module 3 surfaces.
from module3.runtime import IRuntimeClient, Runtime, TEST_CONFIG
from module3.runtime.events import (
    InputEventType,
    make_end_of_turn,
    make_final_response,
    make_interruption,
    make_text_chunk,
    make_tool_call,
    make_tool_result,
)


class FakeModule1:
    def __init__(self, runtime: IRuntimeClient) -> None:
        self.runtime = runtime
        self.interruptions_handled = 0

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.INTERRUPTION, self.on_interruption)

    async def on_interruption(self, event) -> None:
        self.interruptions_handled += 1
        await self.runtime.request_cancellation(event.session_id, reason="user_barge_in")


class FakeModule2:
    """Slow path that rejects late tool results through IRuntimeClient."""

    def __init__(self, runtime: IRuntimeClient) -> None:
        self.runtime = runtime
        self.tasks: dict[str, str] = {}
        self.calls: dict[str, str] = {}
        self.gates: dict[str, asyncio.Event] = {}
        self.stale_discards = 0
        self.final_responses = 0

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.TOOL_RESULT, self.on_tool_result)

    async def start_tool(self, session_id: str, label: str) -> tuple[str, str]:
        started = asyncio.Event()
        gate = asyncio.Event()
        call_id = str(uuid.uuid4())

        async def slow_work() -> None:
            started.set()
            await gate.wait()

        task_id = await self.runtime.create_background_task(
            session_id,
            slow_work(),
            task_type=f"fake_module_2_{label}",
            call_id=call_id,
        )
        await started.wait()
        self.tasks[call_id] = task_id
        self.calls[task_id] = call_id
        self.gates[task_id] = gate
        tool_call, emitted_call_id = make_tool_call(
            session_id,
            self.runtime.clock.now(),
            tool_name=f"tool_{label}",
            call_id=call_id,
            task_id=task_id,
        )
        assert emitted_call_id == call_id
        await self.runtime.emit_output(tool_call)
        return task_id, call_id

    async def on_tool_result(self, event) -> None:
        task_id = self.tasks.get(event.call_id or "")
        if task_id is None:
            return
        if self.runtime.is_stale_result(event.session_id, task_id):
            await self.runtime.handle_stale_result(event.session_id, task_id, event.call_id)
            self.stale_discards += 1
            return

        self.gates[task_id].set()
        self.final_responses += 1
        await self.runtime.emit_output(
            make_final_response(
                event.session_id,
                event.timestamp_ms,
                f"Authoritative result for {event.call_id}",
                task_id=task_id,
                generation=self.runtime.get_current_generation(event.session_id),
            )
        )


async def _until(predicate, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Timed out waiting for asynchronous integration state")


def _index_after(entries, entry_type: str, start: int = -1) -> int:
    return next(index for index, entry in enumerate(entries) if index > start and entry.entry_type == entry_type)


@pytest.mark.parametrize("run", range(20))
async def test_realistic_interrupt_tool_flow_is_causal_and_session_isolated(run: int):
    runtime = Runtime(config=TEST_CONFIG)
    module1 = FakeModule1(runtime)
    module2 = FakeModule2(runtime)
    module1.install()
    module2.install()
    assert isinstance(runtime, IRuntimeClient)

    await runtime.start()
    session_a = runtime.create_session(metadata={"name": "A", "run": run})
    session_b = runtime.create_session(metadata={"name": "B", "run": run})
    try:
        await runtime.submit_event(make_text_chunk(session_a, 0, "Book Tool A"))
        task_a, call_a = await module2.start_tool(session_a, "A")
        task_session_b, _ = await module2.start_tool(session_b, "session_b")
        assert not runtime.is_stale_result(session_a, task_a)
        assert not runtime.is_stale_result(session_b, task_session_b)

        # A single external interruption is handled once by Fake Module 1.
        await runtime.submit_event(make_interruption(session_a, 10, "change request"))
        await _until(lambda: module1.interruptions_handled == 1)
        assert runtime.get_current_generation(session_a) == 2
        assert runtime.get_current_generation(session_b) == 1
        assert runtime.is_stale_result(session_a, task_a)
        assert not runtime.is_stale_result(session_b, task_session_b)

        # Tool A returns after its generation was invalidated. It is discarded,
        # including when the same delivery is retried.
        late_a = make_tool_result(session_a, 20, call_a, task_a, result={"old": True})
        await runtime.submit_event(late_a)
        await _until(lambda: module2.stale_discards == 1)
        assert module2.final_responses == 0
        await runtime.submit_event(late_a.model_copy(update={"event_id": str(uuid.uuid4())}))
        for _ in range(10):
            await asyncio.sleep(0)
        assert module2.stale_discards == 2  # external deliveries were both seen

        stale_entries = [
            entry for entry in runtime.tracer.entries()
            if entry.session_id == session_a and entry.entry_type == "STALE_RESULT_DISCARDED"
        ]
        assert len(stale_entries) == 1
        assert stale_entries[0].task_id == task_a and stale_entries[0].call_id == call_a

        # Corrected Tool B belongs to generation 2 and may become authoritative.
        task_b, call_b = await module2.start_tool(session_a, "B")
        result_b = make_tool_result(session_a, 30, call_b, task_b, result={"corrected": True})
        await runtime.submit_event(result_b)
        await _until(lambda: module2.final_responses == 1)
        assert not runtime.is_stale_result(session_a, task_b)

        trace_a = [entry for entry in runtime.tracer.entries() if entry.session_id == session_a]
        sequence = [
            "TEXT_CHUNK", "TASK_CREATED", "TASK_STARTED", "TOOL_CALL",
            "INTERRUPTION", "TASK_CANCEL_REQUESTED", "GENERATION_INVALIDATED",
            "TOOL_RESULT", "STALE_RESULT_DISCARDED", "TASK_CREATED",
            "TOOL_CALL", "TOOL_RESULT", "FINAL_RESPONSE",
        ]
        position = -1
        for entry_type in sequence:
            position = _index_after(trace_a, entry_type, position)

        assert sum(entry.entry_type == "GENERATION_INVALIDATED" for entry in trace_a) == 1
        assert sum(entry.entry_type == "FINAL_RESPONSE" for entry in trace_a) == 1
    finally:
        await runtime.close_session(session_a)
        await runtime.close_session(session_b)
        await runtime.stop()
