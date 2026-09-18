"""
module3/tests/test_cancellation.py

Tests for the generation-based cancellation coordinator and stale-result protection.
"""
import asyncio
import pytest

from module3.runtime.tasks.cancellation import CancellationCoordinator
from module3.runtime.tasks.lifecycle import TaskStatus
from module3.runtime.tasks.registry import TaskRegistry


@pytest.fixture
def registry():
    return TaskRegistry(session_id="cancel-test-session")


@pytest.fixture
def coord(registry):
    return CancellationCoordinator(session_id="cancel-test-session", registry=registry)


class TestCancellationCoordinator:
    def test_initial_generation(self, coord):
        assert coord.current_generation == 1

    def test_increment_generation(self, coord):
        old, new = coord.increment_generation(reason="interruption")
        assert old == 1
        assert new == 2
        assert coord.current_generation == 2

    def test_multiple_increments(self, coord):
        coord.increment_generation()
        coord.increment_generation()
        old, new = coord.increment_generation()
        assert old == 3
        assert new == 4

    def test_is_stale_with_old_generation(self, registry, coord):
        record = registry.create_task("test", generation=1)
        assert not coord.is_stale(record)   # generation 1 == current 1
        coord.increment_generation()         # current becomes 2
        assert coord.is_stale(record)        # generation 1 < current 2

    def test_is_not_stale_with_current_generation(self, registry, coord):
        coord.increment_generation()  # current = 2
        record = registry.create_task("test", generation=2)
        assert not coord.is_stale(record)

    def test_is_stale_by_id(self, registry, coord):
        record = registry.create_task("test", generation=1)
        coord.increment_generation()
        assert coord.is_stale_by_id(record.task_id)

    def test_is_stale_call(self, registry, coord):
        record = registry.create_task("test", generation=1, call_id="c1")
        coord.increment_generation()
        assert coord.is_stale_call("c1")

    async def test_cancel_task_transitions_to_cancellation_requested(self, registry, coord):
        record = registry.create_task("test", generation=1)
        registry.transition(record.task_id, TaskStatus.RUNNING)
        result = await coord.cancel_task(record.task_id, timestamp_ms=50.0)
        assert result is True
        assert registry.get(record.task_id).status == TaskStatus.CANCELLATION_REQUESTED

    async def test_cancel_terminal_task_returns_false(self, registry, coord):
        record = registry.create_task("test", generation=1)
        registry.transition(record.task_id, TaskStatus.RUNNING)
        registry.transition(record.task_id, TaskStatus.COMPLETED)
        result = await coord.cancel_task(record.task_id)
        assert result is False

    async def test_cancel_unknown_task_returns_false(self, coord):
        result = await coord.cancel_task("nonexistent-task-id")
        assert result is False

    async def test_cancel_all_active(self, registry, coord):
        r1 = registry.create_task("t1", generation=1)
        r2 = registry.create_task("t2", generation=1)
        r3 = registry.create_task("t3", generation=1)
        registry.transition(r1.task_id, TaskStatus.RUNNING)
        registry.transition(r2.task_id, TaskStatus.RUNNING)
        registry.transition(r3.task_id, TaskStatus.RUNNING)
        registry.transition(r3.task_id, TaskStatus.COMPLETED)  # r3 is terminal

        cancelled = await coord.cancel_all_active(timestamp_ms=100.0)
        assert len(cancelled) == 2
        assert r3.task_id not in cancelled

    async def test_handle_stale_result_marks_task_stale(self, registry, coord):
        record = registry.create_task("test", generation=1)
        registry.transition(record.task_id, TaskStatus.RUNNING)
        coord.increment_generation()  # current=2, task is gen 1 -> stale

        stale_events = []

        async def trace_cb(**kwargs):
            stale_events.append(kwargs)

        coord._trace_callback = trace_cb
        await coord.handle_stale_result(record.task_id, "c1", timestamp_ms=200.0)

        assert registry.get(record.task_id).status == TaskStatus.STALE
        assert len(stale_events) == 1
        assert stale_events[0]["task_generation"] == 1
        assert stale_events[0]["current_generation"] == 2

    async def test_confirm_cancelled(self, registry, coord):
        record = registry.create_task("test", generation=1)
        registry.transition(record.task_id, TaskStatus.RUNNING)
        await coord.cancel_task(record.task_id)
        await coord.confirm_cancelled(record.task_id, timestamp_ms=100.0)
        assert registry.get(record.task_id).status == TaskStatus.CANCELLED
