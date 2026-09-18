"""
module3/tests/test_tasks.py

Tests for task lifecycle, registry, and state transitions.
"""
import pytest

from module3.runtime.tasks.lifecycle import (
    TaskStatus, ExecutionPhase, InvalidTransitionError, validate_transition
)
from module3.runtime.tasks.registry import TaskRegistry


class TestTaskStatus:
    def test_terminal_states(self):
        assert TaskStatus.COMPLETED.is_terminal()
        assert TaskStatus.CANCELLED.is_terminal()
        assert TaskStatus.FAILED.is_terminal()
        assert TaskStatus.STALE.is_terminal()
        assert not TaskStatus.PENDING.is_terminal()
        assert not TaskStatus.RUNNING.is_terminal()

    def test_valid_transitions(self):
        assert TaskStatus.PENDING.can_transition_to(TaskStatus.RUNNING)
        assert TaskStatus.RUNNING.can_transition_to(TaskStatus.COMPLETED)
        assert TaskStatus.RUNNING.can_transition_to(TaskStatus.WAITING)
        assert TaskStatus.WAITING.can_transition_to(TaskStatus.COMPLETED)
        assert TaskStatus.RUNNING.can_transition_to(TaskStatus.CANCELLATION_REQUESTED)
        assert TaskStatus.CANCELLATION_REQUESTED.can_transition_to(TaskStatus.CANCELLED)

    def test_invalid_transitions(self):
        # Terminal states cannot transition
        assert not TaskStatus.COMPLETED.can_transition_to(TaskStatus.RUNNING)
        assert not TaskStatus.CANCELLED.can_transition_to(TaskStatus.RUNNING)
        assert not TaskStatus.FAILED.can_transition_to(TaskStatus.RUNNING)
        # Cannot skip states
        assert not TaskStatus.PENDING.can_transition_to(TaskStatus.COMPLETED)

    def test_validate_transition_raises_on_invalid(self):
        with pytest.raises(InvalidTransitionError) as exc_info:
            validate_transition("task-1", TaskStatus.COMPLETED, TaskStatus.RUNNING)
        assert "task-1" in str(exc_info.value)

    def test_validate_transition_ok_on_valid(self):
        validate_transition("task-1", TaskStatus.PENDING, TaskStatus.RUNNING)  # Should not raise


class TestTaskRegistry:
    def setup_method(self):
        self.registry = TaskRegistry(session_id="test-session")

    def test_create_task(self):
        record = self.registry.create_task(
            task_type="slow_reasoning",
            generation=1,
            created_ms=0.0,
        )
        assert record.task_id
        assert record.status == TaskStatus.PENDING
        assert record.generation == 1
        assert record.session_id == "test-session"

    def test_create_task_with_explicit_id(self):
        record = self.registry.create_task(
            task_type="test",
            generation=1,
            task_id="explicit-id",
        )
        assert record.task_id == "explicit-id"

    def test_create_task_registers_call_id(self):
        record = self.registry.create_task(
            task_type="tool_call",
            generation=1,
            call_id="call-001",
        )
        found = self.registry.get_by_call_id("call-001")
        assert found is not None
        assert found.task_id == record.task_id

    def test_transition_pending_to_running(self):
        record = self.registry.create_task("test", 1)
        updated = self.registry.transition(record.task_id, TaskStatus.RUNNING, timestamp_ms=10.0)
        assert updated.status == TaskStatus.RUNNING
        assert updated.started_ms == 10.0

    def test_transition_running_to_completed(self):
        record = self.registry.create_task("test", 1)
        self.registry.transition(record.task_id, TaskStatus.RUNNING, timestamp_ms=5.0)
        self.registry.transition(record.task_id, TaskStatus.COMPLETED, timestamp_ms=100.0)
        record = self.registry.get(record.task_id)
        assert record.status == TaskStatus.COMPLETED
        assert record.completed_ms == 100.0

    def test_invalid_transition_raises(self):
        record = self.registry.create_task("test", 1)
        self.registry.transition(record.task_id, TaskStatus.RUNNING)
        self.registry.transition(record.task_id, TaskStatus.COMPLETED)
        with pytest.raises(InvalidTransitionError):
            self.registry.transition(record.task_id, TaskStatus.RUNNING)

    def test_get_returns_none_for_unknown(self):
        result = self.registry.get("nonexistent-task-id")
        assert result is None

    def test_get_or_raise_raises_for_unknown(self):
        with pytest.raises(KeyError):
            self.registry._get_or_raise("bad-id")

    def test_active_tasks_excludes_terminal(self):
        r1 = self.registry.create_task("t1", 1)
        r2 = self.registry.create_task("t2", 1)
        self.registry.transition(r1.task_id, TaskStatus.RUNNING)
        self.registry.transition(r1.task_id, TaskStatus.COMPLETED)
        # r2 is still PENDING (active)
        active = self.registry.active_tasks()
        assert len(active) == 1
        assert active[0].task_id == r2.task_id

    def test_tasks_by_generation(self):
        r1 = self.registry.create_task("t1", generation=1)
        r2 = self.registry.create_task("t2", generation=2)
        r3 = self.registry.create_task("t3", generation=1)
        gen1 = self.registry.tasks_by_generation(1)
        assert len(gen1) == 2
        gen2 = self.registry.tasks_by_generation(2)
        assert len(gen2) == 1

    def test_len(self):
        assert len(self.registry) == 0
        self.registry.create_task("t1", 1)
        self.registry.create_task("t2", 1)
        assert len(self.registry) == 2
