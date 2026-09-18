"""
module3/tests/test_sessions.py

Tests for session isolation and lifecycle.
"""
import asyncio
import pytest

from module3.runtime.sessions.manager import SessionManager


class TestSessionManager:
    def setup_method(self):
        self.manager = SessionManager()

    def test_create_session_returns_id(self):
        sid = self.manager.create_session(created_ms=0.0).session_id
        assert sid

    def test_create_session_with_explicit_id(self):
        ctx = self.manager.create_session(session_id="custom-id")
        assert ctx.session_id == "custom-id"

    def test_create_duplicate_session_raises(self):
        self.manager.create_session(session_id="dup-id")
        with pytest.raises(ValueError, match="already exists"):
            self.manager.create_session(session_id="dup-id")

    def test_get_session_returns_context(self):
        ctx = self.manager.create_session()
        result = self.manager.get_session(ctx.session_id)
        assert result is ctx

    def test_get_nonexistent_session_returns_none(self):
        result = self.manager.get_session("does-not-exist")
        assert result is None

    def test_get_or_raise_raises_for_unknown(self):
        with pytest.raises(KeyError):
            self.manager.get_or_raise("bad-id")

    def test_session_isolation(self):
        """State in session A must not affect session B."""
        ctx_a = self.manager.create_session(session_id="session-A")
        ctx_b = self.manager.create_session(session_id="session-B")

        # Modify session A state
        ctx_a.metadata["user"] = "alice"
        ctx_a.task_registry.create_task("test", generation=1)

        # Session B should be unaffected
        assert "user" not in ctx_b.metadata
        assert len(ctx_b.task_registry.all_tasks()) == 0

    def test_session_has_isolated_task_registry(self):
        ctx_a = self.manager.create_session()
        ctx_b = self.manager.create_session()

        r_a = ctx_a.task_registry.create_task("task_a", generation=1)
        assert ctx_b.task_registry.get(r_a.task_id) is None

    def test_session_has_isolated_generation(self):
        ctx_a = self.manager.create_session()
        ctx_b = self.manager.create_session()

        ctx_a.cancellation.increment_generation()
        ctx_a.cancellation.increment_generation()

        # Session B should still be at generation 1
        assert ctx_b.current_generation == 1

    async def test_close_session(self):
        ctx = self.manager.create_session()
        assert not ctx.closed
        await self.manager.close_session(ctx.session_id, closed_ms=100.0)
        assert ctx.closed
        assert ctx.closed_ms == 100.0

    async def test_close_session_cancels_active_tasks(self):
        ctx = self.manager.create_session()
        from module3.runtime.tasks.lifecycle import TaskStatus
        r = ctx.task_registry.create_task("test", generation=1)
        ctx.task_registry.transition(r.task_id, TaskStatus.RUNNING)

        await self.manager.close_session(ctx.session_id, closed_ms=500.0)
        # Task should be in CANCELLATION_REQUESTED state
        record = ctx.task_registry.get(r.task_id)
        assert record.status == TaskStatus.CANCELLATION_REQUESTED

    def test_all_session_ids(self):
        self.manager.create_session(session_id="s1")
        self.manager.create_session(session_id="s2")
        self.manager.create_session(session_id="s3")
        ids = self.manager.all_session_ids()
        assert set(ids) == {"s1", "s2", "s3"}

    def test_active_session_ids_excludes_closed(self):
        self.manager.create_session(session_id="open")
        ctx_closed = self.manager.create_session(session_id="closed")
        ctx_closed.closed = True  # Mark as closed directly
        active = self.manager.active_session_ids()
        assert "open" in active
        assert "closed" not in active

    def test_snapshot(self):
        self.manager.create_session(session_id="snap-s")
        snaps = self.manager.snapshot()
        assert len(snaps) == 1
        assert snaps[0]["session_id"] == "snap-s"
