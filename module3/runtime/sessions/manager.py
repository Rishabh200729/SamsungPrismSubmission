"""
module3/runtime/sessions/manager.py

Session lifecycle manager.

Each session represents one isolated conversation/interaction context.
Sessions are the primary isolation boundary — no cross-session state
access is possible through this API.

Session state includes:
- Isolated TaskRegistry
- Isolated CancellationCoordinator
- Event history references (event_ids only, not full events)
- Session metadata
- Current generation counter (owned by the CancellationCoordinator)

Inspired by:
- Sema Code: session isolation, no cross-session memory leakage
- Challenge spec §9: "NO CROSS-SESSION MEMORY LEAKAGE"

The manager itself holds a dict of sessions. Each SessionContext is
a self-contained unit. The runtime routes events to the correct
SessionContext using the session_id on the event.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..tasks.cancellation import CancellationCoordinator
from ..tasks.registry import TaskRegistry
from .state import SessionSnapshot, SessionStateStore
from ..tools import CallLedger, ToolManifestRegistry

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """
    Isolated runtime context for one session.

    All mutable state within a session lives here.
    The runtime never shares a SessionContext across sessions.
    """
    session_id: str
    created_ms: float

    # Isolated subsystems
    task_registry: TaskRegistry = field(init=False)
    cancellation: CancellationCoordinator = field(init=False)
    state_store: SessionStateStore = field(init=False)
    tool_manifest: ToolManifestRegistry = field(init=False)
    call_ledger: CallLedger = field(init=False)

    # Lightweight event ledger — stores the most recent event_ids seen by this
    # session. BUG-9 FIX: bounded deque(maxlen=1000) prevents unbounded memory
    # growth in long sessions (e.g. 100 video frames/sec = 360,000 events/hr).
    event_log: deque[str] = field(default_factory=lambda: deque(maxlen=1000))

    # Arbitrary module-specific metadata stored by handlers
    metadata: dict[str, Any] = field(default_factory=dict)

    # Closed flag — once closed, new events should not be accepted
    closed: bool = field(default=False)
    closed_ms: float | None = field(default=None)

    def __post_init__(self) -> None:
        # Build isolated subsystems referencing this session_id
        self.task_registry = TaskRegistry(session_id=self.session_id)
        self.cancellation = CancellationCoordinator(
            session_id=self.session_id,
            registry=self.task_registry,
        )
        self.state_store = SessionStateStore(self.session_id)
        self.tool_manifest = ToolManifestRegistry()
        self.call_ledger = CallLedger()

    def set_trace_callback(self, callback: Any) -> None:
        """Inject trace callback after construction (avoids circular imports)."""
        self.cancellation._trace_callback = callback

    @property
    def current_generation(self) -> int:
        return self.cancellation.current_generation

    def record_event_id(self, event_id: str) -> None:
        """Log that an event was processed in this session (capped at 1000)."""
        self.event_log.append(event_id)

    def get_snapshot(self) -> SessionSnapshot:
        return self.state_store.snapshot

    def apply_slot_patch(self, expected_version: int, patch: dict[str, Any], *, source_event_id: str | None, timestamp_ms: float, intent: str | None = None) -> SessionSnapshot:
        return self.state_store.patch_slots(expected_version, patch, source_event_id=source_event_id, timestamp_ms=timestamp_ms, intent=intent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_ms": self.created_ms,
            "closed": self.closed,
            "closed_ms": self.closed_ms,
            "current_generation": self.current_generation,
            "snapshot": self.state_store.snapshot.public_dict(),
            "total_events": len(self.event_log),
            "total_tasks": len(self.task_registry),
            "active_tasks": len(self.task_registry.active_tasks()),
            "metadata": self.metadata,
        }


class SessionManager:
    """
    Lifecycle manager for all active sessions.

    Provides create / get / close. Enforces isolation — callers
    can only access their own session by providing its session_id.

    Usage::

        manager = SessionManager()
        ctx = manager.create_session(created_ms=0.0)
        ctx = manager.get_session(ctx.session_id)
        await manager.close_session(ctx.session_id, closed_ms=1000.0)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        created_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> SessionContext:
        """
        Create and register a new isolated session context.

        Args:
            session_id: Optional explicit ID. Auto-generated if None.
            created_ms: Virtual/wall-clock timestamp for creation.
            metadata:   Arbitrary initial metadata.

        Returns:
            SessionContext — the caller should retain this reference.
        """
        sid = session_id or str(uuid.uuid4())
        if sid in self._sessions:
            raise ValueError(f"Session {sid!r} already exists")

        ctx = SessionContext(session_id=sid, created_ms=created_ms)
        if metadata:
            ctx.metadata.update(metadata)

        self._sessions[sid] = ctx
        logger.info("SessionManager: created session=%s ts=%.1f", sid, created_ms)
        return ctx

    def get_session(self, session_id: str) -> SessionContext | None:
        """
        Retrieve a session context by ID.

        Returns None (not raises) if not found — callers should handle
        missing sessions gracefully (e.g., events for unknown sessions).
        """
        return self._sessions.get(session_id)

    def get_or_raise(self, session_id: str) -> SessionContext:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            raise KeyError(f"Session {session_id!r} not found")
        return ctx

    async def close_session(
        self,
        session_id: str,
        closed_ms: float = 0.0,
        cancel_active_tasks: bool = True,
    ) -> None:
        """
        Close a session and optionally cancel all active tasks.

        After closing:
        - No new events should be routed to this session.
        - Active tasks are cancelled.
        - The session context remains accessible for post-session inspection.
        """
        ctx = self._sessions.get(session_id)
        if ctx is None:
            logger.warning("SessionManager: close_session unknown session_id=%s", session_id)
            return

        if ctx.closed:
            logger.debug("SessionManager: session=%s already closed", session_id)
            return

        if cancel_active_tasks:
            cancelled = await ctx.cancellation.cancel_all_active(
                timestamp_ms=closed_ms,
                reason="session_closed",
            )
            if cancelled:
                logger.info(
                    "SessionManager: cancelled %d active tasks on close session=%s",
                    len(cancelled), session_id,
                )

        ctx.closed = True
        ctx.closed_ms = closed_ms
        logger.info("SessionManager: closed session=%s ts=%.1f", session_id, closed_ms)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def all_session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def active_session_ids(self) -> list[str]:
        return [sid for sid, ctx in self._sessions.items() if not ctx.closed]

    def session_count(self) -> int:
        return len(self._sessions)

    def snapshot(self) -> list[dict[str, Any]]:
        """Export a summary snapshot of all sessions for debugging."""
        return [ctx.to_dict() for ctx in self._sessions.values()]
