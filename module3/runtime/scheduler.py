"""
module3/runtime/scheduler.py

Fast / Slow / Multimodal worker scheduler — §7 of the orchestration kernel spec.

Responsibilities
----------------
- Classifies proposed work into three worker types.
- Enforces concurrency rules:
    * READ_ONLY work may run concurrently.
    * STATE_MODIFYING writes are serialized (no parallel write branches).
    * Audio/video analysis runs independently via MultimodalWorker.
- Carries generation + snapshot_version on every proposal so the output gate
  can reject stale work without the worker needing to check itself.
- Does NOT modify session state directly — all state changes go through the
  output gate (emit_output / apply_slot_patch on the Runtime API).

Worker classes
--------------
FastWorker   — fillers, confirmations, slot-patch acknowledgement, clarification.
               Must respond within a tight latency budget; never calls tools.
SlowWorker   — planner, tool orchestration, synthesis.
               May spawn background tasks; must submit proposals, not emit directly.
MultimodalWorker — audio/frame interpretation, evidence extraction.
               Runs independently of the main reasoning path.

ActionProposal
--------------
Every worker produces ActionProposal objects which carry enough context for
the output gate to validate generation + snapshot_version before publishing.

Usage::

    scheduler = WorkerScheduler(runtime)
    proposal = ActionProposal(
        session_id=sid,
        generation=gen,
        snapshot_version=sv,
        source_event_ids=[event.event_id],
        kind="filler",
        payload={"text": "One moment..."},
    )
    await scheduler.submit(proposal)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActionProposal — the proposal contract between workers and the gate
# ---------------------------------------------------------------------------

class ProposalKind(str, Enum):
    FILLER = "filler"
    CLARIFICATION = "clarification"
    SLOT_PATCH = "slot_patch"
    TOOL_CALL = "tool_call"
    FINAL_RESPONSE = "final_response"
    CANCELLATION = "cancellation"
    MULTIMODAL_EVIDENCE = "multimodal_evidence"


@dataclass
class ActionProposal:
    """
    A worker's proposed action, carrying all context needed for gate validation.

    Fields
    ------
    session_id         — the session this proposal belongs to.
    generation         — the generation counter at proposal time.
    snapshot_version   — the snapshot version at proposal time.
    source_event_ids   — event IDs that caused this proposal (causal trace).
    kind               — what kind of action this is.
    payload            — kind-specific data (e.g., filler text, tool arguments).
    is_write           — True for STATE_MODIFYING tool calls.
    idempotency_key    — required when is_write=True.
    """
    session_id: str
    generation: int
    snapshot_version: int
    source_event_ids: list[str]
    kind: ProposalKind
    payload: dict[str, Any] = field(default_factory=dict)
    is_write: bool = False
    idempotency_key: str | None = None


# ---------------------------------------------------------------------------
# Worker types
# ---------------------------------------------------------------------------

class WorkerKind(str, Enum):
    FAST = "fast"
    SLOW = "slow"
    MULTIMODAL = "multimodal"


# A ProposalHandler is an async callable that accepts an ActionProposal and
# returns None. It is typically rt.emit_output wrapped with proposal context.
ProposalHandler = Callable[[ActionProposal], Awaitable[None]]


# ---------------------------------------------------------------------------
# WorkerScheduler
# ---------------------------------------------------------------------------

class WorkerScheduler:
    """
    Coordinates fast, slow, and multimodal worker coroutines.

    Permitted
    ---------
    - Concurrent read-only work.
    - Separate audio/video analysis independent of reasoning path.
    - Background tool calls.
    - Cancellation propagation to all active workers.

    Forbidden
    ---------
    - State-changing execution without output-gate approval.
    - Writes during unresolved slot ambiguity (checked via slot_ambiguous flag).
    - Side-effecting parallel write branches.
    - A worker modifying session state directly.

    Thread-safety: runs entirely within one asyncio event loop.
    """

    def __init__(self, proposal_handler: ProposalHandler) -> None:
        self._handler = proposal_handler
        # Write serialization lock — only one STATE_MODIFYING call at a time.
        self._write_lock = asyncio.Lock()
        # Track active coroutines per session for cancellation.
        self._active: dict[str, list[asyncio.Task[None]]] = {}
        # Per-session ambiguity flag — set True when slot ambiguity is unresolved.
        self._slot_ambiguous: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, proposal: ActionProposal, worker_kind: WorkerKind = WorkerKind.SLOW) -> None:
        """
        Schedule a proposal for execution.

        Fast and multimodal proposals are fired concurrently.
        Slow write proposals are serialized through _write_lock.
        """
        if proposal.is_write and self._slot_ambiguous.get(proposal.session_id):
            logger.warning(
                "Scheduler: write proposal blocked — slot ambiguity unresolved session=%s",
                proposal.session_id,
            )
            return

        if worker_kind == WorkerKind.FAST:
            task = asyncio.create_task(
                self._run_fast(proposal), name=f"fast_{proposal.session_id[:8]}"
            )
        elif worker_kind == WorkerKind.MULTIMODAL:
            task = asyncio.create_task(
                self._run_multimodal(proposal), name=f"mm_{proposal.session_id[:8]}"
            )
        else:
            if proposal.is_write:
                task = asyncio.create_task(
                    self._run_slow_write(proposal), name=f"slow_w_{proposal.session_id[:8]}"
                )
            else:
                task = asyncio.create_task(
                    self._run_slow_read(proposal), name=f"slow_r_{proposal.session_id[:8]}"
                )

        self._track(proposal.session_id, task)

    async def cancel_session(self, session_id: str) -> None:
        """Cancel all active worker tasks for a session."""
        tasks = self._active.pop(session_id, [])
        for t in tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        logger.info("Scheduler: cancelled %d workers for session=%s", len(tasks), session_id)

    def set_slot_ambiguous(self, session_id: str, ambiguous: bool) -> None:
        """Mark whether slot ambiguity is unresolved for a session (blocks writes)."""
        self._slot_ambiguous[session_id] = ambiguous

    # ------------------------------------------------------------------
    # Worker implementations
    # ------------------------------------------------------------------

    async def _run_fast(self, proposal: ActionProposal) -> None:
        """Fast path: filler, clarification, slot-patch ack. No writes allowed."""
        if proposal.is_write:
            logger.error("Scheduler: FastWorker cannot issue writes — dropping proposal")
            return
        try:
            await self._handler(proposal)
        except Exception as exc:
            logger.error("Scheduler[FAST]: handler error: %r", exc)

    async def _run_slow_read(self, proposal: ActionProposal) -> None:
        """Slow path, read-only: concurrent execution permitted."""
        try:
            await self._handler(proposal)
        except Exception as exc:
            logger.error("Scheduler[SLOW_READ]: handler error: %r", exc)

    async def _run_slow_write(self, proposal: ActionProposal) -> None:
        """Slow path, write: serialized through _write_lock."""
        async with self._write_lock:
            if self._slot_ambiguous.get(proposal.session_id):
                logger.warning(
                    "Scheduler[SLOW_WRITE]: slot ambiguity detected inside lock — aborting write session=%s",
                    proposal.session_id,
                )
                return
            try:
                await self._handler(proposal)
            except Exception as exc:
                logger.error("Scheduler[SLOW_WRITE]: handler error: %r", exc)

    async def _run_multimodal(self, proposal: ActionProposal) -> None:
        """Multimodal: audio/frame evidence extraction, independent lane."""
        try:
            await self._handler(proposal)
        except Exception as exc:
            logger.error("Scheduler[MULTIMODAL]: handler error: %r", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _track(self, session_id: str, task: asyncio.Task[None]) -> None:
        self._active.setdefault(session_id, [])
        # Prune completed tasks before appending.
        self._active[session_id] = [t for t in self._active[session_id] if not t.done()]
        self._active[session_id].append(task)
        task.add_done_callback(lambda _: self._prune(session_id))

    def _prune(self, session_id: str) -> None:
        if session_id in self._active:
            self._active[session_id] = [t for t in self._active[session_id] if not t.done()]
