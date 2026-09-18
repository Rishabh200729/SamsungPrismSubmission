"""
module3/runtime/runtime.py

Main Runtime class — the execution kernel for PRISM.

The Runtime is the central composition point that wires together:
- InputQueue
- OutputQueue
- Dispatcher
- SessionManager
- TraceRecorder
- Clock (virtual or real)

It exposes the public API used by all other modules.

Architecture:
    submit_event()  -> InputQueue -> (dispatch loop) -> Dispatcher -> Handlers
    emit_output()   -> TraceRecorder + OutputQueue
    create_background_task() -> TaskRegistry + asyncio.Task

Inspired by:
- DuplexOmni: interaction/thinking layer separation via async coexistence
- Sema Code: programmable shared core
- Structured Graph Harness: explicit task lifecycle
- Gaia2: deterministic reproducibility via clock injection

The dispatch loop runs as a background asyncio.Task. It is non-blocking
for the main event loop — producers can submit_event() continuously without
waiting for handlers to complete.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable, Coroutine

from .clock.base import Clock
from .clock.real_clock import RealClock
from .clock.virtual_clock import VirtualClock
from .config import RuntimeConfig, DEFAULT_CONFIG
from .dispatcher.dispatcher import Dispatcher, Handler
from .events.base import AnyEventType, BaseEvent, EventCategory, InputEventType, InternalEventType
from .events import internal_events as ie
from .events.base import OutputEventType
from .events.output_events import make_cancellation
from .events.input_events import ToolManifestPayload
from .tools import CallRecord, ToolSideEffect
from .queues.input_queue import InputQueue
from .queues.output_queue import OutputQueue
from .sessions.manager import SessionContext, SessionManager
from .tasks.lifecycle import ExecutionPhase, TaskStatus
from .tasks.registry import TaskRecord
from .tracing.recorder import TraceRecorder

logger = logging.getLogger(__name__)


class _DuplicateCallError(Exception):
    """Sentinel: output gate detected a duplicate call_id (soft-reject, already traced)."""


class Runtime:
    """
    PRISM shared runtime — the execution kernel.

    Usage::

        rt = Runtime()
        await rt.start()

        sid = rt.create_session()
        event = make_text_chunk(session_id=sid, timestamp_ms=0.0, text="Book a flight")
        await rt.submit_event(event)

        output = await rt.next_output()

        await rt.stop()
    """

    def __init__(
        self,
        config: RuntimeConfig = DEFAULT_CONFIG,
        clock: Clock | None = None,
    ) -> None:
        self._config = config
        self._clock: Clock = clock or (
            VirtualClock() if config.deterministic_mode else RealClock()
        )

        self._input_queue = InputQueue(
            maxsize=config.input_queue_maxsize,
            drop_oldest=config.input_queue_drop_oldest,
        )
        self._output_queue = OutputQueue(maxsize=config.output_queue_maxsize)
        self._dispatcher = Dispatcher()
        self._session_manager = SessionManager()
        self._tracer = TraceRecorder(record_wall_clock=config.record_wall_clock)

        # Background dispatch loop handle
        self._dispatch_loop_task: asyncio.Task[None] | None = None
        self._running = False

        # Core protocol handlers are non-optional: integrations may add policy,
        # but cannot bypass interruption or manifest state installation.
        self._dispatcher.register_handler(InputEventType.INTERRUPTION, self._handle_interruption)
        self._dispatcher.register_handler(InputEventType.TOOL_MANIFEST, self._handle_tool_manifest)

        logging.basicConfig(level=getattr(logging, config.log_level, logging.WARNING))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background dispatch loop."""
        if self._running:
            logger.warning("Runtime: already running")
            return
        self._running = True
        self._dispatch_loop_task = asyncio.create_task(
            self._dispatcher.run_dispatch_loop(
                get_event=self._input_queue.get,
                task_done=self._input_queue.task_done,
            ),
            name="prism_dispatch_loop",
        )
        logger.info("Runtime: started (deterministic=%s)", self._config.deterministic_mode)

    async def stop(self) -> None:
        """Stop the runtime gracefully."""
        if not self._running:
            return
        self._running = False
        await self._input_queue.close()
        if self._dispatch_loop_task:
            try:
                await asyncio.wait_for(self._dispatch_loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # BUG-3 FIX: cancel() only schedules cancellation; must await to
                # ensure the task actually terminates before stop() returns.
                self._dispatch_loop_task.cancel()
                try:
                    await self._dispatch_loop_task
                except asyncio.CancelledError:
                    pass  # Expected — task was cancelled cleanly
        logger.info("Runtime: stopped")

    # ------------------------------------------------------------------
    # Session API
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new isolated session. Returns session_id."""
        ts = self._clock.now()
        ctx = self._session_manager.create_session(
            session_id=session_id,
            created_ms=ts,
            metadata=metadata,
        )
        # Wire up stale-result trace callback
        ctx.set_trace_callback(self._stale_result_trace_callback)

        # Record session creation in trace
        self._tracer.record_transition(
            session_id=ctx.session_id,
            timestamp_ms=ts,
            entry_type=InternalEventType.SESSION_CREATED.value,
            source="runtime",
        )
        return ctx.session_id

    async def close_session(self, session_id: str) -> None:
        """Close a session and cancel its tasks."""
        ts = self._clock.now()
        await self._session_manager.close_session(session_id, closed_ms=ts)
        self._tracer.record_transition(
            session_id=session_id,
            timestamp_ms=ts,
            entry_type=InternalEventType.SESSION_CLOSED.value,
            source="runtime",
        )

    def get_session(self, session_id: str) -> SessionContext | None:
        return self._session_manager.get_session(session_id)

    def get_session_metadata(self, session_id: str) -> dict[str, Any]:
        ctx = self._session_manager.get_session(session_id)
        return ctx.metadata if ctx else {}

    def get_current_generation(self, session_id: str) -> int:
        ctx = self._session_manager.get_session(session_id)
        return ctx.current_generation if ctx else 0

    # ------------------------------------------------------------------
    # Event submission
    # ------------------------------------------------------------------

    async def submit_event(self, event: BaseEvent) -> None:
        """
        Submit an input event into the runtime.

        The event is placed on the input queue and will be dispatched
        to registered handlers asynchronously. This call returns as
        soon as the event is enqueued — it does NOT wait for handlers.

        BUG-4 FIX: Events for unknown or closed sessions are dropped here
        rather than entering the queue. This enforces session isolation —
        a closed session cannot receive new events.
        """
        ctx = self._session_manager.get_session(event.session_id)
        if ctx is None:
            logger.warning(
                "Runtime: submit_event unknown session_id=%s, dropping event %s",
                event.session_id, event.event_type.value,
            )
            return
        if ctx.closed:
            logger.warning(
                "Runtime: submit_event session_id=%s is closed, dropping event %s",
                event.session_id, event.event_type.value,
            )
            return
        ctx.record_event_id(event.event_id)
        # Record accepted input before dispatch. Dispatcher handlers run
        # concurrently, so tracing at submission establishes the causal order
        # between an input (especially TOOL_RESULT) and its handler effects.
        self._tracer.record_event(event, source="input_submitter")
        dropped = await self._input_queue.put(event)
        if dropped is not None:
            self._tracer.record_transition(
                session_id=dropped.session_id,
                timestamp_ms=event.timestamp_ms,
                entry_type=InternalEventType.QUEUE_OVERFLOW.value,
                source="runtime.input_queue",
                metadata={
                    "dropped_event_id": dropped.event_id,
                    "dropped_event_type": dropped.event_type.value,
                    "dropped_session_id": dropped.session_id,
                    "queue_maxsize": self._config.input_queue_maxsize,
                },
            )

    # ------------------------------------------------------------------
    # Output emission
    # ------------------------------------------------------------------

    async def emit_output(self, event: BaseEvent, *, priority: int = 0) -> None:
        """
        Emit an output event from a module.

        Records in trace AND places on output queue.
        Duplicate call_ids are traced and silently dropped (soft-reject);
        all other gate failures propagate as ValueError.
        """
        try:
            self._authorize_output(event)
        except _DuplicateCallError:
            # Already traced inside _authorize_output — just drop the event.
            return
        self._tracer.record_event(event, source="output_emitter")
        await self._output_queue.emit(event, priority=priority)

    def _authorize_output(self, event: BaseEvent) -> None:
        """Single output gate; integrations cannot publish stale/invalid work.

        Raises:
            _DuplicateCallError: duplicate call_id — caller should drop silently.
            ValueError:           all other protocol violations.
        """
        ctx = self._session_manager.get_session(event.session_id)
        if ctx is None or ctx.closed:
            raise ValueError("cannot emit output for unknown or closed session")
        current = ctx.current_generation
        if event.generation is not None and event.generation != current:
            raise ValueError("stale generation output rejected")
        if event.event_type == OutputEventType.TOOL_CALL:
            payload = event.payload
            tool = ctx.tool_manifest.get(payload["tool_name"])
            if tool is None and self._config.require_tool_manifest:
                raise ValueError(f"tool not declared by manifest: {payload['tool_name']}")
            if tool is not None:
                ctx.tool_manifest.validate_args(tool, payload.get("arguments", {}))
            generation = payload.get("generation") or current
            if generation != current:
                raise ValueError("stale tool call rejected")
            snapshot_version = payload.get("snapshot_version") or ctx.get_snapshot().snapshot_version
            if snapshot_version != ctx.get_snapshot().snapshot_version:
                raise ValueError("stale snapshot tool call rejected")
            side_effect = ToolSideEffect(payload.get("side_effect") or (tool.side_effect.value if tool else ToolSideEffect.READ_ONLY.value))
            key = payload.get("idempotency_key")
            if side_effect is ToolSideEffect.STATE_MODIFYING and not key:
                key = f"{event.session_id}:{generation}:{payload['tool_name']}:{sorted(payload.get('arguments', {}).items())}"
                payload["idempotency_key"] = key
            call_id = payload["call_id"]
            try:
                ctx.call_ledger.register(
                    CallRecord(call_id, generation, snapshot_version, payload["tool_name"], side_effect, key)
                )
            except ValueError as exc:
                if "duplicate call_id" in str(exc):
                    # Soft-reject: trace the violation, then signal the caller to drop.
                    self._tracer.record_transition(
                        session_id=event.session_id,
                        timestamp_ms=event.timestamp_ms,
                        entry_type=InternalEventType.DUPLICATE_CALL_ID.value,
                        call_id=call_id,
                        generation=generation,
                        source="output_gate",
                        metadata={"tool_name": payload["tool_name"], "rejected_event_id": event.event_id},
                    )
                    logger.warning(
                        "Runtime: duplicate call_id=%s rejected for session=%s",
                        call_id, event.session_id,
                    )
                    raise _DuplicateCallError(call_id) from exc
                raise
            if event.task_id and ctx.task_registry.get(event.task_id):
                ctx.task_registry.register_call_id(event.task_id, call_id)
            ctx.state_store.add_pending_call(call_id)
        elif event.event_type == OutputEventType.FINAL_RESPONSE:
            snapshot = event.payload.get("state_snapshot", {})
            if snapshot.get("generation") not in (None, current):
                raise ValueError("stale final response rejected")

    async def _handle_tool_manifest(self, event: BaseEvent) -> None:
        ctx = self._session_manager.get_session(event.session_id)
        if ctx:
            ctx.tool_manifest.install(ToolManifestPayload.model_validate(event.payload).tools)

    async def _handle_interruption(self, event: BaseEvent) -> None:
        """Priority semantic action: invalidate first, then make cancellation observable."""
        ctx = self._session_manager.get_session(event.session_id)
        if ctx is None:
            return
        ctx.metadata["_interrupt_guard"] = True
        old_generation, new_generation = await self.request_cancellation(event.session_id, reason=event.payload.get("reason", "interruption"))
        ctx.state_store.invalidate(generation=new_generation, reason=event.payload.get("reason", "interruption"))
        for record in ctx.call_ledger.supersede_before(new_generation):
            await self.emit_output(make_cancellation(event.session_id, self._clock.now(), cancelled_task_id=record.call_id, cancelled_call_id=record.call_id, reason="superseded", generation_at_cancel=old_generation, generation=new_generation), priority=100)
        # Clear BOTH guard keys so the next interrupt gets a fresh generation increment.
        loop = asyncio.get_running_loop()
        loop.call_soon(ctx.metadata.pop, "_interrupt_guard", None)
        loop.call_soon(ctx.metadata.pop, "_interrupt_pair", None)

    async def next_output(
        self,
        event_type: Any | None = None,
        session_id: str | None = None,
        timeout_ms: float | None = None,
    ) -> BaseEvent:
        """Consume the next output event (blocks until available)."""
        return await self._output_queue.next_output(event_type, session_id, timeout_ms)

    async def drain_outputs(self) -> list[BaseEvent]:
        """Drain all currently queued outputs (non-blocking)."""
        return await self._output_queue.drain_all()

    async def drain_all_outputs(self) -> list[BaseEvent]:
        """Compatibility alias for draining all currently queued outputs."""
        return await self.drain_outputs()

    def get_session_event_log(self, session_id: str) -> list[str]:
        """Return a copy of a session's bounded input event ledger."""
        ctx = self._session_manager.get_session(session_id)
        return list(ctx.event_log) if ctx else []

    def export_trace_jsonl(self, path: str) -> None:
        """Export the current runtime trace as JSONL."""
        self._tracer.export_jsonl(path)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, event_type: AnyEventType, handler: Handler) -> None:
        """Register an async event handler for the given event type."""
        self._dispatcher.register_handler(event_type, handler)

    def register_wildcard_handler(self, handler: Handler) -> None:
        """Register a handler that receives every event."""
        self._dispatcher.register_wildcard_handler(handler)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    async def create_background_task(
        self,
        session_id: str,
        coro: Coroutine[Any, Any, Any],
        task_type: str,
        call_id: str | None = None,
        parent_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a background coroutine as a tracked task and launch it.

        Returns task_id. The coroutine runs concurrently — this call is
        non-blocking (the coroutine is scheduled, not awaited).

        The runtime wraps the coroutine in a lifecycle wrapper that:
        1. Records TASK_STARTED when execution begins.
        2. Records TASK_COMPLETED on success.
        3. Records TASK_FAILED on exception.
        4. Records TASK_CANCELLED on asyncio.CancelledError.
        """
        ctx = self._session_manager.get_or_raise(session_id)
        ts = self._clock.now()
        generation = ctx.current_generation

        record = ctx.task_registry.create_task(
            task_type=task_type,
            generation=generation,
            created_ms=ts,
            call_id=call_id,
            parent_task_id=parent_task_id,
            metadata=metadata or {},
        )
        task_id = record.task_id

        # Record creation in trace
        self._tracer.record_transition(
            session_id=session_id,
            timestamp_ms=ts,
            entry_type=InternalEventType.TASK_CREATED.value,
            task_id=task_id,
            call_id=call_id,
            generation=generation,
            status_before=None,
            status_after=TaskStatus.PENDING.value,
            source="runtime",
        )

        # Wrap coroutine in lifecycle observer
        wrapped = self._lifecycle_wrapper(session_id, task_id, generation, coro)
        asyncio_task = asyncio.create_task(wrapped, name=f"task_{task_id[:8]}")

        ctx.task_registry.attach_asyncio_task(task_id, asyncio_task)

        logger.info(
            "Runtime: created background task_id=%s type=%s gen=%d session=%s",
            task_id, task_type, generation, session_id,
        )
        return task_id

    async def _lifecycle_wrapper(
        self,
        session_id: str,
        task_id: str,
        generation: int,
        coro: Coroutine[Any, Any, Any],
    ) -> None:
        """Wrap a coroutine with task lifecycle tracking."""
        ts = self._clock.now()
        ctx = self._session_manager.get_session(session_id)
        if ctx is None:
            logger.warning("Runtime: lifecycle_wrapper session %s not found", session_id)
            # BUG-1 FIX: close the coroutine so Python doesn't emit
            # 'RuntimeWarning: coroutine was never awaited'.
            coro.close()
            return

        # BUG-2 FIX: if the task was already cancelled before this wrapper
        # even started (e.g., interruption arrived between create_background_task
        # and the first event loop tick), do NOT execute the coroutine.
        record = ctx.task_registry.get(task_id)
        if record and record.status == TaskStatus.CANCELLATION_REQUESTED:
            logger.info(
                "Runtime: task %s already cancelled before start, skipping execution",
                task_id,
            )
            coro.close()
            return

        # PENDING -> RUNNING
        try:
            ctx.task_registry.transition(task_id, TaskStatus.RUNNING, timestamp_ms=ts)
            self._tracer.record_transition(
                session_id=session_id,
                timestamp_ms=ts,
                entry_type=InternalEventType.TASK_STARTED.value,
                task_id=task_id,
                generation=generation,
                status_before=TaskStatus.PENDING.value,
                status_after=TaskStatus.RUNNING.value,
                source="runtime",
            )
        except Exception as exc:
            logger.error("Runtime: lifecycle start failed task_id=%s: %r", task_id, exc)
            # BUG-1 FIX: close coroutine here too to avoid leak.
            coro.close()
            return

        try:
            await coro

            # RUNNING -> COMPLETED (only if not already cancelled/stale)
            ts = self._clock.now()
            record = ctx.task_registry.get(task_id)
            if record and not record.status.is_terminal():
                ctx.task_registry.transition(task_id, TaskStatus.COMPLETED, timestamp_ms=ts)
                self._tracer.record_transition(
                    session_id=session_id,
                    timestamp_ms=ts,
                    entry_type=InternalEventType.TASK_COMPLETED.value,
                    task_id=task_id,
                    generation=generation,
                    status_before=TaskStatus.RUNNING.value,
                    status_after=TaskStatus.COMPLETED.value,
                    source="runtime",
                )

        except asyncio.CancelledError:
            try:
                coro.close()
            except Exception:
                pass
            ts = self._clock.now()
            record = ctx.task_registry.get(task_id)
            if record and not record.status.is_terminal():
                try:
                    ctx.task_registry.transition(task_id, TaskStatus.CANCELLED, timestamp_ms=ts)
                except Exception:
                    pass
            self._tracer.record_transition(
                session_id=session_id,
                timestamp_ms=ts,
                entry_type=InternalEventType.TASK_CANCELLED.value,
                task_id=task_id,
                generation=generation,
                status_before=TaskStatus.CANCELLATION_REQUESTED.value,
                status_after=TaskStatus.CANCELLED.value,
                source="runtime",
            )
            raise  # Re-raise so asyncio knows the task was cancelled

        except Exception as exc:
            ts = self._clock.now()
            record = ctx.task_registry.get(task_id)
            if record and not record.status.is_terminal():
                try:
                    ctx.task_registry.transition(task_id, TaskStatus.FAILED, timestamp_ms=ts)
                except Exception:
                    pass
            self._tracer.record_transition(
                session_id=session_id,
                timestamp_ms=ts,
                entry_type=InternalEventType.TASK_FAILED.value,
                task_id=task_id,
                generation=generation,
                status_before=TaskStatus.RUNNING.value,
                status_after=TaskStatus.FAILED.value,
                source="runtime",
                metadata={"error": str(exc)},
            )
            logger.error("Runtime: task_id=%s failed: %r", task_id, exc)

    async def cancel_task(
        self,
        session_id: str,
        task_id: str,
        reason: str = "interrupted",
    ) -> bool:
        """Request cancellation of a specific task."""
        ctx = self._session_manager.get_session(session_id)
        if ctx is None:
            return False
        ts = self._clock.now()

        # Record cancellation request in trace
        record = ctx.task_registry.get(task_id)
        if record:
            self._tracer.record_transition(
                session_id=session_id,
                timestamp_ms=ts,
                entry_type=InternalEventType.TASK_CANCEL_REQUESTED.value,
                task_id=task_id,
                generation=record.generation,
                status_before=record.status.value,
                status_after=TaskStatus.CANCELLATION_REQUESTED.value,
                source="runtime",
                metadata={"reason": reason},
            )

        return await ctx.cancellation.cancel_task(task_id, timestamp_ms=ts, reason=reason)

    async def request_cancellation(
        self,
        session_id: str,
        reason: str = "interruption",
    ) -> tuple[int, int]:
        """
        Increment session generation and cancel all active tasks.
        Returns (old_generation, new_generation).

        This is the primary interrupt handler integration point.
        Module 1's interrupt handler should call this.
        """
        ctx = self._session_manager.get_session(session_id)
        if ctx is None:
            raise KeyError(f"Session {session_id!r} not found")
        if ctx.metadata.get("_interrupt_guard") and "_interrupt_pair" in ctx.metadata:
            return ctx.metadata["_interrupt_pair"]

        ts = self._clock.now()

        # cancel_task() has no suspension points. Applying the cancellation
        # requests and immediately invalidating the generation is therefore
        # atomic with respect to other event-loop work. This keeps stale-result
        # protection authoritative while exposing the intended causal trace.
        if self._config.cancel_on_interrupt:
            for record in list(ctx.task_registry.active_tasks()):
                status_before = record.status.value
                if await ctx.cancellation.cancel_task(record.task_id, timestamp_ms=ts, reason=reason):
                    self._tracer.record_transition(
                        session_id=session_id,
                        timestamp_ms=ts,
                        entry_type=InternalEventType.TASK_CANCEL_REQUESTED.value,
                        task_id=record.task_id,
                        generation=record.generation,
                        status_before=status_before,
                        status_after=TaskStatus.CANCELLATION_REQUESTED.value,
                        source="runtime.request_cancellation",
                        metadata={"reason": reason},
                    )

        old_gen, new_gen = ctx.cancellation.increment_generation(reason=reason)
        if ctx.metadata.get("_interrupt_guard"):
            ctx.metadata["_interrupt_pair"] = (old_gen, new_gen)
        self._tracer.record_transition(
            session_id=session_id,
            timestamp_ms=ts,
            entry_type=InternalEventType.GENERATION_INVALIDATED.value,
            generation=new_gen,
            source="runtime",
            metadata={"old_generation": old_gen, "new_generation": new_gen, "reason": reason},
        )

        return old_gen, new_gen

    # ------------------------------------------------------------------
    # Stale-result protection
    # ------------------------------------------------------------------

    def is_stale_result(self, session_id: str, task_id: str) -> bool:
        """True if task's generation < session's current generation,
        OR if the task is completely unknown (never registered)."""
        ctx = self._session_manager.get_session(session_id)
        if ctx is None:
            return True
        record = ctx.task_registry.get(task_id)
        if record is None:
            return True  # Unknown task ID = treat as stale/invalid
        return ctx.cancellation.is_stale(record)

    def is_stale_call(self, session_id: str, call_id: str) -> bool:
        """True if the tool call belongs to a stale task generation."""
        ctx = self._session_manager.get_session(session_id)
        if ctx is None:
            return True
        return ctx.cancellation.is_stale_call(call_id)

    async def handle_stale_result(
        self,
        session_id: str,
        task_id: str,
        call_id: str | None = None,
    ) -> None:
        """
        Explicitly handle a stale result — mark task STALE and record trace.
        Call this instead of silently ignoring late results.
        """
        ctx = self._session_manager.get_session(session_id)
        if ctx is None:
            return
        ts = self._clock.now()
        await ctx.cancellation.handle_stale_result(task_id, call_id, timestamp_ms=ts)

    # ------------------------------------------------------------------
    # Task lookup
    # ------------------------------------------------------------------

    def get_task(self, session_id: str, task_id: str) -> TaskRecord | None:
        ctx = self._session_manager.get_session(session_id)
        return ctx.task_registry.get(task_id) if ctx else None

    def get_task_by_call_id(self, session_id: str, call_id: str) -> TaskRecord | None:
        ctx = self._session_manager.get_session(session_id)
        return ctx.task_registry.get_by_call_id(call_id) if ctx else None

    # ------------------------------------------------------------------
    # Trace access
    # ------------------------------------------------------------------

    @property
    def tracer(self) -> TraceRecorder:
        return self._tracer

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def input_queue(self) -> InputQueue:
        return self._input_queue

    @property
    def output_queue(self) -> OutputQueue:
        return self._output_queue

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    async def _trace_all_events(self, event: BaseEvent) -> None:
        """Wildcard handler: records every dispatched event in the trace."""
        self._tracer.record_event(event, source="dispatcher")

    async def _stale_result_trace_callback(self, **kwargs: Any) -> None:
        """Called by CancellationCoordinator when a stale result is discarded.

        BUG-5 FIX: session_id is now passed directly as a kwarg instead of
        being reverse-looked-up via O(N×M) scan across all sessions and tasks.
        """
        session_id = kwargs.get("session_id", "unknown")
        task_id = kwargs.get("task_id", "unknown")
        call_id = kwargs.get("call_id")
        task_generation = kwargs.get("task_generation", 0)
        current_generation = kwargs.get("current_generation", 0)
        timestamp_ms = kwargs.get("timestamp_ms", 0.0)
        self._tracer.record_transition(
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            entry_type=InternalEventType.STALE_RESULT_DISCARDED.value,
            task_id=task_id,
            call_id=call_id,
            generation=current_generation,
            source="cancellation_coordinator",
            metadata={
                "task_generation": task_generation,
                "current_generation": current_generation,
            },
        )

    def _find_session_for_task(self, task_id: str) -> str:
        """Best-effort reverse lookup of session for a task_id.

        NOTE: This is kept as a fallback but should not be called on the
        hot path. session_id is now passed directly into _stale_result_trace_callback
        (Bug-5 fix). Use this only for debugging/introspection.
        """
        for sid in self._session_manager.all_session_ids():
            ctx = self._session_manager.get_session(sid)
            if ctx and ctx.task_registry.get(task_id):
                return sid
        return "unknown"
