"""
module3/runtime/interfaces.py

Public integration interfaces for Modules 1, 2, and 4.

These are the ONLY things that other modules need to import from Module 3.
They provide the full surface area without exposing internals.

DESIGN RULE: Other modules MUST NOT import from runtime internals directly.
They should only import from:
    - module3.runtime.interfaces (this file)
    - module3.runtime.events (event factories and types)
    - module3.runtime.config (configuration)

The Runtime class itself is importable but all access goes through
the defined public API.

Module integration summary:

    Module 1 (Fast Path / Floor Manager):
        - submit_event()        — inject input events
        - emit_output()         — emit FILLER, CLARIFICATION
        - create_session()      — start a session
        - register_handler()    — register fast-path handler
        - request_cancellation() — trigger interrupt

    Module 2 (Slow Path / Reasoning + Tool Orchestration):
        - create_background_task() — register a slow reasoning task
        - emit_output()            — emit TOOL_CALL, FINAL_RESPONSE
        - get_session()            — inspect session state
        - get_task()               — check task status
        - is_stale_result()        — check before publishing result
        - handle_stale_result()    — discard and trace a late tool result

    Module 4 (Multimodal):
        - submit_event()           — inject AUDIO_WAV / VIDEO_FRAME
        - register_handler()       — register multimodal handler
        - emit_output()            — emit CLARIFICATION from vision
        - create_background_task() — register multimodal processing task
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .clock import Clock
from .events.base import AnyEventType, BaseEvent
from .tasks.lifecycle import TaskStatus
from .tasks.registry import TaskRecord


@runtime_checkable
class IRuntimeClient(Protocol):
    """
    The interface any module uses to interact with the PRISM runtime.

    Implementing this protocol is NOT required — modules can use the
    Runtime class directly. This protocol documents the expected surface.
    """

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @property
    def clock(self) -> Clock:
        """The runtime clock used for causally ordered event timestamps."""
        ...

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new session. Returns session_id."""
        ...

    async def close_session(self, session_id: str) -> None:
        """Close a session and cancel its active tasks."""
        ...

    # ------------------------------------------------------------------
    # Event flow
    # ------------------------------------------------------------------

    async def submit_event(self, event: BaseEvent) -> None:
        """
        Submit an input event to the runtime.
        The event will be routed through the dispatcher to registered handlers.
        """
        ...

    async def emit_output(self, event: BaseEvent, *, priority: int = 0) -> None:
        """
        Emit an output event from a module into the output queue.
        Use this to produce FILLER, TOOL_CALL, FINAL_RESPONSE, etc.
        """
        ...

    async def next_output(
        self,
        event_type: Any | None = None,
        session_id: str | None = None,
        timeout_ms: float | None = None,
    ) -> BaseEvent:
        """Consume the next output event (used by consumers/evaluators)."""
        ...

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, event_type: AnyEventType, handler: Any) -> None:
        """
        Register an async handler for a specific event type.
        The handler receives BaseEvent and must be a coroutine function.
        """
        ...

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    async def create_background_task(
        self,
        session_id: str,
        coro: Any,
        task_type: str,
        call_id: str | None = None,
        parent_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register and launch a background coroutine as a tracked task.
        Returns task_id. The coroutine runs concurrently — non-blocking.
        """
        ...

    async def cancel_task(self, session_id: str, task_id: str, reason: str = "interrupted") -> bool:
        """Request cancellation of a specific task. Returns True if applied."""
        ...

    async def request_cancellation(self, session_id: str, reason: str = "interruption") -> tuple[int, int]:
        """
        Cancel all active tasks and increment the session generation.
        Returns (old_generation, new_generation).
        Use this when processing an INTERRUPTION event.

        For one interruption, the observable trace is
        TASK_CANCEL_REQUESTED followed by GENERATION_INVALIDATED. Generation
        invalidation remains atomic with respect to accepting stale results.
        """
        ...

    def get_task(self, session_id: str, task_id: str) -> TaskRecord | None:
        """Retrieve task record for status inspection."""
        ...

    def is_stale_result(self, session_id: str, task_id: str) -> bool:
        """
        True if the task's generation is older than the session's
        current generation. Call this before publishing a task result.
        """
        ...

    def is_stale_call(self, session_id: str, call_id: str) -> bool:
        """True if the tool call belongs to a stale generation."""
        ...

    async def handle_stale_result(
        self,
        session_id: str,
        task_id: str,
        call_id: str | None = None,
    ) -> None:
        """Discard a stale tool result exactly once and record its trace entry.

        Repeated calls for the same task are safe no-ops. The first call keeps
        the supplied session, task, and call identifiers in the trace record.
        """
        ...

    # ------------------------------------------------------------------
    # Session inspection
    # ------------------------------------------------------------------

    def get_session_metadata(self, session_id: str) -> dict[str, Any]:
        """Return session metadata dict."""
        ...

    def get_current_generation(self, session_id: str) -> int:
        """Return the current generation counter for the session."""
        ...
