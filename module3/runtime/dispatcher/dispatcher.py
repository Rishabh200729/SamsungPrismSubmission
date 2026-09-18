"""
module3/runtime/dispatcher/dispatcher.py

Central event dispatcher / router.

The dispatcher receives events from the input queue and routes them to
registered handler coroutines based on event type. It contains ZERO
business logic — it is purely a routing mechanism.

Handler registration follows a dependency-injection pattern:

    dispatcher.register_handler(InputEventType.TEXT_CHUNK, my_fast_path_handler)
    dispatcher.register_handler(InputEventType.INTERRUPTION, my_interrupt_handler)

Multiple handlers per event type are supported (fan-out). Handlers are
called concurrently via asyncio.gather with error isolation per handler.

Inspired by:
- Sema Code: programmable shared core with registered interrupt control
- Bohus & Horvitz: separating event ingestion from the policy decisions
  that result from those events (handlers own the policy)
- DuplexOmni: interaction layer must not block while thinking layer runs

IMPORTANT: Handlers must be coroutines (async def). They must NOT block
the event loop. If they need to do heavy work, they must launch background
tasks via the TaskRegistry.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

from ..events.base import AnyEventType, BaseEvent

logger = logging.getLogger(__name__)

# A handler is an async callable that receives a single BaseEvent
Handler = Callable[[BaseEvent], Awaitable[None]]


class Dispatcher:
    """
    Model-agnostic event router.

    Modules register handlers for specific event types.
    The dispatcher routes each event to all matching handlers concurrently.

    Thread-safety: designed for use within a single asyncio event loop.
    """

    def __init__(self) -> None:
        # event_type -> list of registered handlers
        self._handlers: dict[AnyEventType, list[Handler]] = defaultdict(list)
        # Optional wildcard handlers receive every event regardless of type
        self._wildcard_handlers: list[Handler] = []
        self._dispatch_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register_handler(
        self,
        event_type: AnyEventType,
        handler: Handler,
    ) -> None:
        """
        Register a handler coroutine for the given event type.

        Args:
            event_type: The InputEventType, OutputEventType, or InternalEventType
                        to listen for.
            handler:    An async callable (coroutine function) that accepts a
                        single BaseEvent argument.

        Multiple handlers for the same event type are all called concurrently.
        """
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(
                f"Handler {handler!r} must be an async def (coroutine function). "
                "Blocking handlers are not allowed in the runtime event loop."
            )
        self._handlers[event_type].append(handler)
        logger.debug("Dispatcher: registered handler %s for %s", handler.__name__, event_type)

    def register_wildcard_handler(self, handler: Handler) -> None:
        """
        Register a handler that receives ALL events regardless of type.
        Useful for the TraceRecorder or audit loggers.
        """
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(f"Wildcard handler {handler!r} must be async def")
        self._wildcard_handlers.append(handler)
        logger.debug("Dispatcher: registered wildcard handler %s", handler.__name__)

    def unregister_handler(self, event_type: AnyEventType, handler: Handler) -> bool:
        """Remove a specific handler. Returns True if found and removed."""
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def handler_count(self, event_type: AnyEventType) -> int:
        return len(self._handlers.get(event_type, []))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, event: BaseEvent) -> None:
        """
        Route one event to all registered handlers concurrently.

        Errors in individual handlers are caught, logged, and do NOT
        propagate — this ensures one faulty handler cannot crash the
        entire runtime event loop.

        Args:
            event: The event to dispatch.
        """
        self._dispatch_count += 1

        specific_handlers = self._handlers.get(event.event_type, [])
        all_handlers = specific_handlers + self._wildcard_handlers

        if not all_handlers:
            logger.debug(
                "Dispatcher: no handlers for %s (session=%s)",
                event.event_type.value,
                event.session_id,
            )
            return

        logger.debug(
            "Dispatcher: dispatching %s to %d handler(s) (session=%s ts=%.1f)",
            event.event_type.value,
            len(all_handlers),
            event.session_id,
            event.timestamp_ms,
        )

        # Run all handlers concurrently, isolating individual failures
        results = await asyncio.gather(
            *[self._call_handler(h, event) for h in all_handlers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                self._error_count += 1
                logger.error(
                    "Dispatcher: handler raised exception for %s: %r",
                    event.event_type.value,
                    result,
                )

    async def _call_handler(self, handler: Handler, event: BaseEvent) -> None:
        """Wrapper that isolates one handler call."""
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise  # Let cancellation propagate
        except Exception:
            # BUG-6 FIX: bare `raise` preserves the original traceback.
            # `raise exc` would create a new exception object and lose the
            # original call stack, making debugging much harder.
            raise

    # ------------------------------------------------------------------
    # Continuous dispatch loop (used by the Runtime's main event loop)
    # ------------------------------------------------------------------

    async def run_dispatch_loop(
        self,
        get_event: Callable[[], Awaitable[BaseEvent]],
        task_done: Callable[[], None] | None = None,
    ) -> None:
        """
        Continuously fetch events from get_event() and dispatch them.

        This is the main event loop body. It runs until it receives a
        None sentinel (meaning the input queue has been closed) or is
        cancelled externally.

        Args:
            get_event:  Async callable that blocks until an event is available
                        (typically InputQueue.get).
            task_done:  Optional callable to acknowledge event consumption
                        (typically InputQueue.task_done).
        """
        logger.info("Dispatcher: event loop started")
        while True:
            try:
                event = await get_event()
            except asyncio.CancelledError:
                logger.info("Dispatcher: event loop cancelled")
                break

            # None sentinel = queue closed
            if event is None:  # type: ignore[comparison-overlap]
                if task_done:
                    task_done()
                logger.info("Dispatcher: received close sentinel, stopping event loop")
                break

            try:
                await self.dispatch(event)
            except asyncio.CancelledError:
                if task_done:
                    task_done()
                break
            except Exception as exc:
                logger.error("Dispatcher: unhandled error during dispatch: %r", exc)
            finally:
                if task_done:
                    task_done()

        logger.info(
            "Dispatcher: event loop stopped (dispatched=%d errors=%d)",
            self._dispatch_count,
            self._error_count,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def registered_types(self) -> list[AnyEventType]:
        return list(self._handlers.keys())
