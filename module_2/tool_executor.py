"""
tool_executor.py
Closes the loop create_background_task() left open: without this,
execute_tool() was a no-op stub and no TOOL_RESULT ever got submitted,
so on_tool_result() never fired and no FINAL_RESPONSE was ever emitted
for a real (non-test-injected) run.

Runs the real/mock tool call under a timeout and always submits a
TOOL_RESULT event back — success or failure — so the adapter's
on_tool_result handler runs.

v2 — fixes a real correctness bug, not a style pass: retries used to
apply unconditionally to every tool, including state-modifying ones.
That's a genuine double-booking risk, and idempotency.py's own
docstring claim ("Module 3 dedupes by idempotency_key") did NOT
actually cover it, because a retry here calls the mock tool function
directly (`fn(arguments)`), bypassing Module 3's TOOL_CALL/idempotency
gate entirely — that gate only ever sees ONE outgoing TOOL_CALL per
orchestrator decision, so a retry loop hidden inside the executor is
invisible to it. Concretely: mock_tools.book_flight() mints a fresh
random booking_id on every call with no dedup of its own, so a
transient failure followed by an internal retry-success can silently
create two distinct bookings for what the user experiences as one
request.

Fix: only auto-retry READ_ONLY tools (a re-run search or status check
is safe by definition — it changes no state and returns fresh, still-
correct data). A failed STATE_MODIFYING call is surfaced to the
orchestrator as a single failure, which already has a correct path for
this (on_tool_result's failure branch emits a clarification and asks
the user whether to retry, keeping slots intact) — that path is where
a *deliberate*, user-visible retry belongs, not a silent one buried in
the executor.
"""

import asyncio
from typing import Callable, Awaitable, Dict, Any, Optional

from tool_manifest import ToolManifest

try:
    from module3.runtime.events import make_tool_result
except ImportError:
    make_tool_result = None  # only used in real-runtime mode


class ToolExecutionError(Exception):
    pass


async def execute_and_report(
    runtime,
    session_id: str,
    call_id: str,
    task_id_box: Dict[str, Optional[str]],
    tool_name: str,
    arguments: Dict[str, Any],
    registry: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]],
    manifest: ToolManifest,
    timeout_s: float = 5.0,
    max_retries: int = 1,
):
    """
    task_id_box: a mutable {"id": None} the caller fills in with the real
    task_id right after create_background_task() returns. Safe because
    this coroutine can't start running before that happens (it isn't
    scheduled until the caller yields control back to the event loop).

    manifest: REQUIRED (not optional) so the retry decision below can
    check tool kind. Passing the wrong/empty manifest degrades safely —
    an unclassified tool defaults to STATE_MODIFYING per
    ToolManifest._classify's own "safest default" — so at worst you get
    zero retries on a tool that could have safely had one, never the
    other way around.
    """
    fn = registry.get(tool_name)

    # Read-only tools are safe to retry silently: re-running a search or
    # status check changes no state, so at most it costs latency. A
    # state-modifying tool gets exactly one attempt here — its failure
    # path is a deliberate, user-visible retry-or-not decision made by
    # the orchestrator, not something this executor should shortcut.
    try:
        is_read_only = not manifest.is_state_modifying(tool_name)
    except KeyError:
        # Tool not in the manifest yet (e.g. a genuinely unseen tool from
        # pub_09-style scenarios) — fail safe, same as the classifier's
        # own default: treat as NOT safe to blindly retry.
        is_read_only = False

    attempts_allowed = 1 + max_retries if is_read_only else 1

    attempt = 0
    last_error = None

    while attempt < attempts_allowed:
        attempt += 1
        try:
            if fn is None:
                raise ToolExecutionError(f"No implementation registered for tool '{tool_name}'")
            result = await asyncio.wait_for(fn(arguments), timeout=timeout_s)
            await runtime.submit_event(make_tool_result(
                session_id=session_id, timestamp_ms=runtime.clock.now(),
                call_id=call_id, task_id=task_id_box.get("id"),
                result=result, success=True,
            ))
            return
        except asyncio.TimeoutError:
            last_error = f"timeout after {timeout_s}s"
        except Exception as exc:
            last_error = str(exc)

    await runtime.submit_event(make_tool_result(
        session_id=session_id, timestamp_ms=runtime.clock.now(),
        call_id=call_id, task_id=task_id_box.get("id"),
        result=None, success=False, error=last_error,
    ))
