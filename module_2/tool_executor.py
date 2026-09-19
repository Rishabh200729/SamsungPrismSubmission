"""
tool_executor.py
Closes the loop create_background_task() left open: without this,
execute_tool() was a no-op stub and no TOOL_RESULT ever got submitted,
so on_tool_result() never fired and no FINAL_RESPONSE was ever emitted
for a real (non-test-injected) run.

Runs the real/mock tool call under a timeout, retries transient failures
up to max_retries using the SAME idempotency_key (safe: Module 3's gate
already dedupes by that key), and always submits a TOOL_RESULT event back
— success or failure — so the adapter's on_tool_result handler runs.
"""

import asyncio
from typing import Callable, Awaitable, Dict, Any, Optional

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
    timeout_s: float = 5.0,
    max_retries: int = 1,
):
    """
    task_id_box: a mutable {"id": None} the caller fills in with the real
    task_id right after create_background_task() returns. Safe because
    this coroutine can't start running before that happens (it isn't
    scheduled until the caller yields control back to the event loop).
    """
    fn = registry.get(tool_name)
    attempt = 0
    last_error = None

    while attempt <= max_retries:
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
