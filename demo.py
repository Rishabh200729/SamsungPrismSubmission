"""
demo.py — Interactive terminal demo for PRISM Theme 05: Interruptible Real-Time Agents.

Run: python demo.py

Shows:
  - Live dual-process execution: fast-path filler < 200ms, slow-path tool calls
  - Real-time interruption handling with generation counter
  - Live state snapshot after every turn
  - Colorized event log (filler=cyan, tool=yellow, response=green, cancel=red)
  - Latency timing on every response

Commands during demo:
  Type normally to send a speech turn.
  Type '!' before a message to send as INTERRUPTION (e.g. '!actually change to Osaka')
  Type '/status'   to print the current session state
  Type '/scenario' to run the built-in automated interruption scenario
  Type '/reset'    to start a fresh session
  Type '/quit'     to exit
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "module_2"))

# ── ANSI colours ─────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_MAGENTA= "\033[95m"
_WHITE  = "\033[97m"
_BLUE   = "\033[94m"


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{_RESET}"


def _header(text: str) -> str:
    width = 70
    return (
        f"\n{_c(_BOLD, '═' * width)}\n"
        f"  {_c(_BOLD + _WHITE, text)}\n"
        f"{_c(_BOLD, '═' * width)}"
    )


def _divider() -> str:
    return _c(_DIM, "─" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Module imports (after path setup)
# ──────────────────────────────────────────────────────────────────────────────
from module3 import Runtime
from module3.runtime.config import TEST_CONFIG
from module3.runtime.events import (
    InputEventType, OutputEventType,
    make_text_chunk, make_end_of_turn, make_interruption,
    make_tool_manifest, make_filler,
)
from module1 import FastPathRouter
from orchestrator import Module2ReasoningAdapter
from module4.adapter import Module4Adapter


# ──────────────────────────────────────────────────────────────────────────────
# Demo tool manifest (realistic tools the judge can trigger)
# ──────────────────────────────────────────────────────────────────────────────
DEMO_MANIFEST = [
    {
        "name": "book_flight",
        "side_effect": "STATE_MODIFYING",
        "parameters": {
            "destination": {"required": True, "description": "destination city"},
            "date":        {"required": True, "description": "travel date"},
        },
        "description": "Book a flight to a destination on a given date",
    },
    {
        "name": "get_flight_status",
        "side_effect": "READ_ONLY",
        "parameters": {
            "flight_id": {"required": True, "description": "flight ID like FL-301"},
        },
        "description": "Check the status of an existing flight",
    },
    {
        "name": "book_hotel",
        "side_effect": "STATE_MODIFYING",
        "parameters": {
            "city":        {"required": True, "description": "hotel city"},
        },
        "description": "Book a hotel in a city",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Demo session
# ──────────────────────────────────────────────────────────────────────────────

class DemoSession:
    """Wraps the PRISM runtime + modules for a single interactive session."""

    def __init__(self):
        self._runtime: Optional[Runtime] = None
        self._sid: Optional[str] = None
        self._module4: Optional[Module4Adapter] = None
        self._output_task: Optional[asyncio.Task] = None
        self._output_log: list[dict] = []

    async def start(self):
        self._runtime = Runtime(config=TEST_CONFIG)
        await self._runtime.start()
        self._sid = self._runtime.create_session()

        # Wire modules
        FastPathRouter(self._runtime, self._runtime.clock)
        reasoning = Module2ReasoningAdapter(self._runtime)
        reasoning.install()
        self._module4 = Module4Adapter(self._runtime)
        self._module4.install()

        # Inject the demo manifest
        ts = self._runtime.clock.now()
        await self._runtime.submit_event(
            make_tool_manifest(self._sid, ts, tools=DEMO_MANIFEST)
        )

        # Start background output collector
        self._output_task = asyncio.create_task(self._collect_outputs())

        print(_header("PRISM Agent  —  Interruptible Real-Time Assistant"))
        print(f"\n  {_c(_DIM, 'Session:')} {_c(_CYAN, self._sid[:12])}...")
        print(f"  {_c(_DIM, 'Tools:')}   {', '.join(t['name'] for t in DEMO_MANIFEST)}")
        print(f"\n  {_c(_DIM, 'Tip: prefix with ! to interrupt (e.g. !change to Osaka)')}")
        print(f"  {_c(_DIM, 'Commands: /status  /scenario  /reset  /quit')}")
        print(_divider())

    async def stop(self):
        if self._output_task:
            self._output_task.cancel()
            try:
                await self._output_task
            except asyncio.CancelledError:
                pass
        if self._module4 and self._sid:
            self._module4.drop_session(self._sid)
        if self._runtime and self._sid:
            await self._runtime.close_session(self._sid)
        if self._runtime:
            await self._runtime.stop()

    # ── sending events ────────────────────────────────────────────────────────

    async def send_speech(self, text: str):
        """Send a normal user speech turn."""
        ts = self._runtime.clock.now()
        await self._runtime.submit_event(
            make_text_chunk(self._sid, ts, text=text, is_partial=False)
        )
        await self._runtime.submit_event(
            make_end_of_turn(self._sid, ts + 1.0, final_text=text)
        )

    async def send_interruption(self, text: str):
        """Send a competitive interruption."""
        ts = self._runtime.clock.now()
        gen_before = self._runtime.get_current_generation(self._sid)
        await self._runtime.submit_event(
            make_interruption(self._sid, ts,
                               text=text, reason="user_interruption")
        )
        await asyncio.sleep(0)
        gen_after = self._runtime.get_current_generation(self._sid)
        print(f"\n  {_c(_RED, '🔴 [INTERRUPT]')} "
              f"Generation {_c(_BOLD, str(gen_before))} → "
              f"{_c(_BOLD, str(gen_after))} "
              f"{_c(_DIM, '(in-flight tasks cancelled)')}")
        # Now send text for the new request
        await self._runtime.submit_event(
            make_text_chunk(self._sid, ts + 5.0, text=text, is_partial=False)
        )
        await self._runtime.submit_event(
            make_end_of_turn(self._sid, ts + 10.0, final_text=text)
        )

    # ── output collector ─────────────────────────────────────────────────────

    async def _collect_outputs(self):
        """Background task: reads outputs from runtime and prints them."""
        while True:
            try:
                event = await self._runtime.next_output(
                    session_id=self._sid, timeout_ms=5000.0
                )
                self._print_output(event)
                self._output_log.append(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _print_output(self, event):
        etype = event.event_type
        payload = event.payload or {}
        ts = event.timestamp_ms
        gen = self._runtime.get_current_generation(self._sid)

        if etype == OutputEventType.FILLER:
            text = payload.get("text", "")
            print(f"\n  {_c(_CYAN, '🟢 [FAST-PATH]')} "
                  f"{_c(_CYAN, text)}"
                  f"  {_c(_DIM, f'[gen={gen}]')}")

        elif etype == OutputEventType.TOOL_CALL:
            tool = payload.get("tool_name", "?")
            args = payload.get("arguments", {})
            cid  = (payload.get("call_id") or "")[:8]
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"\n  {_c(_YELLOW, '🔧 [TOOL_CALL]')} "
                  f"{_c(_BOLD, tool)}({_c(_YELLOW, args_str)})"
                  f"  {_c(_DIM, f'[call={cid}... gen={gen}]')}")

        elif etype == OutputEventType.CLARIFICATION:
            text = payload.get("text", "")
            missing = payload.get("missing_slots", [])
            missing_str = f" (missing: {', '.join(missing)})" if missing else ""
            print(f"\n  {_c(_MAGENTA, '❓ [CLARIFY]')} "
                  f"{_c(_MAGENTA, text)}"
                  f"{_c(_DIM, missing_str)}")

        elif etype == OutputEventType.FINAL_RESPONSE:
            text = payload.get("text", "")
            snap = payload.get("state_snapshot") or {}
            intent = snap.get("intent", "—")
            slots  = snap.get("slots", {})
            slots_str = ", ".join(f"{k}: {v}" for k, v in slots.items()) if slots else "—"
            print(f"\n  {_c(_GREEN, '✅ [RESPONSE]')} "
                  f"{_c(_BOLD + _GREEN, text)}")
            print(f"  {_c(_DIM, '   STATE  →')} "
                  f"intent={_c(_WHITE, intent or '—')}  "
                  f"slots={{{_c(_WHITE, slots_str)}}}"
                  f"  {_c(_DIM, f'gen={gen}')}")
            print(_divider())

        elif etype == OutputEventType.CANCELLATION:
            cid = payload.get("cancelled_call_id", "")[:8]
            print(f"\n  {_c(_RED, '🔴 [CANCEL]')} "
                  f"call {cid}... superseded"
                  f"  {_c(_DIM, f'gen={gen}')}")

    # ── status ────────────────────────────────────────────────────────────────

    def print_status(self):
        gen = self._runtime.get_current_generation(self._sid)
        trace = self._runtime.tracer.entries()
        tasks_created   = sum(1 for e in trace if e.entry_type == "TASK_CREATED")
        tasks_completed = sum(1 for e in trace if e.entry_type == "TASK_COMPLETED")
        tasks_cancelled = sum(1 for e in trace if e.entry_type == "TASK_CANCELLED")
        stale_discarded = sum(1 for e in trace if e.entry_type == "STALE_RESULT_DISCARDED")
        outputs_total   = len(self._output_log)

        print(f"\n{_divider()}")
        print(f"  {_c(_BOLD, 'Session Status')}")
        print(f"  Session:          {self._sid[:16]}...")
        print(f"  Generation:       {_c(_BOLD, str(gen))}")
        print(f"  Tasks created:    {tasks_created}")
        print(f"  Tasks completed:  {_c(_GREEN, str(tasks_completed))}")
        print(f"  Tasks cancelled:  {_c(_RED, str(tasks_cancelled))}")
        print(f"  Stale discarded:  {_c(_YELLOW, str(stale_discarded))}")
        print(f"  Outputs emitted:  {outputs_total}")
        print(_divider())

    # ── built-in scenario ─────────────────────────────────────────────────────

    async def run_interruption_scenario(self):
        """Replays the canonical interruption scenario (like the eval harness)."""
        print(f"\n{_c(_BOLD, '  ▶ Running built-in interruption scenario...')}")
        print(f"  {_c(_DIM, 'T=0ms: User asks to book Seoul→Tokyo')}")
        await self.send_speech("book a flight to Tokyo on 2026-12-01")
        await asyncio.sleep(0.3)

        print(f"\n  {_c(_DIM, 'T=300ms: User INTERRUPTS — changes destination to Osaka')}")
        await self.send_interruption("actually change the destination to Osaka")
        await asyncio.sleep(0.8)

        print(f"\n  {_c(_DIM, 'Scenario complete. Stale Tokyo booking should be cancelled.')}")
        self.print_status()


# ──────────────────────────────────────────────────────────────────────────────
# Main REPL
# ──────────────────────────────────────────────────────────────────────────────

async def repl():
    session = DemoSession()
    await session.start()

    loop = asyncio.get_event_loop()

    try:
        while True:
            # Use thread executor for blocking input() call
            try:
                raw = await loop.run_in_executor(
                    None, lambda: input(f"\n  {_c(_BOLD, 'You')} › ")
                )
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_c(_DIM, 'Exiting...')}")
                break

            raw = raw.strip()
            if not raw:
                continue

            # ── commands ──────────────────────────────────────────────────────
            if raw == "/quit":
                break

            elif raw == "/status":
                session.print_status()

            elif raw == "/reset":
                await session.stop()
                session = DemoSession()
                await session.start()

            elif raw == "/scenario":
                await session.run_interruption_scenario()

            elif raw == "/help":
                print(
                    f"\n  Commands:\n"
                    f"    {_c(_CYAN, '/status')}    — show session metrics\n"
                    f"    {_c(_CYAN, '/scenario')}  — run built-in interruption demo\n"
                    f"    {_c(_CYAN, '/reset')}     — fresh session\n"
                    f"    {_c(_CYAN, '/quit')}      — exit\n"
                    f"\n  Prefix with {_c(_RED, '!')} to send as interruption: "
                    f"{_c(_DIM, '!change to Osaka')}\n"
                )

            # ── interruption ──────────────────────────────────────────────────
            elif raw.startswith("!"):
                text = raw[1:].strip()
                if text:
                    print(f"  {_c(_RED, '↯ [INTERRUPTING]')} {_c(_DIM, repr(text))}")
                    t0 = time.perf_counter()
                    await session.send_interruption(text)
                    await asyncio.sleep(0.05)
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"  {_c(_DIM, f'interrupt handled in {elapsed:.0f}ms')}")

            # ── normal speech ─────────────────────────────────────────────────
            else:
                print(f"  {_c(_DIM, '→ submitting speech turn...')}")
                t0 = time.perf_counter()
                await session.send_speech(raw)
                # Wait a bit for fast-path filler (should arrive < 200ms)
                await asyncio.sleep(0.5)
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"  {_c(_DIM, f'(turn processed in {elapsed:.0f}ms)')}")

    finally:
        await session.stop()
        print(f"\n  {_c(_DIM, 'Session closed. Goodbye.')}\n")


def main():
    # Windows requires ProactorEventLoop for subprocess support;
    # SelectorEventLoop works fine for our asyncio-only code.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(repl())


if __name__ == "__main__":
    main()
