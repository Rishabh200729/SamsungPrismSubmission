"""
main_agent.py — Modular ParticipantAgent (Theme 05: Interruptible Real-Time Agents)

This is the composed agent for submission. It wires all 4 modules together
under a single Module 3 Runtime, then exposes the same `ParticipantAgent`
interface (in_queue / out_queue / setup / run) that the evaluation harness
expects.

Architecture:
  Harness in_queue
      │  (raw harness dicts)
      ▼
  HarnessAdapter._translate_in()  — converts to Module 3 BaseEvents
      │
      ▼
  Runtime.submit_event()          — Module 3 event bus
      │
      ├─▶ FastPathRouter           — Module 1: filler, ACK, interruption floor
      ├─▶ Module2ReasoningAdapter  — Module 2: NLU → slot tracking → TOOL_CALL
      └─▶ Module4Adapter           — Module 4: multimodal grounding, schema enforcement

  Runtime.next_output()           — Module 3 output queue
      │
      ▼
  HarnessAdapter._translate_out() — converts BaseEvents back to harness dicts
      │
      ▼
  Harness out_queue

Evaluation harness contract (from agent.py):
  - ParticipantAgent(in_queue, out_queue)
  - async setup()   — called once before run()
  - async run()     — main event loop; drives until scenario_end
  - Output dicts must have keys: action, payload[text], optionally state_snapshot
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "module_2"))   # bare imports inside module_2

# ── Module 3 imports ──────────────────────────────────────────────────────────
from module3 import Runtime
from module3.runtime.config import PRODUCTION_CONFIG, TEST_CONFIG
from module3.runtime.events import (
    InputEventType,
    OutputEventType,
    make_text_chunk,
    make_end_of_turn,
    make_interruption,
    make_tool_result,
    make_tool_manifest,
    make_audio_wav,
    make_video_frame,
)

# ── Module 1 ──────────────────────────────────────────────────────────────────
from module1 import FastPathRouter

# ── Module 2 ──────────────────────────────────────────────────────────────────
from orchestrator import Module2ReasoningAdapter

# ── Module 4 ──────────────────────────────────────────────────────────────────
from module4.adapter import Module4Adapter

logger = logging.getLogger("main_agent")


# ──────────────────────────────────────────────────────────────────────────────
# Harness ↔ Module 3 translation layer
# ──────────────────────────────────────────────────────────────────────────────

class HarnessAdapter:
    """
    Converts between the evaluation harness's flat dict protocol and
    Module 3's typed BaseEvent objects.

    Harness → Runtime (inbound):
        user_speech_chunk  → TEXT_CHUNK / END_OF_TURN
        user_audio_chunk   → AUDIO_WAV (+ END_OF_TURN on end_of_turn=True)
        video_frame        → VIDEO_FRAME
        interruption       → INTERRUPTION
        tool_result        → TOOL_RESULT
        tool_manifest      → TOOL_MANIFEST
        scenario_end       → (signals run loop to stop)

    Runtime → Harness (outbound):
        FILLER             → filler_speech
        CLARIFICATION      → clarification_request
        TOOL_CALL          → tool_call
        FINAL_RESPONSE     → final_response
        CANCELLATION       → cancel_tool  (internal; harness may not act on it)
    """

    # Map harness event_type strings → translation methods
    _IN_HANDLERS: dict

    def __init__(self, runtime: Runtime, session_id: str, out_queue: asyncio.Queue):
        self._rt = runtime
        self._sid = session_id
        self._out_q = out_queue
        self._text_buffer: list[str] = []
        self._scenario_done = False

    # ── inbound ───────────────────────────────────────────────────────────────

    async def handle_harness_event(self, event: Dict[str, Any]) -> bool:
        """
        Translate and submit one harness event. Returns False when the
        scenario_end sentinel is received (caller should stop the loop).
        """
        etype = event.get("event_type", "")
        payload = event.get("payload", {}) or {}
        ts = self._rt.clock.now()

        if etype == "tool_manifest":
            tools = payload.get("tools", {})
            # harness delivers tools as a dict {name: spec}; Module 3 wants a list
            tools_list = _manifest_dict_to_list(tools)
            await self._rt.submit_event(
                make_tool_manifest(self._sid, ts, tools=tools_list)
            )

        elif etype == "user_speech_chunk":
            text = payload.get("text", "")
            is_eot = payload.get("end_of_turn", False)
            if text:
                self._text_buffer.append(text)
                await self._rt.submit_event(
                    make_text_chunk(self._sid, ts, text=text,
                                    is_partial=not is_eot)
                )
            if is_eot:
                final = " ".join(self._text_buffer).strip()
                self._text_buffer.clear()
                await self._rt.submit_event(
                    make_end_of_turn(self._sid, ts, final_text=final or text)
                )

        elif etype == "user_audio_chunk":
            audio_ref = payload.get("audio_ref") or payload.get("audio_b64", "")
            duration = payload.get("duration_ms", 500.0)
            is_eot = payload.get("end_of_turn", False)
            if audio_ref:
                await self._rt.submit_event(
                    make_audio_wav(self._sid, ts,
                                   audio_b64=audio_ref, duration_ms=duration)
                )
            if is_eot:
                # Treat audio end-of-turn as an END_OF_TURN with empty text;
                # Module 2 / Module 4 will handle transcription via audio events.
                await self._rt.submit_event(
                    make_end_of_turn(self._sid, ts, final_text="")
                )

        elif etype == "video_frame":
            frame_ref = payload.get("image_ref") or payload.get("frame_b64", "")
            frame_idx = payload.get("frame_index", 0)
            await self._rt.submit_event(
                make_video_frame(
                    self._sid, ts,
                    frame_b64=frame_ref,
                    width=payload.get("width", 640),
                    height=payload.get("height", 480),
                    frame_index=frame_idx,
                )
            )

        elif etype == "interruption":
            text = payload.get("text", "")
            self._text_buffer.clear()
            await self._rt.submit_event(
                make_interruption(self._sid, ts,
                                   text=text,
                                   reason=payload.get("reason", "user_interruption"))
            )

        elif etype == "tool_result":
            call_id = payload.get("call_id", "")
            task_id = payload.get("task_id") or call_id
            result = payload.get("result") or {}
            success = payload.get("status", "success") == "success"
            error = payload.get("result", {}).get("error") if not success else None
            await self._rt.submit_event(
                make_tool_result(
                    session_id=self._sid, timestamp_ms=ts,
                    call_id=call_id, task_id=task_id,
                    result=result, success=success, error=error,
                )
            )

        elif etype == "scenario_end":
            return False   # tell caller to stop

        else:
            logger.debug("HarnessAdapter: unknown event_type=%s, ignored", etype)

        return True   # continue

    # ── outbound ──────────────────────────────────────────────────────────────

    async def collect_and_forward_outputs(self, timeout_ms: float = 200.0):
        """
        Drain all currently available outputs from the runtime and
        translate them into harness dicts placed on out_queue.
        """
        try:
            while True:
                out_event = await self._rt.next_output(
                    session_id=self._sid, timeout_ms=timeout_ms
                )
                harness_msg = self._translate_out(out_event)
                if harness_msg:
                    await self._out_q.put(harness_msg)
        except (asyncio.TimeoutError, Exception):
            pass   # timeout = no more outputs right now

    def _translate_out(self, event) -> Optional[Dict[str, Any]]:
        """Convert a Module 3 output BaseEvent → harness dict."""
        etype = event.event_type
        payload = event.payload or {}

        if etype == OutputEventType.FILLER:
            return {
                "action": "filler_speech",
                "payload": {"text": payload.get("text", "")},
            }

        elif etype == OutputEventType.CLARIFICATION:
            return {
                "action": "clarification_request",
                "payload": {"text": payload.get("text", "")},
                "missing_slots": payload.get("missing_slots", []),
            }

        elif etype == OutputEventType.TOOL_CALL:
            return {
                "action": "tool_call",
                "payload": {
                    "call_id": payload.get("call_id", ""),
                    "api_name": payload.get("tool_name", ""),
                    "args": payload.get("arguments", {}),
                },
            }

        elif etype == OutputEventType.FINAL_RESPONSE:
            snap = payload.get("state_snapshot") or {}
            return {
                "action": "final_response",
                "payload": {"text": payload.get("text", "")},
                "state_snapshot": snap,
            }

        elif etype == OutputEventType.CANCELLATION:
            return {
                "action": "cancel_tool",
                "payload": {"call_id": payload.get("cancelled_call_id", "")},
            }

        else:
            logger.debug("HarnessAdapter: unrecognised output event_type=%s", etype)
            return None


# ──────────────────────────────────────────────────────────────────────────────
# ParticipantAgent (the class the harness instantiates)
# ──────────────────────────────────────────────────────────────────────────────

class ParticipantAgent:
    """
    Dual-process real-time agent using the modular PRISM stack.

    Drop-in replacement for the monolithic agent.py:
        - Same constructor signature: (in_queue, out_queue)
        - Same async setup() / run() interface
        - All evaluation criteria (interruption recovery, latency, stale
          result discarding, idempotency, multimodal grounding) are
          handled by the underlying modules, not in-lined here.
    """

    def __init__(self, in_queue: asyncio.Queue, out_queue: asyncio.Queue):
        self.in_q = in_queue
        self.out_q = out_queue

        # Runtime and module instances created in setup()
        self._runtime: Optional[Runtime] = None
        self._session_id: Optional[str] = None
        self._adapter: Optional[HarnessAdapter] = None
        self._module4: Optional[Module4Adapter] = None

    async def setup(self):
        """
        Boot the modular stack. Called once by the harness before run().
        """
        # Use TEST_CONFIG for deterministic virtual-clock mode;
        # swap to PRODUCTION_CONFIG for wall-clock / real-latency mode.
        self._runtime = Runtime(config=TEST_CONFIG)

        await self._runtime.start()
        self._session_id = self._runtime.create_session()

        # Module 1: fast-path filler, ACK, interruption floor management
        _fast_path = FastPathRouter(self._runtime, self._runtime.clock)

        # Module 2: NLU → slot tracking → tool call → result → final response
        reasoning = Module2ReasoningAdapter(self._runtime)
        reasoning.install()

        # Module 4: multimodal grounding + ambiguity clarification
        # Wire Module 4's evaluate_ambiguity() into Module 2 via direct call
        # (race-free path, per module4/adapter.py docstring).
        self._module4 = Module4Adapter(self._runtime)
        self._module4.install()

        # Harness ↔ Runtime translation layer
        self._adapter = HarnessAdapter(
            runtime=self._runtime,
            session_id=self._session_id,
            out_queue=self.out_q,
        )

        logger.info(
            "ParticipantAgent.setup complete: session_id=%s", self._session_id
        )

    async def run(self):
        """
        Main event loop. Reads harness events, translates and submits them to
        the runtime, then collects and forwards any outputs that the modules
        produced. Runs until the scenario_end event is received.
        """
        if self._runtime is None or self._adapter is None:
            raise RuntimeError("setup() must be called before run()")

        try:
            while True:
                # Get next event from harness
                event = await self.in_q.get()

                try:
                    # Submit to runtime (translates harness dict → BaseEvent)
                    should_continue = await self._adapter.handle_harness_event(event)
                except Exception as exc:
                    logger.error("Error handling event %s: %r", event.get("event_type"), exc)
                    should_continue = True
                    await self._safe_fallback()

                # Give event loop a tick for handlers to execute
                await asyncio.sleep(0)
                await asyncio.sleep(0)

                # Collect any outputs the modules produced and forward to harness
                try:
                    await self._adapter.collect_and_forward_outputs(timeout_ms=150.0)
                except Exception as exc:
                    logger.error("Error collecting outputs: %r", exc)

                if not should_continue:
                    break

        finally:
            await self._shutdown()

    async def _safe_fallback(self):
        """Emit a safe fallback response if a handler crashes."""
        try:
            await self.out_q.put({
                "action": "final_response",
                "payload": {"text": "Sorry, something went wrong — could you try again?"},
                "state_snapshot": {"intent": None, "slots": {}},
            })
        except Exception:
            pass

    async def _shutdown(self):
        """Clean up modules and stop the runtime."""
        try:
            if self._module4 and self._session_id:
                self._module4.drop_session(self._session_id)
            if self._runtime and self._session_id:
                await self._runtime.close_session(self._session_id)
            if self._runtime:
                await self._runtime.stop()
        except Exception as exc:
            logger.warning("Shutdown error (non-fatal): %r", exc)


# Keep the alias for harness scripts that reference BaselineAgent
BaselineAgent = ParticipantAgent


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _manifest_dict_to_list(tools_dict: Dict[str, Any]) -> list:
    """
    Convert harness tool manifest format (dict keyed by name) to the list
    format that make_tool_manifest() and ToolManifest.load_from_event_payload()
    expect: [{name, side_effect, parameters}, ...].
    """
    if isinstance(tools_dict, list):
        return tools_dict  # already in list form
    result = []
    for name, spec in tools_dict.items():
        kind = spec.get("kind", "read_only")
        side_effect = "STATE_MODIFYING" if kind == "state_modifying" else "READ_ONLY"
        # Convert args spec to parameters format
        args = spec.get("args", {}) or {}
        parameters = {}
        for arg_name, arg_spec in args.items():
            parameters[arg_name] = {
                "required": arg_spec.get("required", False),
                "description": arg_spec.get("description", arg_name),
                "type": arg_spec.get("type", "string"),
            }
        result.append({
            "name": name,
            "side_effect": side_effect,
            "parameters": parameters,
            "description": spec.get("description", ""),
        })
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (python main_agent.py)
# ──────────────────────────────────────────────────────────────────────────────

async def _self_test():
    """Smoke test: boot the agent, send a basic flight booking turn, check output."""
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()

    agent = ParticipantAgent(in_q, out_q)
    await agent.setup()

    run_task = asyncio.create_task(agent.run())

    # Inject a tool manifest first
    await in_q.put({
        "event_type": "tool_manifest",
        "payload": {
            "tools": {
                "book_flight": {
                    "kind": "state_modifying",
                    "args": {
                        "destination": {"required": True},
                        "date": {"required": True},
                    },
                }
            }
        },
    })

    await asyncio.sleep(0.05)

    # Send a speech turn
    await in_q.put({
        "event_type": "user_speech_chunk",
        "payload": {"text": "book a flight to Goa on 2026-12-01", "end_of_turn": True},
    })

    # Give agent time to react
    await asyncio.sleep(0.5)

    # End scenario
    await in_q.put({"event_type": "scenario_end", "payload": {}})
    await asyncio.wait_for(run_task, timeout=3.0)

    # Drain outputs
    outputs = []
    while not out_q.empty():
        outputs.append(out_q.get_nowait())

    print("\n=== Self-test outputs ===")
    for o in outputs:
        print(f"  [{o['action']}] {o.get('payload', {}).get('text', '')}")
    print(f"\n  Total outputs: {len(outputs)}")
    has_tool_call = any(o["action"] == "tool_call" for o in outputs)
    print(f"  TOOL_CALL emitted: {has_tool_call}")
    print("  PASS\n" if has_tool_call else "  FAIL — expected a TOOL_CALL\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_self_test())
