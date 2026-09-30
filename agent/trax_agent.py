#!/usr/bin/env python3
"""
agent/trax_agent.py — TRAX Real-Time Voice Agent Entrypoint
Samsung PRISM GenAI Hackathon 3.0 — Theme 05: Interruptible Real-Time Agents

Clean top-level voice agent that implements the Two-Phase Transactional Tooling (TPTT)
and disfluency-aware TRP gating architecture without modifying external benchmark code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, AgentServer, llm, inference

# Load local environment
load_dotenv(".env")
load_dotenv(".env.local")

# LiveKit registers plugins when their modules are imported, and requires that
# registration to happen on the worker's main thread.  Import the selected
# provider here (module initialization), rather than lazily inside a job task.
_REALTIME_PLUGIN = None
_REALTIME_PROVIDER = os.getenv("LK_PROVIDER", "gemini2_5")
try:
    if _REALTIME_PROVIDER == "gemini2_5":
        from livekit.plugins import google as _REALTIME_PLUGIN
    elif _REALTIME_PROVIDER == "gpt_realtime":
        from livekit.plugins import openai as _REALTIME_PLUGIN
except Exception:
    _REALTIME_PLUGIN = None

# Compatibility for different livekit-agents versions
if hasattr(llm, "function_tool"):
    ai_callable_decorator = llm.function_tool
else:
    ai_callable_decorator = llm.ai_callable

# ── TRAX Core Layer ──────────────────────────────────────────────────────────
from prism import ToolCall, EffectClass, TRPState
from prism.trp_gate import TRPGate
from prism.tool_dispatcher import ToolDispatcher, EFFECT_MAP
from prism.saga_coordinator import SagaCoordinator
from prism.barge_in import BargeInController
from agent.tool_specs import SYSTEM_PROMPT, TOOL_SPECS
# ──────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("prism.agent")


# ------------------------------------------------------------------------------
# Mock API Resolution (Works against vanilla FDB or built-in fallback)
# ------------------------------------------------------------------------------

def get_api_registry():
    """
    Attempt to load MockAPIRegistry from external benchmark directory if present,
    otherwise provide self-contained mock backend.
    """
    # Check common benchmark locations
    candidate_paths = [
        Path(__file__).resolve().parent.parent / "scratch_vanilla_test" / "FDB-vanilla" / "v3",
        Path(__file__).resolve().parent.parent / "external" / "FDB-v3" / "v3",
    ]
    for p in candidate_paths:
        if (p / "mock_apis.py").exists():
            sys.path.insert(0, str(p))
            try:
                from mock_apis import MockAPIRegistry
                return MockAPIRegistry(latency_profile="instant")
            except Exception as exc:
                log.warning("Found %s but failed to import: %s", p, exc)

    class BuiltinMockRegistry:
        def call(self, function_name: str, **kwargs) -> dict:
            if function_name == "search_flights":
                return {"status": "success", "flights": [{"flight_id": "FL123", "price": 450}]}
            elif function_name in ("book_flight", "book_ticket"):
                return {"status": "success", "booking_id": "BK-999"}
            return {"status": "success", "function": function_name, "args": kwargs}

    return BuiltinMockRegistry()


# ------------------------------------------------------------------------------
# Latency Tracker
# ------------------------------------------------------------------------------

class LatencyTracker:
    def __init__(self):
        self.user_done_at = 0.0
        self.tool_start_at = 0.0
        self.tool_end_at = 0.0
        self.agent_start_at = 0.0
        self.query_received = False

    def reset(self):
        self.__init__()

    def log_breakdown(self, tool_name: str = "", room_name: str = "unknown"):
        if not self.user_done_at or not self.agent_start_at or not self.tool_start_at:
            return

        reasoning = (self.tool_start_at - self.user_done_at) if self.tool_start_at else 0
        execution = (self.tool_end_at - self.tool_start_at) if self.tool_start_at and self.tool_end_at else 0
        synthesis = (self.agent_start_at - (self.tool_end_at or self.user_done_at))
        total = self.agent_start_at - self.user_done_at

        report = f"\n⏱️ LATENCY BREAKDOWN ({tool_name}) for room {room_name}:\n"
        report += f"  - Reasoning (Model -> Tool): {reasoning:.2f}s\n"
        if execution:
            report += f"  - Tool Execution (API):    {execution:.2f}s\n"
        report += f"  - Synthesis (Tool -> Spoken): {synthesis:.2f}s\n"
        report += f"  - TOTAL SEARCH LATENCY:      {total:.2f}s\n"
        log.info(report)
        metrics = {
            "room": room_name,
            "tool_name": tool_name,
            "user_done_at": self.user_done_at,
            "tool_start_at": self.tool_start_at,
            "tool_end_at": self.tool_end_at,
            "agent_start_at": self.agent_start_at,
            "reasoning": round(reasoning, 6),
            "execution": round(execution, 6),
            "synthesis": round(synthesis, 6),
            "total": round(total, 6),
        }
        heartbeat_path = os.getenv("PRISM_HEARTBEAT_PATH", "/tmp/agent_heartbeat.log")
        try:
            with open(heartbeat_path, "a", encoding="utf-8") as heartbeat:
                heartbeat.write("LATENCY_TRACK_JSON: " + json.dumps(metrics) + "\n")
        except OSError as exc:
            log.warning("Unable to write latency record: %s", exc)


class SilenceTicker:
    """Emit elapsed silence marks and cancel them as soon as the user resumes."""

    def __init__(self, gate: TRPGate) -> None:
        self._gate = gate
        self._task: Optional[asyncio.Task] = None

    def user_started(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def user_stopped(self) -> None:
        self.user_started()

        async def emit_marks() -> None:
            previous = 0
            for mark in (300, 500, 900):
                await asyncio.sleep((mark - previous) / 1000.0)
                previous = mark
                await self._gate.on_silence_detected(mark)

        self._task = asyncio.create_task(emit_marks())


class TurnLifecycleCoordinator:
    """Serializes duplicate VAD onset events and preserves continued turns."""

    def __init__(
        self,
        gate: TRPGate,
        dispatcher: ToolDispatcher,
        saga: SagaCoordinator,
        barge: BargeInController,
        silence_ticker: SilenceTicker,
    ) -> None:
        self._gate = gate
        self._dispatcher = dispatcher
        self._saga = saga
        self._barge = barge
        self._silence_ticker = silence_ticker
        self._lock = asyncio.Lock()

    async def user_speech_started(self) -> None:
        self._silence_ticker.user_started()
        async with self._lock:
            if self._barge.is_handling_barge_in:
                return
            if await self._barge.on_user_speech_started():
                return
            # LISTENING and REPAIRING are an ongoing utterance.  Resetting here
            # would discard correction evidence and uncommitted work.
            if self._gate.state in (TRPState.TRP_CONFIRMED, TRPState.ABORTED):
                self._gate.reset_for_new_turn()
                self._dispatcher.reset_for_new_turn()
                self._saga.clear_for_new_turn()


# ------------------------------------------------------------------------------
# Assistant Function Tool Suite
# ------------------------------------------------------------------------------

class AssistantFnc:
    """Tool function declarations exposed to the real-time model."""

    def __init__(self, tracker: LatencyTracker, room_name: str, dispatcher: ToolDispatcher) -> None:
        self.room_name = room_name
        self.tracker = tracker
        self.dispatcher = dispatcher

    @ai_callable_decorator(description=TOOL_SPECS["search_flights"]["description"])
    async def search_flights(self, destination: str, date: str):
        """
        Args:
            destination: The city or airport. Preserve the user's exact wording.
            date: The travel date in natural-language format.
        """
        # Canonical benchmark date formatting
        if isinstance(date, str):
            m = re.match(r"^\d{4}-(\d{2})-(\d{2})$", date.strip())
            if m:
                try:
                    dt = datetime.strptime(date.strip(), "%Y-%m-%d")
                    date = f"{dt.strftime('%B')} {dt.day}"
                except Exception:
                    pass
            date = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", date.strip(), flags=re.IGNORECASE)

        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="search_flights",
            args={"destination": destination, "date": date},
            effect_class=EFFECT_MAP["search_flights"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["book_flight"]["description"])
    async def book_flight(self, passenger_name: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="book_flight",
            args={"passenger_name": passenger_name},
            effect_class=EFFECT_MAP["book_flight"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["update_identity_doc"]["description"])
    async def update_identity_doc(self, doc_type: str, doc_number: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="update_identity_doc",
            args={"doc_type": doc_type, "doc_number": doc_number},
            effect_class=EFFECT_MAP["update_identity_doc"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["get_card_benefits"]["description"])
    async def get_card_benefits(self, card_type: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="get_card_benefits",
            args={"card_type": card_type},
            effect_class=EFFECT_MAP["get_card_benefits"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["get_exchange_rate"]["description"])
    async def get_exchange_rate(self, amount: float, from_currency: str, to_currency: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="get_exchange_rate",
            args={"amount": amount, "from_currency": from_currency, "to_currency": to_currency},
            effect_class=EFFECT_MAP["get_exchange_rate"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["modify_autopay"]["description"])
    async def modify_autopay(self, bill_type: str, source_account: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="modify_autopay",
            args={"bill_type": bill_type, "source_account": source_account},
            effect_class=EFFECT_MAP["modify_autopay"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["search_apartments"]["description"])
    async def search_apartments(
        self,
        city: str = None,
        bedrooms: int = None,
        max_price: float = None,
        pets_allowed: bool = None,
    ):
        """
        Args:
            city: City to search in.
            bedrooms: Number of bedrooms required.
            max_price: Maximum monthly rent budget as a number (not a string).
            pets_allowed: Set to true when user mentions pets or pet-friendly. Required when pets mentioned.
        """
        args = {}
        if city is not None: args["city"] = city
        if bedrooms is not None: args["bedrooms"] = bedrooms
        if max_price is not None: args["max_price"] = max_price
        if pets_allowed is not None: args["pets_allowed"] = pets_allowed  # always include when explicitly provided

        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="search_apartments",
            args=args,
            effect_class=EFFECT_MAP["search_apartments"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["calculate_commute"]["description"])
    async def calculate_commute(self, origin_address: str, destination_address: str, mode: str = "driving"):
        """
        Args:
            origin_address: Starting address. Preserve any user-supplied description verbatim.
            destination_address: Destination. Preserve any user-supplied description verbatim.
            mode: Requested transport mode. Defaults to driving.
        """
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="calculate_commute",
            args={"origin_address": origin_address, "destination_address": destination_address, "mode": mode},
            effect_class=EFFECT_MAP["calculate_commute"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["update_search_filter"]["description"])
    async def update_search_filter(self, filter_name: str, value: Union[str, int, float, bool]):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="update_search_filter",
            args={"filter_name": filter_name, "value": value},
            effect_class=EFFECT_MAP["update_search_filter"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["track_order"]["description"])
    async def track_order(self, order_id: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="track_order",
            args={"order_id": order_id},
            effect_class=EFFECT_MAP["track_order"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["search_products"]["description"])
    async def search_products(self, query: str, max_price: float = None, category: str = None):
        args = {"query": query}
        if max_price is not None: args["max_price"] = max_price
        if category is not None: args["category"] = category

        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="search_products",
            args=args,
            effect_class=EFFECT_MAP["search_products"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description=TOOL_SPECS["add_to_cart"]["description"])
    async def add_to_cart(self, product_id: str, quantity: int = 1):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="add_to_cart",
            args={"product_id": product_id, "quantity": quantity},
            effect_class=EFFECT_MAP["add_to_cart"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json


# ------------------------------------------------------------------------------
# Realtime Model Provider Factory
# ------------------------------------------------------------------------------

def get_realtime_model():
    global _REALTIME_PLUGIN, _REALTIME_PROVIDER
    provider = os.getenv("LK_PROVIDER", "gemini2_5")
    if _REALTIME_PLUGIN is None or provider != _REALTIME_PROVIDER:
        try:
            if provider == "gemini2_5":
                from livekit.plugins import google as _REALTIME_PLUGIN
            elif provider == "gpt_realtime":
                from livekit.plugins import openai as _REALTIME_PLUGIN
            _REALTIME_PROVIDER = provider
        except Exception as exc:
            log.warning("Could not dynamically load realtime plugin for provider '%s': %s", provider, exc)

    if provider == "gemini2_5":
        if _REALTIME_PLUGIN is None:
            raise RuntimeError(
                "LiveKit Google plugin is not loaded. Ensure livekit-plugins-google is installed and worker is restarted."
            )
        return _REALTIME_PLUGIN.realtime.RealtimeModel(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
            voice=os.getenv("GOOGLE_VOICE", "Puck"),
            temperature=0.0,   # pin for deterministic, reproducible benchmark re-runs
            language="en-US",  # prevent auto-detection switching to non-English
        )
    elif provider == "gpt_realtime":
        if _REALTIME_PLUGIN is None:
            raise RuntimeError(
                "LiveKit OpenAI plugin is not loaded. Ensure livekit-plugins-openai is installed and worker is restarted."
            )
        return _REALTIME_PLUGIN.realtime.RealtimeModel(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview"),
            voice=os.getenv("OPENAI_VOICE", "alloy"),
            temperature=0.0,   # pin for deterministic, reproducible benchmark re-runs
        )
    raise ValueError(f"Unsupported provider: {provider}")


# ------------------------------------------------------------------------------
# Voice Agent and Session Lifecycle
# ------------------------------------------------------------------------------

class VoiceAgent(Agent):
    def __init__(self) -> None:
        # One shared, general-purpose instruction source prevents stale or
        # scenario-specific copies from drifting into the live agent.
        super().__init__(instructions=SYSTEM_PROMPT)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    print(f"!!! TRAX AGENT JOINING ROOM: {ctx.room.name} !!!")

    model = get_realtime_model()
    tracker = LatencyTracker()

    # ── TRAX Transactional Layer ─────────────────────────────────────────────
    # Every LiveKit room receives fresh mutable backend/session state.
    registry = get_api_registry()
    saga = SagaCoordinator()
    gate = TRPGate()
    dispatcher = ToolDispatcher(
        registry_fn=registry.call,
        saga=saga,
        tool_log_path=os.getenv("PRISM_TOOL_LOG_PATH", "/tmp/agent_tool_calls.log"),
        room_name=ctx.room.name,
        correction_epoch_fn=lambda: gate.correction_epoch,
    )
    gate.add_state_listener(dispatcher.on_trp_state_change)
    # ──────────────────────────────────────────────────────────────────────────

    fnc_ctx = AssistantFnc(tracker, ctx.room.name, dispatcher=dispatcher)
    tools = llm.find_function_tools(fnc_ctx)

    vad = inference.VAD(model="silero")
    session = AgentSession(
        llm=model,
        vad=vad,
        tools=tools,
        allow_interruptions=True,
        min_endpointing_delay=1.0,
        max_endpointing_delay=4.0,
        # Disable AEC warmup window — without this the first 3s of agent speech
        # silently ignore barge-in attempts (empirically seen in FDB-v3 travel_10).
        # Our BargeInController calls interrupt(force=True) directly, so warmup
        # suppression is redundant and harmful.
        aec_warmup_duration=0.0,
        min_interruption_duration=0.2,
    )

    barge = BargeInController(
        session=session,
        trp_gate=gate,
        dispatcher=dispatcher,
        saga=saga,
    )
    silence_ticker = SilenceTicker(gate)
    lifecycle = TurnLifecycleCoordinator(gate, dispatcher, saga, barge, silence_ticker)

    # ── Realtime Event Handlers ───────────────────────────────────────────────

    @session.on("user_input_transcribed")
    def on_user_input(msg: agents.voice.UserInputTranscribedEvent):
        # Safety net: If user speech is transcribed while agent is speaking, trigger barge-in immediately
        if (barge.is_agent_speaking or getattr(session, "agent_state", None) == "speaking") and not barge.is_handling_barge_in:
            log.info("barge_in: user transcript arrived while agent speaking — triggering barge-in cascade")
            asyncio.create_task(lifecycle.user_speech_started())

        if not tracker.query_received:
            tracker.user_done_at = time.time()
            tracker.query_received = True
            log.info("DEBUG: User query ended at %f", tracker.user_done_at)
        # Reset flag on each final segment so the NEXT turn gets a fresh user_done_at
        elif msg.is_final:
            tracker.query_received = False
        asyncio.create_task(gate.on_transcript_token(msg.transcript, msg.is_final))

    @session.on("agent_state_changed")
    def on_agent_state(ev: agents.voice.AgentStateChangedEvent):
        barge.set_agent_speaking(ev.new_state == "speaking")
        if ev.new_state == "speaking" and tracker.query_received and not tracker.agent_start_at:
            tracker.agent_start_at = time.time()
            # Run log_breakdown in a thread to avoid blocking the event loop
            asyncio.create_task(
                asyncio.to_thread(tracker.log_breakdown, "Search Tool", ctx.room.name)
            )
            tracker.reset()

        if ev.new_state == "listening" and not barge.is_handling_barge_in:
            gate.reset_for_new_turn()
            dispatcher.reset_for_new_turn()
            saga.clear_for_new_turn()

    @session.on("user_state_changed")
    def on_user_state_changed(ev: agents.voice.UserStateChangedEvent):
        new_state = getattr(ev, "new_state", None)
        if new_state == "speaking":
            asyncio.create_task(lifecycle.user_speech_started())
        elif new_state == "listening":
            silence_ticker.user_stopped()

    @session.on("overlapping_speech")
    def on_overlapping_speech(ev):
        asyncio.create_task(lifecycle.user_speech_started())

    @session.on("user_speech_started")
    def on_user_speech_started():
        asyncio.create_task(lifecycle.user_speech_started())

    await session.start(room=ctx.room, agent=VoiceAgent())
    print("!!! TRAX AGENT STARTED in ROOM (Listening) !!!")


if __name__ == "__main__":
    agents.cli.run_app(server)
