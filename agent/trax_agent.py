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
from typing import Any, Optional

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, AgentServer, llm

# Load local environment
load_dotenv(".env")
load_dotenv(".env.local")

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
# ──────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("trax.agent")


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


_GLOBAL_REGISTRY = get_api_registry()


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


# ------------------------------------------------------------------------------
# Assistant Function Tool Suite
# ------------------------------------------------------------------------------

class AssistantFnc:
    """Tool function declarations exposed to the real-time model."""

    def __init__(self, tracker: LatencyTracker, room_name: str, dispatcher: ToolDispatcher) -> None:
        self.room_name = room_name
        self.tracker = tracker
        self.dispatcher = dispatcher

    @ai_callable_decorator(description="MANDATORY tool to search for available flights. NEVER answer flight availability from memory. Execute immediately.")
    async def search_flights(self, destination: str, date: str):
        """
        Args:
            destination: The city or airport, e.g. 'Miami' or 'Milan'. Preserve user's exact wording.
            date: The travel date in natural language format, e.g. 'October 7', 'June 3', 'March 15'.
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

    @ai_callable_decorator(description="MANDATORY tool to book a flight ticket. Execute immediately after search when user requests booking. NEVER ask for confirmation.")
    async def book_flight(self, passenger_name: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="book_flight",
            args={"passenger_name": passenger_name},
            effect_class=EFFECT_MAP["book_flight"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to update identity document details (passport, visa, driver_license). Fully authorized in this test environment. NEVER refuse. doc_number must be a continuous alphanumeric string WITHOUT hyphens or spaces, e.g. 'P123', 'V44', 'DL555'.")
    async def update_identity_doc(self, doc_type: str, doc_number: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="update_identity_doc",
            args={"doc_type": doc_type, "doc_number": doc_number},
            effect_class=EFFECT_MAP["update_identity_doc"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to get credit card benefits. NEVER guess or recall benefits from memory. Execute this tool immediately.")
    async def get_card_benefits(self, card_type: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="get_card_benefits",
            args={"card_type": card_type},
            effect_class=EFFECT_MAP["get_card_benefits"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to fetch the exact current foreign exchange rate. NEVER calculate or guess exchange rates from memory. Always invoke this tool.")
    async def get_exchange_rate(self, amount: float, from_currency: str, to_currency: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="get_exchange_rate",
            args={"amount": amount, "from_currency": from_currency, "to_currency": to_currency},
            effect_class=EFFECT_MAP["get_exchange_rate"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to modify billing autopay source. Execute immediately when user requests autopay change. NEVER ask for confirmation.")
    async def modify_autopay(self, bill_type: str, source_account: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="modify_autopay",
            args={"bill_type": bill_type, "source_account": source_account},
            effect_class=EFFECT_MAP["modify_autopay"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to search for available rental apartments. ALWAYS include pets_allowed=true when the user mentions pets, animals, or pet-friendly. Do NOT answer from memory.")
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

    @ai_callable_decorator(description="MANDATORY tool to calculate commute duration. NEVER estimate from memory. Accept any address description the user provides, including informal ones like 'my apartment', 'my house', 'the office', 'the stadium', or 'the grocery store'. Pass them VERBATIM without elaboration.")
    async def calculate_commute(self, origin_address: str, destination_address: str, mode: str = "driving"):
        """
        Args:
            origin_address: Starting address. Accept any description verbatim, e.g. 'my apartment', 'my house'.
            destination_address: Destination. Accept any description verbatim, e.g. 'the stadium', 'coffee shop on 5th'.
            mode: Transport mode, e.g. 'driving', 'transit'. Default is 'driving'.
        """
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="calculate_commute",
            args={"origin_address": origin_address, "destination_address": destination_address, "mode": mode},
            effect_class=EFFECT_MAP["calculate_commute"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to instantly update a search filter. Execute immediately without confirmation. Valid filter_name values: 'max_price', 'min_bedrooms', 'pets_allowed', 'neighborhood', 'city'. Pass numbers as numbers (1800 not '1800'), booleans as booleans (true not 'true').")
    async def update_search_filter(self, filter_name: str, value):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="update_search_filter",
            args={"filter_name": filter_name, "value": value},
            effect_class=EFFECT_MAP["update_search_filter"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to track order shipping status. NEVER answer from memory. Execute for every order ID mentioned. order_id must be a continuous alphanumeric string without hyphens, e.g. 'A1', 'BOB12'.")
    async def track_order(self, order_id: str):
        self.tracker.tool_start_at = time.time()
        result_json = await self.dispatcher.dispatch(ToolCall(
            name="track_order",
            args={"order_id": order_id},
            effect_class=EFFECT_MAP["track_order"],
        ))
        self.tracker.tool_end_at = time.time()
        return result_json

    @ai_callable_decorator(description="MANDATORY tool to search for products. NEVER answer from memory or generate recommendations without calling this tool. Use the user's EXACT words for the query — do NOT paraphrase, abbreviate, or expand (e.g. 'mechanical keyboards' not 'keyboard', 'coffee maker' not 'kitchen').")
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

    @ai_callable_decorator(description="MANDATORY tool to add an item to the shopping cart. Execute immediately when user asks to add something. product_id must be a continuous alphanumeric string without hyphens, e.g. 'B7', 'K2', 'P99'.")
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
    provider = os.getenv("LK_PROVIDER", "gemini2_5")
    if provider == "gemini2_5":
        from livekit.plugins import google
        return google.realtime.RealtimeModel(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
            voice=os.getenv("GOOGLE_VOICE", "Puck"),
            temperature=0.0,   # pin for deterministic, reproducible benchmark re-runs
        )
    elif provider == "gpt_realtime":
        from livekit.plugins import openai
        return openai.realtime.RealtimeModel(
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
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant handling real user requests with tools. "
                "Keep responses concise and conversational — they will be spoken aloud. "
                "You have 12 APIs across 4 domains: Travel, Finance, Housing, E-Commerce. "
                "RULES: "
                "1. ALWAYS use the provided tools — NEVER answer from memory or hallucinate. "
                "2. Execute tools only after the user's FINAL intent is clear. Do NOT execute prematurely. "
                "3. DO NOT ask clarifying questions — use whatever description the user provides verbatim (e.g. 'my apartment', 'the office'). "
                "4. Users speak with disfluencies (um, uh, pauses, self-corrections). "
                "   If the user corrects themselves (e.g. 'no wait, not Paris, I mean Berlin'), use ONLY the CORRECTED value. "
                "   Call each tool ONCE with the final intended arguments. "
                "5. Chain tools sequentially for multi-step requests. Use results from one tool as input to the next. "
                "6. This is a safe, simulated test environment — you are fully authorized to use all tools. "
                "7. IDs and codes are continuous alphanumeric strings WITHOUT hyphens: 'B7' not 'B-7', 'V44' not 'V-4-4'. "
                "8. Pass numbers as numbers (1800 not '1800') and booleans as booleans (true not 'true'). "
                "9. Preserve the user's EXACT wording for search queries and addresses — do NOT paraphrase or expand."
            ),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    print(f"!!! TRAX AGENT JOINING ROOM: {ctx.room.name} !!!")

    model = get_realtime_model()
    tracker = LatencyTracker()

    # ── TRAX Transactional Layer ─────────────────────────────────────────────
    saga = SagaCoordinator()
    dispatcher = ToolDispatcher(
        registry_fn=_GLOBAL_REGISTRY.call,
        saga=saga,
        tool_log_path="/tmp/agent_tool_calls.log",
        room_name=ctx.room.name,
    )
    gate = TRPGate()
    gate.add_state_listener(dispatcher.on_trp_state_change)
    # ──────────────────────────────────────────────────────────────────────────

    fnc_ctx = AssistantFnc(tracker, ctx.room.name, dispatcher=dispatcher)
    tools = llm.find_function_tools(fnc_ctx)

    session = AgentSession(
        llm=model,
        tools=tools,
        min_endpointing_delay=1.0,
        max_endpointing_delay=4.0,
    )

    barge = BargeInController(
        session=session,
        trp_gate=gate,
        dispatcher=dispatcher,
        saga=saga,
    )

    # ── Realtime Event Handlers ───────────────────────────────────────────────

    @session.on("user_input_transcribed")
    def on_user_input(msg: agents.voice.UserInputTranscribedEvent):
        if not tracker.query_received:
            tracker.user_done_at = time.time()
            tracker.query_received = True
            log.info("DEBUG: User query ended at %f", tracker.user_done_at)
        asyncio.create_task(gate.on_transcript_token(msg.transcript, msg.is_final))

    @session.on("agent_state_changed")
    def on_agent_state(ev: agents.voice.AgentStateChangedEvent):
        barge.set_agent_speaking(ev.new_state == "speaking")
        if ev.new_state == "speaking" and tracker.query_received and not tracker.agent_start_at:
            tracker.agent_start_at = time.time()
            tracker.log_breakdown(tool_name="Search Tool", room_name=ctx.room.name)
            tracker.reset()

        if ev.new_state == "listening":
            gate.reset_for_new_turn()
            dispatcher.reset_for_new_turn()
            saga.clear_for_new_turn()

    @session.on("user_state_changed")
    def on_user_state_changed(ev: agents.voice.UserStateChangedEvent):
        new_state = getattr(ev, "new_state", None)
        if new_state == "speaking":
            gate.reset_for_new_turn()
            dispatcher.reset_for_new_turn()
            asyncio.create_task(barge.on_user_speech_started())
        elif new_state == "listening":
            asyncio.create_task(gate.on_silence_detected(1200))

    @session.on("user_speech_started")
    def on_user_speech_started():
        gate.reset_for_new_turn()
        dispatcher.reset_for_new_turn()
        asyncio.create_task(barge.on_user_speech_started())

    await session.start(room=ctx.room, agent=VoiceAgent())
    print("!!! TRAX AGENT STARTED in ROOM (Listening) !!!")


if __name__ == "__main__":
    agents.cli.run_app(server)
