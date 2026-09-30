#!/usr/bin/env python3
"""
agent/extension_incar.py — Use-Case Extension: In-Car Navigation Agent
Samsung PRISM GenAI Hackathon 3.0 — Theme 05 (Extension, 20% of Round 1 score)

A driver navigates by voice and changes their mind mid-route:
"take me to the city mall... no wait, actually the airport".
The TRAX stack that protects the FDB-v3 benchmark now protects a stateful,
real-world action (the vehicle's active route):

  TRPGate           holds the floor while the driver repairs the sentence
  ToolDispatcher    never lets the stale destination reach the vehicle
  SagaCoordinator   remembers the previous route so it can be restored
  BargeInController driver talks over the agent -> audio flush + rollback

Nothing under prism/ is modified. This module adds:
  * 3 tools: get_route (READ_ONLY), find_nearby (READ_ONLY),
             update_destination (COMPENSABLE: start / change active route)
  * InCarDispatcher: last-writer-wins for update_destination, route-restore
    compensation, and no orphaned futures on turn reset
  * InCarBargeIn: rolls the route back only if the interrupting speech is a
    correction/new command, not an acknowledgement ("okay, thanks")
  * mock geocoder: any place name resolves deterministically
  * SilenceTicker: feeds the gate real, growing silence durations (300/900 ms)
  * a LiveKit voice entrypoint and an offline, deterministic demo

Run (repo root):
  python -m agent.extension_incar --demo   # offline, no API keys, exit 0 = all checks pass
  python -m agent.extension_incar dev      # live LiveKit voice agent
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path bootstrap — ensures `prism` is importable when this file is run
# directly (`python agent/extension_incar.py`) from any working directory.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---------------------------------------------------------------------------

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from prism import CompensationEntry, EffectClass, ToolCall, TRPState
from prism.barge_in import BargeInController
from prism.saga_coordinator import SagaCoordinator
from prism.tool_dispatcher import EFFECT_MAP, ToolDispatcher, _ABORTED, _SUPERSEDED, _DISCARDED
from prism.trp_gate import _CORRECTION_TERMS_RE, VAD_SILENCE_NORMAL_MS, VAD_SILENCE_REPAIRING_MS, TRPGate

try:  # optional: offline demo and unit tests need no LiveKit / dotenv
    from dotenv import load_dotenv

    load_dotenv(".env")
    load_dotenv(".env.local")
except ImportError:
    pass

try:
    from livekit import agents
    from livekit.agents import Agent, AgentServer, AgentSession, llm
    from agent.trax_agent import get_realtime_model

    _HAS_LIVEKIT = True
except ImportError:
    _HAS_LIVEKIT = False

log = logging.getLogger("trax.incar")


# ------------------------------------------------------------------------------
# Effect classification for the extension's tools.
# EFFECT_MAP falls back to READ_ONLY for unknown names (fail-open), so every
# mutating tool MUST be registered here or it would be executed speculatively.
# ------------------------------------------------------------------------------

EFFECT_MAP.update(
    {
        "get_route": EffectClass.READ_ONLY,          # preview route + ETA, no state change
        "find_nearby": EffectClass.READ_ONLY,        # places near the vehicle
        "update_destination": EffectClass.COMPENSABLE,  # start / change active navigation
    }
)

# Mutating tools where a later call fully replaces an earlier, still-staged one.
_LAST_WRITER_WINS = frozenset({"update_destination"})


# ------------------------------------------------------------------------------
# Mock navigation backend (stateful per session)
# ------------------------------------------------------------------------------

# name -> (x_km, y_km, category). Fictional city grid; the vehicle starts at (0, 0).
_PLACES: dict[str, tuple[float, float, str]] = {
    "downtown hotel": (2.0, 3.0, "hotel"),
    "city mall": (5.0, -6.0, "shopping"),
    "central station": (0.0, 6.0, "station"),
    "airport": (18.0, -4.0, "airport"),
    "main street fuel": (4.0, 1.0, "gas_station"),
    "ring road fuel": (10.0, -1.0, "gas_station"),
    "airport fuel stop": (16.0, -3.0, "gas_station"),
    "central ev charging": (1.0, 2.0, "ev_charging"),
    "harbor ev charging": (7.0, 5.0, "ev_charging"),
    "riverside cafe": (3.0, 4.0, "cafe"),
    "city hospital": (-3.0, 4.0, "hospital"),
}

_CATEGORY_ALIASES = {
    "gas": "gas_station", "gas station": "gas_station", "gas stations": "gas_station",
    "fuel": "gas_station", "petrol": "gas_station", "petrol pump": "gas_station",
    "ev charging": "ev_charging", "ev charger": "ev_charging", "charging": "ev_charging",
    "charger": "ev_charging", "charging station": "ev_charging",
    "coffee": "cafe", "coffee shop": "cafe", "cafe": "cafe",
    "hospital": "hospital", "hotel": "hotel",
}

_AVG_MIN_PER_KM = 1.5  # ~40 km/h


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an|my|our)\s+")


def _synthetic_xy(key: str) -> tuple[float, float]:
    """Stable pseudo-location in a 50x50 km box (sha256, not hash(): must survive restarts)."""
    d = hashlib.sha256(key.encode("utf-8")).digest()
    return d[0] / 255 * 50 - 25, d[1] / 255 * 50 - 25


def _locate(text: str) -> Optional[tuple[str, float, float]]:
    """
    Mock geocoder. Known places resolve to their fixed coordinates; any other
    non-empty name resolves to a deterministic synthetic location, like a real
    geocoder that finds "Delhi airport" or "Sector 17 market". Empty input -> None.
    """
    q = _LEADING_ARTICLE_RE.sub("", _norm(text))
    if not q:
        return None
    if q in _PLACES:
        return q, _PLACES[q][0], _PLACES[q][1]
    hits = [n for n in _PLACES if q in n]
    if len(hits) != 1:
        tokens = set(q.split())
        hits = [n for n in _PLACES if tokens <= set(n.split())]
    if len(hits) == 1:
        return hits[0], _PLACES[hits[0]][0], _PLACES[hits[0]][1]
    x, y = _synthetic_xy(q)
    return q, x, y


def _leg(x: float, y: float) -> dict:
    d = math.hypot(x, y)
    return {"distance_km": round(d, 1), "eta_min": max(1, round(d * _AVG_MIN_PER_KM))}


class NavRegistry:
    """
    Stateful mock of a navigation backend. `call(name, **kwargs)` matches the
    registry_fn contract of ToolDispatcher (synchronous; runs in an executor thread).
    `mutations` is the ground truth of what actually reached the vehicle.
    """

    def __init__(self, latency_s: float = 0.15) -> None:
        self.latency_s = latency_s
        self.active_destination: Optional[str] = None
        self.mutations: list[dict] = []
        self._tools = {
            "get_route": self._get_route,
            "find_nearby": self._find_nearby,
            "update_destination": self._update_destination,
        }

    def call(self, function_name: str, **kwargs: Any) -> dict:
        time.sleep(self.latency_s)  # simulated API latency
        fn = self._tools.get(function_name)
        if fn is None:
            return {"status": "error", "error": "unknown_tool"}
        try:
            return fn(**kwargs)
        except TypeError:
            return {"status": "error", "error": "invalid_args"}

    def restore_destination(self, previous: Optional[str]) -> None:
        """Saga compensation primitive (not model-facing)."""
        self.mutations.append({"op": "restore", "from": self.active_destination, "to": previous})
        self.active_destination = previous

    # -- tools ---------------------------------------------------------------

    # ---- route telemetry helpers ----
    _ROAD_NAMES: dict[str, str] = {
        "airport": "Airport Expressway",
        "city mall": "Ring Road South",
        "central station": "Central Boulevard",
        "downtown hotel": "Main Street",
        "city hospital": "Hospital Lane",
        "main street fuel": "Main Street",
        "ring road fuel": "Ring Road",
        "harbor ev charging": "Harbor Drive",
        "riverside cafe": "Riverside Drive",
    }
    _TRAFFIC: list[str] = ["clear roads", "light traffic", "moderate traffic", "light traffic"]

    def _route_meta(self, name: str, d: float) -> dict:
        road = self._ROAD_NAMES.get(name, "Main Road")
        # deterministic traffic tier from distance bucket
        traffic = self._TRAFFIC[min(3, int(d / 5))]
        turn_dist = max(200, round(d * 0.4 * 1000 / 10) * 10)
        return {
            "via_road": road,
            "traffic": traffic,
            "next_maneuver": f"Stay on {road} for {turn_dist} m, then continue straight.",
        }

    def _get_route(self, destination: str) -> dict:
        loc = _locate(destination)
        if loc is None:
            return {"status": "error", "error": "not_found", "query": destination}
        name, x, y = loc
        leg = _leg(x, y)
        return {
            "status": "success",
            "destination": name,
            "navigation_started": False,
            **leg,
            **self._route_meta(name, leg["distance_km"]),
        }

    def _find_nearby(self, category: str) -> dict:
        key = _norm(category)
        if not key:
            return {"status": "error", "error": "not_found", "query": category}
        cat = _CATEGORY_ALIASES.get(key, key.replace(" ", "_"))
        hits = sorted((math.hypot(x, y), n) for n, (x, y, c) in _PLACES.items() if c == cat)[:3]
        if not hits:  # unknown category: deterministic synthetic results
            label = cat.replace("_", " ")
            hits = sorted(
                (math.hypot(*_synthetic_xy(f"{cat}:{i}")), f"{label} {sfx}")
                for i, sfx in enumerate(("on Main Street", "on Ring Road", "near Central"))
            )
        results = [{"name": n, "distance_km": round(d, 1)} for d, n in hits]
        nearest_turn = f"In {max(100, round(hits[0][0]*300/10)*10)} m, turn right."
        return {
            "status": "success",
            "category": cat,
            "results": results,
            "nearest_hint": nearest_turn,
        }

    def _update_destination(self, destination: str) -> dict:
        loc = _locate(destination)
        if loc is None:
            return {"status": "error", "error": "not_found", "query": destination}
        name, x, y = loc
        previous = self.active_destination
        self.active_destination = name
        self.mutations.append({"op": "update_destination", "from": previous, "to": name})
        leg = _leg(x, y)
        meta = self._route_meta(name, leg["distance_km"])
        result = {
            "status": "success",
            "destination": name,
            "previous_destination": previous,
            "rerouted": previous is not None,
            **leg,
            **meta,
        }
        if previous is not None:
            result["correction_note"] = f"Cancelled route to {previous}. New route confirmed."
        return result


# ------------------------------------------------------------------------------
# TRAX dispatcher specialised for the in-car tools
# ------------------------------------------------------------------------------

class InCarDispatcher(ToolDispatcher):
    """
    ToolDispatcher + three domain guarantees the base class does not give for
    COMPENSABLE tools:

    1. Last-writer-wins: a staged update_destination is superseded by a later one
       in the same turn, so the stale destination is never executed.
    2. Route-restore compensation: rolling back a committed update_destination
       restores the previous route (never clears it if the call had failed).
    3. No orphaned futures: reset_for_new_turn resolves pending calls, so a model
       coroutine awaiting dispatch() can never hang.
    """

    def __init__(
        self,
        nav: NavRegistry,
        saga: SagaCoordinator,
        tool_log_path: str = "/tmp/incar_tool_calls.log",
        room_name: str = "incar",
        correction_epoch_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(
            registry_fn=nav.call,
            saga=saga,
            tool_log_path=tool_log_path,
            room_name=room_name,
            correction_epoch_fn=correction_epoch_fn,
        )
        self._nav = nav
        # Awaited before a destination change commits (see InCarBargeIn.on_new_command).
        self.pre_commit_hook: Optional[Callable[[], Awaitable[None]]] = None

    async def _handle_mutating(self, call: ToolCall) -> None:
        if call.name in _LAST_WRITER_WINS:
            for prior in [c for c in self._staged_queue if c.name == call.name]:
                self._staged_queue.remove(prior)
                if prior._resolution_future is not None and not prior._resolution_future.done():
                    log.info("incar: %s%s superseded before commit by %s", prior.name, prior.args, call.args)
                    prior._resolution_future.set_result(_SUPERSEDED)
        await super()._handle_mutating(call)

    async def _commit_mutating(self, call: ToolCall) -> None:
        if call.name in _LAST_WRITER_WINS and self.pre_commit_hook is not None:
            await self.pre_commit_hook()
        await super()._commit_mutating(call)

    def _build_compensation(self, call: ToolCall, result: dict) -> CompensationEntry:
        if call.name != "update_destination":
            return super()._build_compensation(call, result)

        succeeded = isinstance(result, dict) and result.get("status") == "success"
        previous = result.get("previous_destination") if succeeded else None

        async def compensate() -> None:
            if not succeeded:
                return  # nothing changed on the vehicle, nothing to undo
            await asyncio.get_running_loop().run_in_executor(None, self._nav.restore_destination, previous)
            log.info("saga compensation: destination restored to %r", previous)
            self._audit({"compensation": {"action": "restore_destination", "restored_to": previous}})

        return CompensationEntry(
            call_id=call.call_id,
            tool_name=call.name,
            args=call.args,
            committed_at=call.committed_at or time.monotonic(),
            compensation_fn=compensate,
        )

    def reset_for_new_turn(self) -> None:
        for call in [*self._provisional_cache.values(), *self._staged_queue]:
            fut = call._resolution_future
            if fut is not None and not fut.done():
                fut.set_result(_DISCARDED)
        super().reset_for_new_turn()

    def _audit(self, record: dict) -> None:
        self._write_audit(record)


# ------------------------------------------------------------------------------
# Intent-aware barge-in
# ------------------------------------------------------------------------------

_CANCEL_RE = re.compile(r"\b(stop|cancel|go back|not that|undo|revert|the other one)\b", re.IGNORECASE)
_ACK_RE = re.compile(r"\bno (problem|worries)\b", re.IGNORECASE)


def is_revision(text: str) -> bool:
    """True if the driver is correcting/cancelling (Levelt editing terms + cancel words), not just acknowledging."""
    text = _ACK_RE.sub("", text)
    return bool(_CORRECTION_TERMS_RE.search(text) or _CANCEL_RE.search(text))


class LocalAudioBargeDetector:
    """Calculates RMS audio energy to distinguish silence from speech frames."""

    @staticmethod
    def rms(samples: Any) -> float:
        if not samples:
            return 0.0
        return math.sqrt(sum(float(x) ** 2 for x in samples) / len(samples))


class InCarBargeIn(BargeInController):
    """
    BargeInController that decides *after hearing the driver* whether to roll back.

    Base behaviour rolls back every committed mutation on any barge-in, but VAD fires
    before any words exist: "thanks" and "no wait, cancel that" look identical at onset.
    Here onset still flushes audio and aborts uncommitted work immediately; the
    committed route is rolled back only once the speech turns out to be a revision
    (or the driver issues a new destination command). Acknowledgements keep the route.
    """

    def __init__(self, session, trp_gate, dispatcher, saga) -> None:
        super().__init__(session=session, trp_gate=trp_gate, dispatcher=dispatcher, saga=saga)
        self.awaiting_intent = False

    async def _execute_barge_in(self) -> None:
        try:
            res = self._session.interrupt(force=True)
            if asyncio.isfuture(res) or inspect.isawaitable(res):
                await res
        except Exception as exc:
            log.warning("incar barge_in: session.interrupt() failed: %s", exc)
        await self._gate.force_abort()
        self.awaiting_intent = self._saga.has_committed_mutations
        self._gate.reset_for_new_turn()
        self._dispatcher.reset_for_new_turn()
        if not self.awaiting_intent:
            self._saga.clear_for_new_turn()
        self._agent_speaking = False
        self._handling_barge_in = False

    async def on_interrupting_speech(self, transcript: str, is_final: bool = True) -> None:
        """Feed transcripts after a barge-in; the first revision (or final non-revision) decides."""
        if not self.awaiting_intent or not transcript.strip():
            return
        if is_revision(transcript):
            await self._resolve(revise=True)
        elif is_final:
            await self._resolve(revise=False)

    async def on_new_command(self) -> None:
        """A new destination command supersedes the interrupted reroute: roll it back first."""
        await self._resolve(revise=True)

    def drop_pending(self) -> None:
        """Turn boundary without a decision (e.g. a cough): keep the route."""
        self.awaiting_intent = False

    async def _resolve(self, revise: bool) -> None:
        if not self.awaiting_intent:
            return
        self.awaiting_intent = False
        if revise:
            await self._saga.compensate_all()
        self._saga.clear_for_new_turn()


# ------------------------------------------------------------------------------
# Real silence durations for the TRP gate
# ------------------------------------------------------------------------------

class SilenceTicker:
    """
    Reports growing silence (300 ms, then 900 ms) to the gate after the driver stops,
    cancelled the moment they speak again. LISTENING confirms at 300 ms; REPAIRING
    holds until 900 ms — the dynamic threshold TRPGate was designed around.
    (Reporting a fixed 1200 ms at speech end would satisfy the REPAIRING threshold
    instantly and defeat the hold.)
    """

    def __init__(self, gate: TRPGate, marks_ms: tuple[int, ...] = (VAD_SILENCE_NORMAL_MS, VAD_SILENCE_REPAIRING_MS),
                 time_scale: float = 1.0):
        self._gate = gate
        self._marks = marks_ms
        self._scale = time_scale  # <1 compresses waiting (tests); reported marks stay nominal
        self._task: Optional[asyncio.Task] = None

    def on_user_stopped(self) -> None:
        self.on_user_started()
        self._task = asyncio.get_running_loop().create_task(self._run())

    def on_user_started(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _run(self) -> None:
        elapsed = 0
        for mark in self._marks:
            await asyncio.sleep((mark - elapsed) / 1000.0 * self._scale)
            elapsed = mark
            await self._gate.on_silence_detected(mark)


# ------------------------------------------------------------------------------
# Stack assembly (shared by live agent, demo and tests)
# ------------------------------------------------------------------------------

class FakeSession:
    """Stand-in for AgentSession in offline runs; counts audio flushes."""

    def __init__(self) -> None:
        self.interrupted = 0

    async def interrupt(self, force: bool = False, **kwargs: Any) -> None:
        self.interrupted += 1


@dataclass
class Stack:
    nav: NavRegistry
    saga: SagaCoordinator
    dispatcher: InCarDispatcher
    gate: TRPGate
    barge: Optional[InCarBargeIn] = None
    session: Any = None

    def bind_session(self, session: Any) -> None:
        """Attach the (real or fake) session; the barge-in controller needs it, the session needs the tools."""
        self.session = session
        self.barge = InCarBargeIn(session=session, trp_gate=self.gate, dispatcher=self.dispatcher, saga=self.saga)
        self.dispatcher.pre_commit_hook = self.barge.on_new_command

    def new_turn(self) -> None:
        self.gate.reset_for_new_turn()
        self.dispatcher.reset_for_new_turn()
        self.saga.clear_for_new_turn()
        if self.barge is not None:
            self.barge.drop_pending()

    def model_call(self, name: str, **args: Any) -> "asyncio.Task[str]":
        """What the realtime model does when it emits a tool call."""
        return asyncio.create_task(
            self.dispatcher.dispatch(ToolCall(name=name, args=args, effect_class=EFFECT_MAP[name]))
        )


def build_stack(session: Any = None, latency_s: float = 0.15, log_path: Optional[str] = None,
                room_name: str = "incar") -> Stack:
    nav = NavRegistry(latency_s=latency_s)
    saga = SagaCoordinator()
    gate = TRPGate()
    dispatcher = InCarDispatcher(
        nav, saga,
        tool_log_path=log_path or os.path.join(tempfile.gettempdir(), "incar_tool_calls.log"),
        room_name=room_name,
        correction_epoch_fn=lambda: gate.correction_epoch,
    )
    gate.add_state_listener(dispatcher.on_trp_state_change)
    stack = Stack(nav=nav, saga=saga, dispatcher=dispatcher, gate=gate)
    stack.bind_session(session if session is not None else FakeSession())
    return stack


# ------------------------------------------------------------------------------
# LiveKit voice agent
# ------------------------------------------------------------------------------

_INSTRUCTIONS = (
    "You are TRAX Nav, an intelligent in-car voice assistant. "
    "You help the driver navigate, find places, and preview routes — all hands-free. "
    "Keep responses natural, warm, and conversational (2–3 sentences). "
    "RULES:\n"
    "1. NAVIGATION: When the driver wants to go somewhere or change route, call update_destination. "
    "   Confirm the destination, mention the ETA and the via road. Be friendly.\n"
    "2. PREVIEW: For travel time or route checks, call get_route. Report distance, ETA and traffic.\n"
    "3. AMENITIES: For gas, EV charging, food, or parking, call find_nearby. "
    "   Name the nearest option and its distance.\n"
    "4. CORRECTIONS: If the driver self-corrects mid-sentence (e.g. 'take me to the mall... "
    "   no wait, the airport'), ONLY use the final corrected destination. Acknowledge the change warmly.\n"
    "5. INTERRUPTIONS: If your previous route was cancelled, confirm the rollback warmly "
    "   and ask where they'd like to go. Do NOT stay silent.\n"
    "6. TOOL ONLY: NEVER state ETAs or distances from memory. Always call the tool first.\n"
    "7. SAFETY: Never mention being an AI. Never read long lists. Keep it smooth and driver-safe.\n"
    "8. RESPONSE LENGTH: Whenever the driver says they need a longer response, asks for a longer response, "
    "or asks you to elaborate, reply with an expansive, detailed description of at least 20 to 30 words "
    "dynamically generated for the route or context without using a fixed template."
)


def wire_trax(session: Any, gate: TRPGate, dispatcher: InCarDispatcher,
               saga: SagaCoordinator, barge: InCarBargeIn) -> SilenceTicker:
    """Connect LiveKit session events to the TRAX layers."""
    ticker = SilenceTicker(gate)
    agent_speaking = {"now": False}

    def new_turn() -> None:
        gate.reset_for_new_turn()
        dispatcher.reset_for_new_turn()
        saga.clear_for_new_turn()
        barge.drop_pending()

    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        async def _feed() -> None:
            # Immediate transcript barge-in fallback if agent is speaking
            if agent_speaking["now"] or getattr(session, "agent_state", None) == "speaking":
                await barge.on_user_speech_started()
            await barge.on_interrupting_speech(ev.transcript, ev.is_final)  # no-op unless a barge-in awaits intent
            await gate.on_transcript_token(ev.transcript, ev.is_final)

        asyncio.create_task(_feed())

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        agent_speaking["now"] = ev.new_state == "speaking"
        barge.set_agent_speaking(agent_speaking["now"])
        if ev.new_state == "listening":
            new_turn()  # turn boundary: committed mutations are no longer rollback-eligible

    @session.on("user_state_changed")
    def _on_user_state(ev):
        if ev.new_state == "speaking":
            ticker.on_user_started()
            if agent_speaking["now"] or getattr(session, "agent_state", None) == "speaking":
                # Barge-in: the cascade aborts, compensates and resets by itself.
                # Resetting first would wipe the saga log and lose the rollback.
                asyncio.create_task(barge.on_user_speech_started())
            elif gate.state in (TRPState.TRP_CONFIRMED, TRPState.ABORTED):
                new_turn()  # previous utterance is closed; this is a fresh one
            # Otherwise the driver is continuing/repairing the SAME utterance:
            # do NOT reset, or staged calls would be dropped mid-correction.
        elif ev.new_state == "listening":
            ticker.on_user_stopped()

    # Instant VAD onset hook — fires on the *very first* audio frame the driver
    # produces, 300–500 ms before user_state_changed accumulates enough energy.
    # This is the primary barge-in trigger; user_state_changed is the fallback.
    try:
        @session.on("user_speech_started")
        def _on_user_speech_started():
            ticker.on_user_started()
            if agent_speaking["now"] or getattr(session, "agent_state", None) == "speaking":
                asyncio.create_task(barge.on_user_speech_started())
    except Exception:
        pass  # older livekit-agents versions may not emit this event

    return ticker


if _HAS_LIVEKIT:
    _tool = llm.function_tool if hasattr(llm, "function_tool") else llm.ai_callable

    class InCarFnc:
        """Tool declarations exposed to the realtime model; every call goes through TRAX."""

        def __init__(self, dispatcher: InCarDispatcher) -> None:
            self._dispatcher = dispatcher

        async def _run(self, name: str, args: dict) -> str:
            return await self._dispatcher.dispatch(
                ToolCall(name=name, args=args, effect_class=EFFECT_MAP[name])
            )

        @_tool(description="MANDATORY tool to preview the route, distance and ETA to a place. It does NOT start navigation. NEVER estimate travel time from memory. Pass the place exactly as the driver said it, e.g. 'downtown hotel', 'the airport'.")
        async def get_route(self, destination: str):
            """
            Args:
                destination: The place exactly as spoken, e.g. 'the airport'. Do not paraphrase or expand.
            """
            return await self._run("get_route", {"destination": destination})

        @_tool(description="MANDATORY tool to start navigation or change the active destination while driving. Call ONCE with the driver's FINAL destination only; if they correct themselves ('actually take me to the airport') use the corrected place. NEVER ask for confirmation. Pass the place exactly as spoken.")
        async def update_destination(self, destination: str):
            """
            Args:
                destination: The final place exactly as spoken, e.g. 'downtown hotel'. Do not paraphrase or expand.
            """
            return await self._run("update_destination", {"destination": destination})

        @_tool(description="MANDATORY tool to find places near the vehicle by category, e.g. 'gas station', 'EV charging', 'cafe', 'hospital'. NEVER answer from memory. Use the driver's exact category words.")
        async def find_nearby(self, category: str):
            """
            Args:
                category: The kind of place exactly as spoken, e.g. 'gas station'.
            """
            return await self._run("find_nearby", {"category": category})

    class InCarVoiceAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=_INSTRUCTIONS)

    server = AgentServer()

    @server.rtc_session()
    async def entrypoint(ctx: "agents.JobContext"):
        from livekit.agents import inference
        from agent.trax_agent import get_realtime_model  # single source of truth: pinned temperature=0.0

        stack = build_stack(
            session=None, room_name=ctx.room.name, log_path="/tmp/incar_tool_calls.log",
        )
        vad = inference.VAD(model="silero")
        session = AgentSession(
            llm=get_realtime_model(),
            vad=vad,
            tools=llm.find_function_tools(InCarFnc(stack.dispatcher)),
            min_endpointing_delay=0.8,
            max_endpointing_delay=3.0,
            allow_interruptions=True,
            # Lower AEC warmup to 0.0 s so the agent can be interrupted immediately
            aec_warmup_duration=0.0,
            # 250 ms is enough to detect real speech but not clicks/noise.
            min_interruption_duration=0.25,
        )
        stack.bind_session(session)
        wire_trax(session, stack.gate, stack.dispatcher, stack.saga, stack.barge)
        await session.start(room=ctx.room, agent=InCarVoiceAgent())


# ------------------------------------------------------------------------------
# Offline demo — real TRAX components, scripted driver + model, hard assertions
# ------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rich terminal dashboard helpers (optional dependency)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _Table
    from rich.panel import Panel as _Panel
    from rich.text import Text as _Text
    from rich.live import Live as _Live
    from rich import box as _box
    _HAS_RICH = True

    def _Console(**kwargs):
        kwargs.setdefault("legacy_windows", False)
        return _RichConsole(**kwargs)
except ImportError:
    _HAS_RICH = False


class _Dashboard:
    """
    Live vehicle dashboard. Falls back to plain-text if `rich` is not installed.
    Call `.update()` whenever state changes, then `.print_static()` at the end.
    """

    def __init__(self) -> None:
        self.active_dest: Optional[str] = None
        self.prev_dest: Optional[str] = None
        self.eta_min: Optional[int] = None
        self.dist_km: Optional[float] = None
        self.via_road: Optional[str] = None
        self.traffic: Optional[str] = None
        self.gate_state: str = "LISTENING"
        self.barge_events: list[str] = []
        self.ledger: list[tuple[str, str, str, str]] = []  # (op, from, to, status)
        self._console = _Console() if _HAS_RICH else None

    # ---- state setters ------------------------------------------------

    def nav_update(self, result: dict) -> None:
        self.prev_dest = self.active_dest
        self.active_dest = result.get("destination")
        self.eta_min = result.get("eta_min")
        self.dist_km = result.get("distance_km")
        self.via_road = result.get("via_road")
        self.traffic = result.get("traffic")

    def add_ledger(self, op: str, frm: Optional[str], to: str, status: str) -> None:
        self.ledger.append((op, frm or "None", to, status))

    def set_gate(self, state: str) -> None:
        self.gate_state = state

    def add_barge(self, msg: str) -> None:
        self.barge_events.append(msg)

    # ---- rendering ----------------------------------------------------

    def _render(self) -> None:
        if not _HAS_RICH or self._console is None:
            return
        self._console.print(self._build_panel())

    def _build_panel(self):
        lines = []
        dest_line = f"[bold green]{self.active_dest or '—'}[/bold green]"
        if self.eta_min:
            dest_line += f"  ({self.dist_km} km | {self.eta_min} min)"
        lines.append(f"[bold]ACTIVE DESTINATION:[/bold]   {dest_line}")
        if self.via_road:
            lines.append(f"[bold]ROUTE GUIDANCE:[/bold]       via {self.via_road} ({self.traffic})")
        if self.prev_dest:
            lines.append(f"[bold]SAGA RESTORE MEMORY:[/bold]  {self.prev_dest}")
        lines.append("")
        gate_color = "yellow" if "REPAIR" in self.gate_state else ("green" if "CONFIRM" in self.gate_state else "cyan")
        lines.append(f"[bold]TRAX TRP GATE:[/bold]       [{gate_color}]{self.gate_state}[/{gate_color}]")
        if self.barge_events:
            lines.append(f"[bold]BARGE-IN:[/bold]             [red]{self.barge_events[-1]}[/red]")

        if self.ledger:
            lines.append("")
            lines.append("[bold]TRANSACTION LEDGER:[/bold]")
            for i, (op, frm, to, st) in enumerate(self.ledger, 1):
                color = "green" if "COMMIT" in st else ("red" if "ROLLEDBACK" in st else "yellow")
                lines.append(f"  {i}. [cyan]{op:<22}[/cyan] {frm:<18} -> {to:<18} [{color}]{st}[/{color}]")

        body = "\n".join(lines)
        return _Panel(
            body,
            title="[bold blue]TRAX IN-CAR INTELLIGENT NAVIGATION SYSTEM[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )

    def print_static(self) -> None:
        if _HAS_RICH and self._console:
            self._console.print(self._build_panel())
        else:
            print("\n=== TRAX DASHBOARD ===")
            print(f"  ACTIVE DESTINATION : {self.active_dest}")
            print(f"  ETA                : {self.eta_min} min  ({self.dist_km} km)")
            print(f"  VIA                : {self.via_road}")
            print(f"  TRAFFIC            : {self.traffic}")
            print(f"  GATE               : {self.gate_state}")
            for ev in self.barge_events:
                print(f"  BARGE-IN           : {ev}")
            print("  LEDGER:")
            for i, (op, frm, to, st) in enumerate(self.ledger, 1):
                print(f"    {i}. {op:<22} {frm:<18} -> {to:<18}  [{st}]")


async def run_demo() -> bool:
    logging.getLogger("trax").setLevel(logging.WARNING)
    t0 = time.monotonic()
    dash = _Dashboard()

    def ts() -> str:
        return f"[{time.monotonic() - t0:5.2f}s]"

    def say(who: str, msg: str, color: str = "") -> None:
        if _HAS_RICH:
            _Console().print(f"{ts()} [bold {color or 'white'}]{who:<8}[/bold {color or 'white'}] {msg}")
        else:
            print(f"{ts()} {who:<8} {msg}")

    with tempfile.TemporaryDirectory() as tmp:
        s = build_stack(log_path=os.path.join(tmp, "incar.log"), room_name="incar-demo")

        async def _on_gate_state(state: TRPState) -> None:
            dash.set_gate(state.name)
            say("GATE", f"-> {state.name}", "yellow")

        s.gate.add_state_listener(_on_gate_state)

        def model_calls(name: str, **args: Any):
            say("MODEL", f"emits {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})", "magenta")
            return s.model_call(name, **args)

        async def agent_says(text: str, seconds: float = 0.4) -> None:
            s.barge.set_agent_speaking(True)
            say("AGENT", f'"{text}"', "green")
            await asyncio.sleep(seconds)

        async def agent_done() -> None:
            s.barge.set_agent_speaking(False)
            s.new_turn()

        # ── Print header ──────────────────────────────────────────────────────
        if _HAS_RICH:
            _Console().rule("[bold blue]TRAX In-Car Navigation — Demo Walkthrough[/bold blue]")
        else:
            print("\n" + "=" * 70)
            print("   TRAX In-Car Navigation — Demo Walkthrough")
            print("=" * 70)

        # ── Scenario 1: initial route ──────────────────────────────────────────
        say("=SYS=", "=== Scenario 1: Driver requests a route ===", "blue")
        say("DRIVER", '"Navigate to the downtown hotel."', "cyan")
        await s.gate.on_transcript_token("Navigate to the downtown hotel.", True)
        r = json.loads(await model_calls("update_destination", destination="downtown hotel"))
        dash.nav_update(r)
        dash.add_ledger("update_destination", None, r["destination"], "COMMITTED")
        say(
            "AGENT",
            f'"Routing you to {r["destination"]} via {r.get("via_road", "Main Road")}. '
            f'ETA is {r["eta_min"]} minutes with {r.get("traffic", "clear roads")}. '
            f'{r.get("next_maneuver", "")}"',
            "green",
        )
        await asyncio.sleep(0.5)
        await agent_done()
        dash.print_static()

        # ── Scenario 2 + 3: mid-sentence correction → airport committed → barge-in rollback ───
        # These are ONE continuous driver interaction: the driver first self-corrects
        # (Scenario 2), the model commits airport, the agent starts to speak, then
        # the driver barges in with a new destination (Scenario 3). The saga must
        # NOT be cleared between the commit and the barge-in, so agent_done() is
        # intentionally skipped here — the barge-in IS the end of the agent turn.
        say("=SYS=", "=== Scenario 2: Mid-sentence self-correction (last-writer-wins) ===", "blue")
        say("DRIVER", '"Take me to the city mall..."', "cyan")
        await s.gate.on_transcript_token("Take me to the city mall", False)
        await asyncio.sleep(0.2)
        stale = model_calls("update_destination", destination="city mall")
        await asyncio.sleep(0.3)
        say("DRIVER", '"...no wait, actually the airport."', "cyan")
        await s.gate.on_transcript_token("no wait, actually the airport", False)
        await asyncio.sleep(0.25)
        fresh = model_calls("update_destination", destination="airport")
        stale_out, fresh_out = await asyncio.gather(stale, fresh)
        stale_status = json.loads(stale_out)["status"]
        fresh_res = json.loads(fresh_out)
        after_correction = s.nav.active_destination
        dash.nav_update(fresh_res)
        dash.add_ledger("update_destination", "downtown hotel", "city mall", "SUPERSEDED \u2014 DROPPED")
        dash.add_ledger("update_destination", "downtown hotel", fresh_res["destination"], "COMMITTED")
        say("VEHICLE", f"active destination = {after_correction!r}  (stale call: {stale_status})", "white")

        # Agent starts speaking the airport confirmation — saga still has the
        # airport entry. The driver immediately barges in before we could call
        # agent_done() (which would clear the saga).
        say("=SYS=", "=== Scenario 3: Live barge-in \u2192 saga rollback (same agent turn) ===", "blue")
        await agent_says(  # arms barge-in; does NOT call agent_done()
            f'"Sure, I caught that \u2014 routing you to the airport instead. '
            f'Via {fresh_res.get("via_road", "Airport Expressway")}, '
            f'ETA {fresh_res["eta_min"]} minutes with {fresh_res.get("traffic", "light traffic")}. '
            f'{fresh_res.get("next_maneuver", "")}"',
            seconds=0.25,
        )
        say("DRIVER", '"Hold on \u2014 take me to central station instead."  [BARGING IN]', "red")
        dash.add_barge("BARGE-IN DETECTED \u2192 Outbound Audio Flushed \u2192 Saga Rollback Executing")
        await s.barge.on_user_speech_started()  # VAD onset: audio flushed, rollback deferred
        await s.barge.on_interrupting_speech("Hold on - take me to the central station instead")
        after_rollback = s.nav.active_destination
        dash.add_ledger("saga_rollback", fresh_res["destination"], after_rollback or "None", "ROLLED BACK VIA BARGE-IN")
        say("VEHICLE", f"active destination = {after_rollback!r}  (airport reroute rolled back)", "white")
        await s.gate.on_transcript_token("Hold on - take me to the central station instead", False)
        await asyncio.sleep(0.3)
        final_res = json.loads(await model_calls("update_destination", destination="central station"))
        dash.nav_update(final_res)
        dash.add_ledger("update_destination", after_rollback, final_res["destination"], "COMMITTED")
        await agent_says(  # agent speaks the central station confirmation
            f'"Got it \u2014 routing to Central Station via {final_res.get("via_road", "Central Boulevard")}. '
            f'ETA {final_res["eta_min"]} minutes, {final_res.get("traffic", "clear roads")}. '
            f'{final_res.get("next_maneuver", "")}"',
            seconds=0.25,
        )
        dash.print_static()

        # ── Scenario 4: acknowledgement barge-in \u2014 route kept \u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014
        # Agent is STILL speaking the central station confirmation (barge.set_agent_speaking
        # is True from agent_says above). Driver says "Okay, thanks" \u2014 this should NOT
        # roll back the route because it's an acknowledgement, not a revision.
        say("=SYS=", "=== Scenario 4: Acknowledgement barge-in \u2014 route kept ===", "blue")
        say("DRIVER", '"Okay, thanks."  [talks over confirmation]', "cyan")
        await s.barge.on_user_speech_started()  # agent is still speaking \u2192 barge-in fires
        await s.barge.on_interrupting_speech("Okay, thanks")
        after_ack = s.nav.active_destination
        say("VEHICLE", f"active destination = {after_ack!r}  (acknowledgement: route kept)", "white")
        await agent_done()
        dash.print_static()

        # ── Scenario 5: read-only query self-correction ────────────────────────
        say("=SYS=", "=== Scenario 5: Read-only query mid-sentence correction ===", "blue")
        say("DRIVER", '"Find a gas station near me..."', "cyan")
        await s.gate.on_transcript_token("Find a gas station near me", False)
        await asyncio.sleep(0.2)
        stale_q = model_calls("find_nearby", category="gas station")
        await asyncio.sleep(0.3)
        say("DRIVER", '"...sorry, I mean EV charging."', "cyan")
        await s.gate.on_transcript_token("sorry, I mean EV charging", False)
        await asyncio.sleep(0.25)
        fresh_q = model_calls("find_nearby", category="EV charging")
        stale_q_out, fresh_q_out = await asyncio.gather(stale_q, fresh_q)
        stale_q_status = json.loads(stale_q_out)["status"]
        fresh_q_res = json.loads(fresh_q_out)
        nearest = fresh_q_res["results"][0]
        say(
            "AGENT",
            f'"Sure! The nearest EV charging station is {nearest["name"]}, '
            f'about {nearest["distance_km"]} km away. {fresh_q_res.get("nearest_hint", "")}"',
            "green",
        )
        dash.print_static()

        # ── Verdict ────────────────────────────────────────────────────────────
        history = [m["to"] for m in s.nav.mutations]
        checks = {
            "stale 'city mall' call superseded, never reached vehicle":
                stale_status == "superseded" and "city mall" not in history,
            "corrected destination 'airport' committed": after_correction == "airport",
            "both barge-ins flushed agent audio": s.session.interrupted == 2,
            "barge-in rolled back reroute to previous destination": after_rollback == "downtown hotel",
            "post-barge command applied": s.nav.active_destination == "central station",
            "acknowledgement barge-in kept the route (no rollback)": after_ack == "central station",
            "vehicle mutation history is exactly the intended one":
                history == ["downtown hotel", "airport", "downtown hotel", "central station"],
            "stale read-only query superseded, corrected one returned":
                stale_q_status == "superseded" and fresh_q_res.get("category") == "ev_charging",
        }

    if _HAS_RICH:
        console = _Console()
        console.rule("[bold]Vehicle Mutation History (Ground Truth)[/bold]")
        t = _Table(box=_box.SIMPLE)
        t.add_column("#", style="dim")
        t.add_column("Operation", style="cyan")
        t.add_column("From", style="yellow")
        t.add_column("To", style="green")
        for i, m in enumerate(s.nav.mutations, 1):
            t.add_row(str(i), m["op"], str(m["from"]), str(m["to"]))
        console.print(t)
        console.rule("[bold]TRAX Verification Checks[/bold]")
        for name, ok in checks.items():
            color = "green" if ok else "red"
            icon = "✓" if ok else "✗"
            console.print(f"  [{color}]{icon}[/{color}] {name}")
        ok_all = all(checks.values())
        result_color = "green" if ok_all else "red"
        console.print(f"\n[bold {result_color}]RESULT: {'ALL CHECKS PASSED ✓' if ok_all else 'CHECKS FAILED ✗'}[/bold {result_color}]")
    else:
        print("\n=== Vehicle mutation history (ground truth) ===")
        for m in s.nav.mutations:
            print(f"  {m['op']:<18} {m['from']!s:<16} -> {m['to']}")
        print("\n=== Checks ===")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        ok_all = all(checks.values())
        print(f"\nRESULT: {'ALL CHECKS PASSED' if ok_all else 'FAILED'}")

    return ok_all


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if "--demo" in sys.argv:
        sys.exit(0 if asyncio.run(run_demo()) else 1)
    if not _HAS_LIVEKIT:
        sys.exit("livekit-agents is not installed: `pip install livekit-agents livekit-plugins-google` "
                 "or run the offline demo with --demo")
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()

