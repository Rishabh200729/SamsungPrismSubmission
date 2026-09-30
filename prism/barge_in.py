"""
prism/barge_in.py
ADR 04 — AEC Lockout Elimination & Instant Barge-In Eviction

Detects user audio while the agent is speaking and immediately:
  1. Calls session.interrupt() — instant outbound audio flush,
     bypassing the AEC warmup lockout (the 3.00s window seen in empirical traces).
  2. Forces TRPGate → ABORTED.
  3. Triggers ToolDispatcher.on_trp_state_change(ABORTED) — purge provisional cache.
  4. Triggers SagaCoordinator.compensate_all() — reverse any committed mutations.

Literature basis:
  - Lin et al. (FDB-v3 2026) §6.2: AEC lockout is the primary cause of agent
    speaking over user during corrections (empirically seen at 05:08:24 in travel_10).
  - Défossez et al. (Moshi 2024): native full-duplex models handle barge-in natively
    at the audio codec layer; our approach achieves the same for cascade architectures.

Note on LiveKit API:
  AgentSession.interrupt() exists in livekit-agents ≥1.0 and immediately stops
  the current agent speech response.  We use session.on("user_speech_started") as
  the barge-in trigger — it fires the moment the VAD detects user voice energy,
  which is earlier and more reliable than RMS-thresholded raw audio frame processing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

from prism import TRPState
from prism.trp_gate import TRPGate
from prism.tool_dispatcher import ToolDispatcher
from prism.saga_coordinator import SagaCoordinator

log = logging.getLogger("trax.barge_in")


class BargeInController:
    """
    Detects user speech while agent is speaking and executes the full abort cascade.

    Wire-up in lk_agent_tool.py::

        barge = BargeInController(
            session=session, trp_gate=gate,
            dispatcher=dispatcher, saga=saga,
        )

        @session.on("agent_state_changed")
        def on_agent_state(ev):
            barge.set_agent_speaking(ev.new_state == "speaking")

        @session.on("user_speech_started")
        def on_user_speech():
            asyncio.create_task(barge.on_user_speech_started())
    """

    def __init__(
        self,
        session,                   # livekit.agents.AgentSession — typed as Any to avoid LK import
        trp_gate:   TRPGate,
        dispatcher: ToolDispatcher,
        saga:       SagaCoordinator,
    ) -> None:
        self._session    = session
        self._gate       = trp_gate
        self._dispatcher = dispatcher
        self._saga       = saga

        # Armed when agent is speaking; disarmed otherwise.
        self._agent_speaking: bool = False

        # Guard: prevent multiple concurrent barge-in handlers from firing
        self._handling_barge_in: bool = False

    # -----------------------------------------------------------------------
    # Arming
    # -----------------------------------------------------------------------

    def set_agent_speaking(self, is_speaking: bool) -> None:
        """
        Called from lk_agent_tool.py's on_agent_state_changed handler.
        Arms or disarms the barge-in detector.
        """
        if self._agent_speaking != is_speaking:
            log.info(
                "barge_in: agent speech %s",
                "started" if is_speaking else "ended",
            )
        self._agent_speaking = is_speaking

        # ``session.interrupt()`` commonly causes this state transition before
        # the abort cascade has finished.  The cascade itself owns the guard and
        # clears it only after gate, dispatcher and saga state are consistent.

    @property
    def is_handling_barge_in(self) -> bool:
        """True while the abort cascade owns gate/dispatcher/saga state."""
        return self._handling_barge_in

    @property
    def is_agent_speaking(self) -> bool:
        """True while an agent response is eligible for barge-in."""
        return self._agent_speaking or getattr(self._session, "agent_state", None) == "speaking"

    # -----------------------------------------------------------------------
    # Barge-in detection
    # -----------------------------------------------------------------------

    async def on_user_speech_started(self) -> bool:
        """
        Called when LiveKit's VAD detects user speech onset.
        If the agent is currently speaking, this is a barge-in — execute cascade.

        This event fires at VAD onset, which is earlier than the AEC lockout window
        would allow in the unmodified reference agent.  By calling session.interrupt()
        here we bypass the 3.00s AEC warmup delay seen in the empirical traces.
        """
        is_speaking = self.is_agent_speaking
        log.info(
            "barge_in: user speech onset (agent_speaking=%s, handling=%s)",
            is_speaking,
            self._handling_barge_in,
        )
        if not is_speaking:
            return False   # Normal: user speaking during user-turn, not a barge-in

        if self._handling_barge_in:
            return True   # Guard: this onset is already being handled as a barge-in

        self._handling_barge_in = True
        log.info("barge_in: DETECTED — user spoke while agent speaking, executing abort cascade")

        await self._execute_barge_in()
        return True

    # -----------------------------------------------------------------------
    # Abort cascade
    # -----------------------------------------------------------------------

    async def _execute_barge_in(self) -> None:
        """
        Full abort cascade, executed in order:
        1. Flush agent audio (session.interrupt())
        2. Force TRP gate → ABORTED (propagates to dispatcher via listener)
        3. Saga compensation if any mutations were committed this turn
        4. Reset all components for the fresh user turn

        The cascade is sequential to ensure the audit trail is consistent:
        interrupt first (stops audio), then purge (cleans state), then compensate.
        """
        # ── Step 1: Instant audio flush ────────────────────────────────────
        # session.interrupt() immediately stops outbound TTS playback.
        # This eliminates the AEC warmup lockout — the agent no longer speaks
        # over the user's correction (empirical failure at 05:08:24 in travel_10).
        try:
            # force=True bypasses the AEC warmup lockout (default 3.0s window
            # that silently ignores interruptions right after agent starts speaking).
            res = self._session.interrupt(force=True)
            if asyncio.isfuture(res) or inspect.isawaitable(res):
                await res
            log.info("barge_in: session.interrupt(force=True) called — audio flushed")
        except Exception as exc:
            log.warning("barge_in: session.interrupt() failed: %s — continuing cascade", exc)

        # ── Step 2: Force TRP gate → ABORTED ──────────────────────────────
        # gate.force_abort() calls _transition(ABORTED) which fires all listeners
        # including dispatcher.on_trp_state_change(ABORTED) — purges cache.
        await self._gate.force_abort()

        # ── Step 3: Saga compensation ──────────────────────────────────────
        # If any COMPENSABLE mutations were committed before the barge-in,
        # execute their inverse operations in reverse order.
        if self._saga.has_committed_mutations:
            log.info("barge_in: triggering saga compensation for committed mutations")
            compensated = await self._saga.compensate_all()
            log.info("barge_in: compensated %d mutation(s): %s", len(compensated), compensated)

        # ── Step 4: Reset for fresh turn ──────────────────────────────────
        self._gate.reset_for_new_turn()
        self._dispatcher.reset_for_new_turn()
        self._saga.clear_for_new_turn()
        self._agent_speaking    = False
        self._handling_barge_in = False

        log.info("barge_in: cascade complete — all components reset for fresh user turn")
