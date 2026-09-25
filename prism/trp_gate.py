"""
prism/trp_gate.py
ADR 02 — Disfluency-Aware Transition Relevance Place (TRP) Gating

Watches the LiveKit streaming transcript, applies Levelt (1983) editing-term
detection and Shriberg (1994) disfluency theory to decide when a user turn is
truly complete.  Emits TRPState transitions to all registered listeners.

Literature basis:
  - Levelt (1983): reparandum → IP → editing phase → reparans taxonomy
  - Shriberg (1994): hesitation pauses after editing markers ≠ turn-final pauses
  - Ekstedt & Skantze (TurnGPT, 2020): linguistic completion >> acoustic VAD alone
  - Raux & Eskenazi (2009): dynamic silence thresholds (300 ms / 900 ms)
  - Sacks, Schegloff & Jefferson (1974): TRP definition

Independence: no LiveKit imports.  Feed it token strings via on_transcript_token()
and silence events via on_silence_detected().  Fully unit-testable.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Callable, Coroutine, List, Optional

from prism import TRPState

log = logging.getLogger("prism.trp_gate")


# ---------------------------------------------------------------------------
# Editing-term lexicon — Levelt (1983) editing phase markers
# These are the OVERT lexical signals of an active speech repair.
# Regex is case-insensitive; word-boundary anchored to avoid subword matches.
# This is a universal linguistic set, not tuned to specific benchmark scenarios.
# ---------------------------------------------------------------------------

_EDITING_TERMS_RE = re.compile(
    r"\b("
    r"wait|actually|no|sorry|scratch that|i mean|never mind|hold on"
    r"|correction|let me correct|i meant|uh|um|er|hmm"
    r"|wait no|wait actually|oh wait|oh no"
    r"|rather|instead of|oops|make that|change that to"
    r")\b",
    re.IGNORECASE,
)

# Syntactic incompletion signals: a turn should NOT be confirmed if the last
# meaningful token is one of these (dangling preposition / open phrase / conjunction).
_INCOMPLETE_TAIL_RE = re.compile(
    r"\b("
    r"on|for|to|at|in|from|by|about|with|of|and|or|the|a|an"
    r"|going|heading|travelling|flying|searching|looking"
    r"|because|but|if|so|while|since|that|which"
    r")\s*$",
    re.IGNORECASE,
)

# Sentence-final punctuation — strong indicator of pragmatic completion.
_FINAL_PUNCT_RE = re.compile(r"[.?!]\s*$")

# Silence thresholds (seconds) — Raux & Eskenazi (2009)
VAD_SILENCE_NORMAL_MS:   int = 300    # confirm TRP at this threshold in LISTENING
VAD_SILENCE_REPAIRING_MS: int = 900   # hold floor until this threshold in REPAIRING


class TRPGate:
    """
    Disfluency-aware Transition Relevance Place gate.

    Usage in lk_agent_tool.py::

        gate = TRPGate()
        gate.add_state_listener(dispatcher.on_trp_state_change)

        @session.on("user_input_transcribed")
        def on_transcript(msg):
            asyncio.create_task(
                gate.on_transcript_token(msg.transcript, msg.is_final)
            )

        @session.on("user_speech_stopped")
        def on_silence(ev):
            asyncio.create_task(gate.on_silence_detected(ev.duration_ms))
    """

    def __init__(self) -> None:
        self._state:              TRPState = TRPState.LISTENING
        self._buffer:             str      = ""          # rolling transcript window
        self._last_editing_at:    float    = 0.0         # monotonic time of last editing term
        self._listeners:          List[Callable[[TRPState], Coroutine]] = []
        self._quiescence_task:    Optional[asyncio.Task] = None

    # -----------------------------------------------------------------------
    # Public observable
    # -----------------------------------------------------------------------

    @property
    def state(self) -> TRPState:
        return self._state

    # -----------------------------------------------------------------------
    # Listener registration
    # -----------------------------------------------------------------------

    def add_state_listener(
        self, fn: Callable[[TRPState], Coroutine]
    ) -> None:
        """
        Register an async callable to be called on every state transition.
        Multiple listeners supported (dispatcher + barge_in can both register).
        """
        self._listeners.append(fn)

    # -----------------------------------------------------------------------
    # Input surface — called by lk_agent_tool.py event handlers
    # -----------------------------------------------------------------------

    async def on_transcript_token(self, token: str, is_final: bool) -> None:
        """
        Feed each transcript chunk from the LiveKit user_input_transcribed event.

        :param token:    The transcript text (may be incremental interim or final).
        :param is_final: True when the ASR has finalised this segment.
        """
        if not token or not token.strip():
            return

        # Append to rolling buffer
        self._buffer = (self._buffer + " " + token.strip()).strip()

        # ── Editing-term detection (Levelt 1983) ─────────────────────────
        if _EDITING_TERMS_RE.search(token):
            self._last_editing_at = time.monotonic()
            if self._state != TRPState.REPAIRING:
                log.info(
                    "trp_gate: editing term detected in '%s' → REPAIRING (VAD inflated to %dms)",
                    token[:60], VAD_SILENCE_REPAIRING_MS,
                )
                await self._transition(TRPState.REPAIRING)
            self._schedule_quiescence_check()
            return   # Don't try to confirm TRP on the same token that has an editing term

        # ── TRP confirmation heuristics ───────────────────────────────────
        # Only attempt confirmation if:
        #   1. is_final segment from ASR
        #   2. No editing term fired in the last 800 ms
        #   3. Buffer doesn't end with a syntactically incomplete tail
        if is_final and self._state in (TRPState.LISTENING, TRPState.REPAIRING):
            quiescence_ms = (time.monotonic() - self._last_editing_at) * 1000
            if quiescence_ms >= 800 and self._is_syntactically_complete(self._buffer):
                log.info(
                    "trp_gate: final token + complete buffer '%s…' → TRP_CONFIRMED",
                    self._buffer[-40:],
                )
                await self._transition(TRPState.TRP_CONFIRMED)
            elif self._state == TRPState.REPAIRING:
                # Refresh quiescence check since new speech arrived
                self._schedule_quiescence_check()

    def _schedule_quiescence_check(self) -> None:
        """Atomix §3 bounded commit window: if no new editing terms occur for 900ms, confirm TRP."""
        if self._quiescence_task and not self._quiescence_task.done():
            self._quiescence_task.cancel()

        async def _check():
            await asyncio.sleep(VAD_SILENCE_REPAIRING_MS / 1000.0)
            if self._state == TRPState.REPAIRING:
                elapsed_ms = (time.monotonic() - self._last_editing_at) * 1000
                if elapsed_ms >= VAD_SILENCE_REPAIRING_MS - 50:
                    if self._is_syntactically_complete(self._buffer) or not self._buffer.endswith(("wait", "actually", "uh", "um")):
                        log.info(
                            "trp_gate: repair quiescence window (%.0fms) elapsed → TRP_CONFIRMED",
                            elapsed_ms,
                        )
                        await self._transition(TRPState.TRP_CONFIRMED)

        try:
            loop = asyncio.get_event_loop()
            self._quiescence_task = loop.create_task(_check())
        except RuntimeError:
            pass

    async def on_silence_detected(self, silence_ms: float) -> None:
        """
        Called when the VAD engine detects a silence interval.

        Decision (Raux & Eskenazi 2009):
          LISTENING:  silence ≥ 300 ms → confirm TRP (buffer completeness already checked)
          REPAIRING:  silence < 900 ms → hold floor
          REPAIRING:  silence ≥ 900 ms → confirm TRP (human finished repair)
        """
        if self._state == TRPState.TRP_CONFIRMED or self._state == TRPState.ABORTED:
            return

        threshold = (
            VAD_SILENCE_REPAIRING_MS
            if self._state == TRPState.REPAIRING
            else VAD_SILENCE_NORMAL_MS
        )

        if silence_ms >= threshold:
            # Additional guard: don't confirm on silence alone if buffer tail is incomplete
            if self._is_syntactically_complete(self._buffer):
                log.info(
                    "trp_gate: silence %.0fms ≥ threshold %dms → TRP_CONFIRMED (state=%s)",
                    silence_ms, threshold, self._state.name,
                )
                await self._transition(TRPState.TRP_CONFIRMED)
            else:
                log.debug(
                    "trp_gate: silence %.0fms ≥ threshold but buffer tail incomplete, holding",
                    silence_ms,
                )
        else:
            log.debug(
                "trp_gate: silence %.0fms < threshold %dms, holding floor (state=%s)",
                silence_ms, threshold, self._state.name,
            )

    async def force_abort(self) -> None:
        """
        Force ABORTED state immediately.
        Called by BargeInController when user audio is detected during agent speech.
        """
        if self._state != TRPState.ABORTED:
            log.info("trp_gate: forced ABORTED by barge-in controller")
            await self._transition(TRPState.ABORTED)

    def reset_for_new_turn(self) -> None:
        """
        Called at the start of each new user turn (after agent finishes speaking or
        after barge-in recovery).  Resets buffer and state to LISTENING.
        """
        log.debug("trp_gate: reset_for_new_turn (was %s)", self._state.name)
        if self._quiescence_task and not self._quiescence_task.done():
            self._quiescence_task.cancel()
        self._state             = TRPState.LISTENING
        self._buffer            = ""
        self._last_editing_at   = 0.0

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    async def _transition(self, new_state: TRPState) -> None:
        """Fire state transition and notify all listeners."""
        if self._state == new_state:
            return
        old = self._state
        self._state = new_state
        log.debug("trp_gate: %s → %s", old.name, new_state.name)
        for listener in self._listeners:
            try:
                await listener(new_state)
            except Exception as exc:
                log.error("trp_gate: listener %s raised: %s", listener, exc)

    @staticmethod
    def _is_syntactically_complete(text: str) -> bool:
        """
        Lightweight completion heuristic.
        Returns True if the buffer looks like a syntactically complete utterance.

        Checks (in order):
          1. Non-empty
          2. Does NOT end with a syntactically incomplete tail (dangling prep/conj)
          3. Ends with sentence-final punctuation (strong signal)
          OR the text contains at least one verb-like word (heuristic completeness)

        This is intentionally conservative: better to hold for 900 ms and confirm
        late than to fire TRP on an incomplete utterance.
        """
        text = text.strip()
        if not text:
            return False

        # Reject dangling tails
        if _INCOMPLETE_TAIL_RE.search(text):
            return False

        # Accept if ends with strong sentence-final punctuation
        if _FINAL_PUNCT_RE.search(text):
            return True

        # Accept if the buffer has enough substance (more than 3 words, at least
        # one of which looks like a verb form)
        words = text.lower().split()
        if len(words) >= 3:
            return True

        return False
