"""
module1/fast_path/router.py

FastPathRouter — Module 1 entry point.

OWNERSHIP
---------
Module 1 owns (this file):
- Handler registration for TEXT_CHUNK, END_OF_TURN, INTERRUPTION
- Fast-path event classification and routing
- Filler / silence / ack policy decisions
- Provisional hypothesis lifecycle (via HypothesisBuffer)
- Synthetic competitive INTERRUPTION submission for TEXT_CHUNK corrections
- Fast-path timing telemetry

Module 1 DOES NOT own (all delegated to Module 3):
- Physical task cancellation
- Generation increment / invalidation
- State machine invalidation (state_store.invalidate)
- Call ledger supersession
- Stale-result fencing (output gate)
- Floor transition legality enforcement

RACE AUDIT — INTERRUPTION GENERATION RACE
------------------------------------------
The question: Can Module 1's filler be rejected by the output gate if
Runtime._handle_interruption concurrently advances the session generation?

Short answer: Yes, this race exists, but it is safe-by-design and handled.

Long answer:

asyncio.gather(Runtime._handle_interruption, Module1._on_interruption) runs
both coroutines concurrently from the same event-loop tick. However,
asyncio is cooperative: coroutines only interleave at `await` points.

The critical path in _handle_interruption:
  1. ctx.metadata["_interrupt_guard"] = True          (no suspension)
  2. await request_cancellation(...)
       2a. for record in active_tasks():
               await cancel_task(...)                 ← SUSPEND POINT
       2b. old_gen, new_gen = increment_generation()  (no suspension after tasks)
  3. state_store.invalidate(generation=new_gen)        (no suspension)
  4. for record in call_ledger.supersede_before(...):
         await emit_output(make_cancellation(...))     ← SUSPEND POINT

The critical path in Module1._on_interruption:
  1. Read event.payload["competitive"]                 (no suspension)
  2. Classify interruption type                        (no suspension)
  3. Dispose hypothesis                                (no suspension)
  4. Read current_gen = runtime.get_current_generation()  (no suspension)
  5. If filler allowed: await runtime.emit_output(filler, priority=10)  ← SUSPEND

The race window:
- If Module1._on_interruption reaches step 4 BEFORE _handle_interruption
  completes step 2b (increment_generation), Module 1 reads generation=N
  and stamps the filler with generation=N.
- _handle_interruption then advances to generation=N+1.
- When Module 1's filler reaches emit_output → _authorize_output,
  event.generation=N but ctx.current_generation=N+1 → REJECTED as stale.

Resolution — two-layer defence:

LAYER 1 (primary): Emit the filler WITHOUT a generation stamp.
  The output gate check is:
    if event.generation is not None and event.generation != current: raise
  A filler with generation=None passes this check unconditionally.
  This is correct: a filler is an ACKNOWLEDGEMENT, not a response to a
  specific task result. It carries no semantic dependency on the generation.
  The output gate's generation check is designed to prevent stale *task
  results* from being published — not acknowledgement speech acts.

LAYER 2 (fallback): If caller explicitly stamps with get_current_generation(),
  a stale-stamp rejection is a silent quality loss (missed filler), not a
  correctness failure. The CANCELLATION events emitted by _handle_interruption
  at priority=100 provide implicit acknowledgement. The conversation continues
  correctly regardless.

CONCLUSION: Module 1's filler/ack emissions MUST use generation=None
(or simply omit the generation kwarg on make_filler). They must NOT try to
read-then-stamp the generation before emit_output, because the stamp read
and the emit are not atomic — the race window between them is real.

The test `test_interruption_ack_not_lost_on_generation_race` in
module1/tests/test_router.py verifies this property end-to-end.

FLOOR TRANSITION AFTER INTERRUPTION — HARDENED
-----------------------------------------------
asyncio.gather runs _handle_interruption and _on_interruption concurrently.
The critical ordering:

  _handle_interruption:
    1. _interrupt_guard = True           (no suspend)
    2. await request_cancellation()      ← SUSPEND at each cancel_task()
       2a. cancel_task() per active task (synchronous registry transitions +
           asyncio.Task.cancel() — each cancel_task is itself a coroutine with
           no internal await, so it suspends at the gather level, not inside)
       2b. increment_generation()        (synchronous after loop)
    3. state_store.invalidate()          (synchronous — sets floor=INTERRUPTED)
    4. await emit_output(cancellation)   ← SUSPEND

  _on_interruption:
    1–3. classification, disposal        (no suspend)
    4.   await emit_output(filler)       ← SUSPEND
    5.   update_floor(LISTENING)         (synchronous)

The race: if _on_interruption's step 5 (update_floor LISTENING) executes
BEFORE _handle_interruption's step 3 (invalidate → INTERRUPTED), Module 1
attempts a transition from the PRE-INTERRUPTION floor to LISTENING.

Possible pre-interruption floors: LISTENING, THINKING, SPEAKING, WAITING_FOR_TOOL
  - LISTENING  → LISTENING:  not in legal set → ValueError
  - THINKING   → LISTENING:  LEGAL (in _LEGAL_TRANSITIONS)
  - SPEAKING   → LISTENING:  LEGAL
  - WAITING_FOR_TOOL → LISTENING: LEGAL

If it succeeds, _handle_interruption's invalidate() then forces INTERRUPTED
(via force=True inside invalidate), overwriting Module 1's LISTENING.
Result: floor ends as INTERRUPTED. Correct.

If it fails (LISTENING→LISTENING illegal), ValueError is caught, floor stays
LISTENING until invalidate() overwrites it to INTERRUPTED. Still correct.

Hardened fix: Module 1 reads the current floor before attempting the
transition, and ONLY transitions if the floor is INTERRUPTED (i.e., Module 3
has already run invalidate()). If floor is not yet INTERRUPTED, Module 1
defers: the conversation continues correctly because Module 3's invalidate()
has not yet fired, meaning no cancellation has happened yet either, and the
next TEXT_CHUNK or END_OF_TURN handler will drive the floor correctly.

This makes Module 1's floor-transition completely ordering-independent:
- If Module 3 ran first: floor=INTERRUPTED → Module 1 transitions to LISTENING ✓
- If Module 1 ran first: floor unchanged → Module 3 runs invalidate() → INTERRUPTED ✓
In both cases the final floor state is consistent with the interruption.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from module3.runtime.events.base import InputEventType
from module3.runtime.events.input_events import make_interruption
from module3.runtime.events.output_events import make_filler
from module3.runtime.sessions.state import FloorState

from .hypothesis import HypothesisBuffer
from .telemetry import FastPathTelemetry
from .templates import select_filler

if TYPE_CHECKING:
    from module3.runtime.runtime import Runtime
    from module3.runtime.clock.base import Clock
    from module3.runtime.events.base import BaseEvent

logger = logging.getLogger(__name__)

# Speculation is disabled in v1.
# Set to True to enable the speculative-execution extension point once
# baseline interruption recovery is stable and tested.
SPECULATION_ENABLED: bool = False

# HIGH-1: Separate throttle window for competitive interruption acknowledgements.
# This is SHORTER than the normal filler window (state_store.filler_window_ms=3000ms)
# so that ack fillers can always fire after an interruption, even if a normal
# thinking/acknowledgment filler was emitted moments before.
# Protects against true ack spam (two rapid interruptions) while never blocking
# the first ack of a genuine interruption.
_ACK_THROTTLE_MS: float = 500.0


class FastPathRouter:
    """
    Module 1 fast-path control layer.

    Registers handlers for TEXT_CHUNK, END_OF_TURN, and INTERRUPTION
    against the Module 3 runtime. Applies filler/silence/ack policy.

    Constructor arguments
    ---------------------
    runtime : Runtime
        The live Module 3 runtime instance.
    clock   : Clock
        The runtime clock (virtual in tests, real in production).
        Must be the same clock instance used by the runtime.

    Usage::

        router = FastPathRouter(runtime, clock)
        # Handlers are registered in __init__; no further wiring needed.
        await runtime.start()
    """

    def __init__(self, runtime: "Runtime", clock: "Clock") -> None:
        self._runtime = runtime
        self._clock = clock
        self._hypothesis = HypothesisBuffer()
        self._telemetry = FastPathTelemetry()

        # HIGH-1: Per-session timestamp of the last competitive interruption
        # acknowledgement filler. Keyed by session_id. Independent of the normal
        # filler throttle in state_store so that interruption acks are never
        # blocked by a recent END_OF_TURN acknowledgment filler.
        self._last_ack_ms: dict[str, float] = {}

        # Register all fast-path handlers
        runtime.register_handler(InputEventType.TEXT_CHUNK, self._on_text_chunk)
        runtime.register_handler(InputEventType.END_OF_TURN, self._on_end_of_turn)
        runtime.register_handler(InputEventType.INTERRUPTION, self._on_interruption)

    # ------------------------------------------------------------------
    # Public introspection (used by tests and evaluation)
    # ------------------------------------------------------------------

    @property
    def telemetry(self) -> FastPathTelemetry:
        return self._telemetry

    @property
    def hypothesis_buffer(self) -> HypothesisBuffer:
        return self._hypothesis

    # ------------------------------------------------------------------
    # TEXT_CHUNK handler
    # ------------------------------------------------------------------

    async def _on_text_chunk(self, event: "BaseEvent") -> None:
        """
        Per-chunk fast-path processing.

        Steps:
        1. Lazy hypothesis validation (dispose if generation advanced)
        2. Snapshot previous entity state for correction detection
        3. Update hypothesis with new text
        4. If correction detected AND active tasks exist → submit synthetic INTERRUPTION
        5. Otherwise: stay silent (user still speaking)
        6. Record telemetry
        """
        t0 = self._clock.now()
        sid = event.session_id

        current_gen = self._runtime.get_current_generation(sid)
        hypothesis = self._hypothesis.get_or_create(sid, current_gen)

        # Snapshot before update
        prev_entities = dict(hypothesis.entities)

        # Update with new text
        text = event.payload.get("text", "")
        hypothesis.update(text)

        # Correction detection: slot value changed AND active work exists
        # HIGH-2 guard: if this hypothesis already submitted a synthetic INTERRUPTION,
        # skip until the hypothesis is replaced (generation advances or dispose fires).
        if hypothesis.is_correction and not hypothesis._correction_submitted:
            ctx = self._runtime.get_session(sid)
            has_active_work = ctx is not None and bool(
                list(ctx.task_registry.active_tasks())
            )
            if has_active_work:
                logger.info(
                    "FastPathRouter: TEXT_CHUNK correction detected session=%s "
                    "prev=%r curr=%r → submitting synthetic INTERRUPTION",
                    sid, prev_entities, hypothesis.entities,
                )
                # HIGH-2: Mark BEFORE awaiting submit_event so that any TEXT_CHUNK
                # arriving concurrently after this yield point also sees the flag.
                # This is safe because asyncio is cooperative — no true concurrency
                # within the event loop — but marking early is still the cleanest
                # invariant: once we decide to submit, we commit to exactly one.
                hypothesis._correction_submitted = True

                # Submit exactly one competitive INTERRUPTION through the canonical path.
                # Runtime._handle_interruption will perform ALL cancellation mechanics.
                # Do NOT call request_cancellation() directly.
                await self._runtime.submit_event(
                    make_interruption(
                        session_id=sid,
                        timestamp_ms=event.timestamp_ms,
                        text=text,
                        reason="self_correction_detected",
                        competitive=True,
                    )
                )
                # _on_interruption will fire for this event and handle ack + hypothesis reset.
                self._telemetry.record(
                    "correction_detected_ms", self._clock.now() - t0
                )
                return

        # No correction or no active work — stay silent (user still speaking)
        self._telemetry.record("fast_path_decision_ms", self._clock.now() - t0)

    # ------------------------------------------------------------------
    # END_OF_TURN handler
    # ------------------------------------------------------------------

    async def _on_end_of_turn(self, event: "BaseEvent") -> None:
        """
        End-of-turn fast-path processing.

        Steps:
        1. Finalize hypothesis with authoritative final text
        2. Drive floor transition LISTENING → THINKING
        3. Emit acknowledgement filler if throttle allows (generation=None)
        4. Record telemetry
        """
        t0 = self._clock.now()
        sid = event.session_id

        current_gen = self._runtime.get_current_generation(sid)
        hypothesis = self._hypothesis.get_or_create(sid, current_gen)
        hypothesis.finalize(event.payload.get("final_text"))

        ctx = self._runtime.get_session(sid)
        if ctx is None:
            return

        # Floor: LISTENING → THINKING
        try:
            ctx.state_store.update_floor(FloorState.THINKING)
        except ValueError:
            # Already in a non-LISTENING state (e.g., interrupted).
            # Module 3 enforces legality; Module 1 just attempts the transition.
            logger.debug(
                "FastPathRouter: LISTENING→THINKING skipped for session=%s "
                "(current floor: %s)", sid, ctx.state_store.snapshot.floor_state.value
            )

        # Filler: emit only if throttle window allows.
        # IMPORTANT: emit with generation=None (no stamp) to avoid the
        # generation-race described in the module docstring.
        now = self._clock.now()
        if ctx.state_store.can_emit_filler(now):
            await self._emit_filler(sid, "acknowledgment", now)
            ctx.state_store.record_filler_emitted(now)

        self._telemetry.record("end_of_turn_ms", self._clock.now() - t0)

    # ------------------------------------------------------------------
    # INTERRUPTION handler
    # ------------------------------------------------------------------

    async def _on_interruption(self, event: "BaseEvent") -> None:
        """
        Interruption fast-path policy handler.

        This handler runs CONCURRENTLY with Runtime._handle_interruption via
        asyncio.gather. It DOES NOT duplicate any cancellation mechanics.

        Steps:
        1. Classify via event.payload["competitive"] (kernel-level canonical flag)
        2. If backchannel (competitive=False): no action at all
        3. If competitive (competitive=True):
           a. Eagerly dispose hypothesis (unconditional, no generation read needed)
           b. Emit ack filler with generation=None (race-safe, see module docstring)
           c. Attempt floor INTERRUPTED → LISTENING
        4. Record telemetry
        """
        t0 = self._clock.now()
        sid = event.session_id

        competitive = event.payload.get("competitive", True)

        if not competitive:
            # Non-competitive backchannel: Module 3 already skipped cancellation.
            # Module 1 also takes no action: no hypothesis disposal, no filler,
            # no floor transition. The conversation continues uninterrupted.
            logger.debug(
                "FastPathRouter: non-competitive interruption (backchannel) "
                "session=%s — no action", sid
            )
            self._telemetry.record("backchannel_ms", self._clock.now() - t0)
            return

        # Competitive interruption: classify reason for ack template selection
        reason = event.payload.get("reason", "user_interruption")
        if "self_correction" in reason:
            filler_category = "correction_ack"
        else:
            filler_category = "barge_in_ack"

        # 2. Eagerly dispose hypothesis — unconditional, independent of generation.
        # The next TEXT_CHUNK will create a fresh hypothesis via get_or_create.
        self._hypothesis.dispose(sid)

        ctx = self._runtime.get_session(sid)
        if ctx is None:
            return

        # 3. Emit ack filler — use generation=None (race-safe, see module docstring).
        # Do NOT read get_current_generation() and stamp — the read and emit are
        # not atomic and the window between them is large enough for the runtime
        # to advance the generation, causing a spurious stale rejection.
        #
        # HIGH-1: Use the DEDICATED interruption-ack throttle (_last_ack_ms),
        # NOT state_store.can_emit_filler(). This ensures a competitive interruption
        # ack is NEVER blocked by a recent END_OF_TURN filler that set the normal
        # filler throttle. The two throttles are independent:
        #   - state_store filler throttle: governs thinking/wait fillers (3000ms window)
        #   - _last_ack_ms ack throttle:   governs interruption acks (500ms window)
        # Both backchannel (competitive=False) and non-interruption fillers continue
        # to use state_store.can_emit_filler() only — unaffected by this change.
        now = self._clock.now()
        if self._can_emit_ack(sid, now):
            await self._emit_filler(sid, filler_category, now)
            self._record_ack_emitted(sid, now)
            # Also mark the normal filler throttle so END_OF_TURN doesn't spam
            # a filler immediately after an ack.
            ctx.state_store.record_filler_emitted(now)

        # 4. Conditionally attempt floor: INTERRUPTED → LISTENING
        #
        # HARDENED ordering-independent logic (see module docstring):
        # Read the current floor. Only transition if it is ALREADY INTERRUPTED
        # (meaning Module 3's _handle_interruption.invalidate() has already run).
        # If floor is not yet INTERRUPTED, Module 3 hasn't run invalidate() yet;
        # we defer and let its invalidate() set INTERRUPTED. The floor will be
        # correct regardless of which handler ran first.
        #
        # We do NOT attempt transition from arbitrary floors to LISTENING because:
        # - LISTENING→LISTENING is illegal (raises ValueError)
        # - If we succeed on THINKING→LISTENING, Module 3 immediately overwrites
        #   with INTERRUPTED via invalidate(). Net effect is the same but it
        #   creates unnecessary version increments in the snapshot log.
        current_floor = ctx.state_store.snapshot.floor_state
        if current_floor == FloorState.INTERRUPTED:
            try:
                ctx.state_store.update_floor(FloorState.LISTENING)
                logger.debug(
                    "FastPathRouter: INTERRUPTED→LISTENING after competitive "
                    "interruption session=%s", sid
                )
            except ValueError:
                # Should not happen (INTERRUPTED→LISTENING is legal), but guard
                # defensively — a missed floor transition is not correctness failure.
                logger.warning(
                    "FastPathRouter: unexpected ValueError on INTERRUPTED→LISTENING "
                    "session=%s floor=%s", sid, ctx.state_store.snapshot.floor_state.value
                )
        else:
            # Module 3 hasn't invalidated yet; floor transition deferred.
            # _handle_interruption's invalidate() will set INTERRUPTED.
            logger.debug(
                "FastPathRouter: floor=%s (not yet INTERRUPTED), deferring "
                "LISTENING transition for session=%s",
                current_floor.value, sid
            )

        self._telemetry.record("interruption_ack_ms", self._clock.now() - t0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _can_emit_ack(self, session_id: str, now_ms: float) -> bool:
        """
        HIGH-1: True if a competitive interruption acknowledgement filler may
        be emitted for this session right now.

        Uses a SEPARATE, shorter throttle window (_ACK_THROTTLE_MS = 500ms)
        from the normal filler throttle in state_store (3000ms window). This
        ensures that a recent END_OF_TURN thinking-filler does NOT suppress
        a competitive interruption acknowledgement.

        Protection against ack spam: two rapid competitive interruptions within
        _ACK_THROTTLE_MS of each other → only the first emits an ack.
        """
        last = self._last_ack_ms.get(session_id)
        if last is None:
            return True
        return (now_ms - last) >= _ACK_THROTTLE_MS

    def _record_ack_emitted(self, session_id: str, now_ms: float) -> None:
        """HIGH-1: Record the timestamp of the last interruption ack for this session."""
        self._last_ack_ms[session_id] = now_ms

    async def _emit_filler(
        self,
        session_id: str,
        category: str,
        timestamp_ms: float,
    ) -> None:
        """
        Emit a filler event.

        Generation is NOT stamped (generation=None) so that the output gate's
        staleness check (`if event.generation is not None and ...`) is bypassed.
        This is intentional: a filler is an acknowledgement speech act, not a
        semantic response tied to a specific generation's task results.
        """
        text = select_filler(category)
        filler_event = make_filler(
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            text=text,
            # generation intentionally omitted (defaults to None in BaseEvent)
        )
        try:
            await self._runtime.emit_output(filler_event, priority=10)
            logger.debug(
                "FastPathRouter: emitted filler [%s] session=%s text=%r",
                category, session_id, text,
            )
        except ValueError as exc:
            # Output gate rejected the filler (e.g. session closed).
            # Log and continue — a missed filler is a quality loss, not a
            # correctness failure.
            logger.warning(
                "FastPathRouter: filler rejected session=%s: %r", session_id, exc
            )
