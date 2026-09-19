"""
module1/tests/test_high_fixes.py

Deterministic regression tests for the two HIGH-priority audit fixes.

HIGH-1: Competitive interruption acknowledgement must emit even when a normal
        filler was emitted recently (separate throttle windows).

HIGH-2: ONE detected correction → EXACTLY ONE synthetic INTERRUPTION event.
        Rapid TEXT_CHUNKs before generation advances must not produce duplicates.

Each test is labelled with its audit requirement identifier.
"""

from __future__ import annotations

import asyncio
import pytest

from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.clock.virtual_clock import VirtualClock
from module3.runtime.events.base import InputEventType, OutputEventType
from module3.runtime.events.input_events import (
    make_text_chunk,
    make_end_of_turn,
    make_interruption,
)
from module3.runtime.tasks.lifecycle import TaskStatus

from module1.fast_path.router import FastPathRouter, _ACK_THROTTLE_MS
from module1.fast_path.hypothesis import HypothesisState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ticks(n: int = 40) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def _wait_for(predicate, attempts: int = 80) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate(), "wait_for timed out"


# ===========================================================================
# HIGH-1 — INTERRUPTION ACK SEPARATE THROTTLE
# ===========================================================================

class TestHigh1AckThrottle:
    """
    HIGH-1: Competitive interruption acks use _last_ack_ms (500ms window),
    NOT state_store.can_emit_filler() (3000ms window).
    """

    @pytest.mark.asyncio
    async def test_A_ack_emits_after_eot_filler_within_normal_window(self):
        """
        HIGH-1 requirement A:
        END_OF_TURN at t=0 → normal filler emitted (state_store throttle set).
        Competitive interruption at t=50ms → ack MUST emit despite 3s throttle.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()

        try:
            # END_OF_TURN at t=0 → emits normal filler
            await rt.submit_event(make_end_of_turn(sid, 0.0, final_text="Book a flight"))
            await _ticks(20)
            outputs_after_eot = await rt.drain_outputs()
            eot_fillers = [o for o in outputs_after_eot if o.event_type == OutputEventType.FILLER]
            assert eot_fillers, "END_OF_TURN must produce a filler (precondition)"

            # Advance only 50ms — well within the 3000ms normal throttle window
            clock.advance(50.0)

            # Competitive interruption at t=50ms
            await rt.submit_event(make_interruption(sid, 50.0, competitive=True))
            await _ticks(40)

            # Interruption ack MUST emit despite normal throttle being active
            outputs_after_int = await rt.drain_outputs()
            ack_fillers = [o for o in outputs_after_int if o.event_type == OutputEventType.FILLER]
            assert ack_fillers, (
                "Competitive interruption ack must emit even if a normal filler "
                "was emitted 50ms ago (within the 3000ms normal throttle window). "
                "HIGH-1 fix: acks use a separate 500ms throttle."
            )
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_B_two_rapid_interruptions_no_ack_spam(self):
        """
        HIGH-1 requirement B:
        Two competitive interruptions within _ACK_THROTTLE_MS of each other
        must produce at most ONE ack filler (spam protection).
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()

        try:
            # First interruption at t=0
            await rt.submit_event(make_interruption(sid, 0.0, competitive=True))
            await _ticks(30)

            # Second interruption at t=100ms (within 500ms ack window)
            clock.advance(100.0)
            await rt.submit_event(make_interruption(sid, 100.0, competitive=True))
            await _ticks(30)

            outputs = await rt.drain_outputs()
            fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
            # At most 1 ack — second interruption within ack throttle window is suppressed
            assert len(fillers) <= 1, (
                f"Two rapid interruptions within {_ACK_THROTTLE_MS}ms must not "
                f"produce ack spam. Got {len(fillers)} fillers."
            )
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_B2_two_interruptions_beyond_ack_window_both_emit(self):
        """
        Two competitive interruptions separated by more than _ACK_THROTTLE_MS
        must each be allowed to emit an ack.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()

        try:
            # First interruption at t=0
            await rt.submit_event(make_interruption(sid, 0.0, competitive=True))
            await _ticks(30)
            outputs1 = await rt.drain_outputs()
            ack1 = [o for o in outputs1 if o.event_type == OutputEventType.FILLER]
            assert ack1, "First interruption ack must emit"

            # Advance past ack throttle window
            clock.advance(_ACK_THROTTLE_MS + 10.0)

            # Second interruption well past ack window
            await rt.submit_event(
                make_interruption(sid, _ACK_THROTTLE_MS + 10.0, competitive=True)
            )
            await _ticks(30)
            outputs2 = await rt.drain_outputs()
            ack2 = [o for o in outputs2 if o.event_type == OutputEventType.FILLER]
            assert ack2, (
                f"Second interruption >{_ACK_THROTTLE_MS}ms after first must also emit ack"
            )
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_C_normal_eot_fillers_remain_throttled(self):
        """
        HIGH-1 requirement C:
        Normal END_OF_TURN fillers are still governed by state_store throttle.
        Two END_OF_TURN events within the 3000ms window → only one filler.
        (This test is a regression — the normal throttle must be unaffected.)
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()

        try:
            await rt.submit_event(make_end_of_turn(sid, 0.0, final_text="Book a flight"))
            await _ticks(20)
            clock.advance(100.0)
            await rt.submit_event(make_end_of_turn(sid, 100.0, final_text="Book a flight"))
            await _ticks(20)

            outputs = await rt.drain_outputs()
            fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
            assert len(fillers) == 1, (
                f"Normal END_OF_TURN throttle must still work: "
                f"expected 1 filler, got {len(fillers)}"
            )
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_D_backchannel_remains_silent(self):
        """
        HIGH-1 requirement D:
        competitive=False backchannel must still produce no filler.
        HIGH-1 fix must not accidentally enable backchannel acks.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()

        try:
            # First: emit an END_OF_TURN filler so state_store throttle is active
            await rt.submit_event(make_end_of_turn(sid, 0.0, final_text="Hello"))
            await _ticks(15)
            await rt.drain_outputs()

            # Backchannel within throttle window
            clock.advance(50.0)
            await rt.submit_event(make_interruption(sid, 50.0, competitive=False))
            await _ticks(20)

            outputs = await rt.drain_outputs()
            fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
            assert fillers == [], (
                f"Backchannel must produce zero fillers regardless of throttle state. "
                f"Got: {fillers}"
            )
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_E_existing_ack_race_test_still_passes(self):
        """
        HIGH-1 requirement E:
        Regression — generation=None ack race safety (originally test_12) must
        still hold after the throttle change.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()

        try:
            await rt.submit_event(make_interruption(sid, 0.0, competitive=True))
            await _ticks(40)

            assert rt.get_current_generation(sid) == 2, "Generation must advance"

            outputs = await rt.drain_outputs()
            fillers = [o for o in outputs if o.event_type == OutputEventType.FILLER]
            assert fillers, "Ack filler must not be lost on generation race"

            for f in fillers:
                assert f.generation is None, (
                    f"All ack fillers must have generation=None, got: {f.generation}"
                )
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_ack_throttle_is_per_session_not_global(self):
        """
        Two concurrent sessions: first session's recent ack must not block
        second session's ack.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid_a = rt.create_session()
        sid_b = rt.create_session()

        try:
            # Session A: interrupt at t=0 → ack emitted, throttle set for A
            await rt.submit_event(make_interruption(sid_a, 0.0, competitive=True))
            await _ticks(30)
            # Session B: interrupt at t=10ms — should NOT be blocked by A's throttle
            clock.advance(10.0)
            await rt.submit_event(make_interruption(sid_b, 10.0, competitive=True))
            await _ticks(30)

            outputs = await rt.drain_outputs()
            # Filter per session
            acks_a = [o for o in outputs if o.event_type == OutputEventType.FILLER
                      and o.session_id == sid_a]
            acks_b = [o for o in outputs if o.event_type == OutputEventType.FILLER
                      and o.session_id == sid_b]
            assert acks_a, "Session A must have ack"
            assert acks_b, "Session B must have ack independent of session A throttle"
        finally:
            await rt.stop()


# ===========================================================================
# HIGH-2 — ONE CORRECTION → ONE SYNTHETIC INTERRUPTION
# ===========================================================================

class TestHigh2SingleSyntheticInterruption:
    """
    HIGH-2: _correction_submitted flag ensures exactly one synthetic INTERRUPTION
    per hypothesis regardless of how many TEXT_CHUNKs arrive before generation
    advances.
    """

    @pytest.mark.asyncio
    async def test_A_one_correction_exactly_one_interruption(self):
        """
        HIGH-2 requirement A:
        A single correction detected → exactly one synthetic INTERRUPTION event.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)

        interruptions: list = []

        async def spy(event):
            if event.payload.get("competitive"):
                interruptions.append(event)

        rt.register_handler(InputEventType.INTERRUPTION, spy)
        await rt.start()
        sid = rt.create_session()
        blocker = asyncio.Event()

        try:
            task_id = await rt.create_background_task(sid, blocker.wait(), "active")
            await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

            await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
            await _ticks(10)
            await rt.submit_event(make_text_chunk(sid, 100.0, "actually change it to Osaka"))
            await _ticks(40)

            await _wait_for(lambda: rt.get_current_generation(sid) >= 2, attempts=80)

            correction_ints = [
                e for e in interruptions
                if e.payload.get("reason") == "self_correction_detected"
            ]
            assert len(correction_ints) == 1, (
                f"Exactly 1 synthetic INTERRUPTION expected, got {len(correction_ints)}: "
                f"{[e.payload for e in correction_ints]}"
            )
        finally:
            blocker.set()
            await rt.stop()

    @pytest.mark.asyncio
    async def test_B_rapid_chunks_before_generation_advances_still_one(self):
        """
        HIGH-2 requirement B:
        Multiple TEXT_CHUNKs arrive in rapid succession before Module 3 has
        advanced the generation. Only ONE synthetic INTERRUPTION must be submitted.

        This directly tests the _correction_submitted guard.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)

        all_interruptions: list = []

        async def spy(event):
            all_interruptions.append(event)

        rt.register_handler(InputEventType.INTERRUPTION, spy)
        await rt.start()
        sid = rt.create_session()
        blocker = asyncio.Event()

        try:
            task_id = await rt.create_background_task(sid, blocker.wait(), "active")
            await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

            # Establish initial destination
            await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
            await _ticks(5)

            # Send correction chunk
            await rt.submit_event(make_text_chunk(sid, 10.0, "change it to Osaka"))
            # Immediately (before generation can advance) send MORE correction-like chunks
            # that would re-trigger is_correction on the same hypothesis
            await rt.submit_event(make_text_chunk(sid, 11.0, "change it to Osaka"))
            await rt.submit_event(make_text_chunk(sid, 12.0, "change it to Osaka"))
            await _ticks(60)

            await _wait_for(lambda: rt.get_current_generation(sid) >= 2, attempts=80)

            synthetic = [
                e for e in all_interruptions
                if e.payload.get("reason") == "self_correction_detected"
                and e.payload.get("competitive") is True
            ]
            assert len(synthetic) == 1, (
                f"Rapid chunks before generation advances must produce exactly 1 "
                f"synthetic INTERRUPTION via _correction_submitted guard. "
                f"Got {len(synthetic)}: {[e.payload for e in synthetic]}"
            )
        finally:
            blocker.set()
            await rt.stop()

    @pytest.mark.asyncio
    async def test_C_new_correction_after_new_generation_is_allowed(self):
        """
        HIGH-2 requirement C:
        After an interruption advances the generation, a NEW correction on the
        new hypothesis (new generation) must still trigger a synthetic INTERRUPTION.
        The _correction_submitted flag is per-hypothesis, so it resets naturally.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)

        interruptions: list = []

        async def spy(event):
            if event.payload.get("competitive") and event.payload.get("reason") == "self_correction_detected":
                interruptions.append(event)

        rt.register_handler(InputEventType.INTERRUPTION, spy)
        await rt.start()
        sid = rt.create_session()

        blocker1 = asyncio.Event()
        blocker2 = asyncio.Event()

        try:
            # Turn 1: establish destination Tokyo, trigger correction → Osaka
            t1 = await rt.create_background_task(sid, blocker1.wait(), "task-1")
            await _wait_for(lambda: rt.get_task(sid, t1).status == TaskStatus.RUNNING)

            await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
            await _ticks(5)
            await rt.submit_event(make_text_chunk(sid, 50.0, "change it to Osaka"))
            await _ticks(30)

            await _wait_for(lambda: rt.get_current_generation(sid) == 2, attempts=80)
            assert len(interruptions) == 1, "First correction must produce 1 interruption"

            # Turn 2: new generation, new task, new correction
            t2 = await rt.create_background_task(sid, blocker2.wait(), "task-2")
            await _wait_for(lambda: rt.get_task(sid, t2).status == TaskStatus.RUNNING)

            clock.advance(1000.0)
            await rt.submit_event(make_text_chunk(sid, 1000.0, "Book a flight to Seoul"))
            await _ticks(5)
            await rt.submit_event(make_text_chunk(sid, 1050.0, "change it to Busan"))
            await _ticks(40)

            await _wait_for(lambda: rt.get_current_generation(sid) == 3, attempts=80)
            assert len(interruptions) == 2, (
                f"Second correction on new generation must produce a second interruption. "
                f"Got {len(interruptions)} total interruptions."
            )
        finally:
            blocker1.set()
            blocker2.set()
            await rt.stop()

    @pytest.mark.asyncio
    async def test_D_unrelated_chunks_no_interruption(self):
        """
        HIGH-2 requirement D:
        TEXT_CHUNKs that don't change any slot value produce no interruptions.
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)

        interruptions: list = []

        async def spy(event):
            interruptions.append(event)

        rt.register_handler(InputEventType.INTERRUPTION, spy)
        await rt.start()
        sid = rt.create_session()
        blocker = asyncio.Event()

        try:
            await rt.create_background_task(sid, blocker.wait(), "active")
            for i, text in enumerate(["Book", "a", "flight", "to", "actually", "change", "it"]):
                await rt.submit_event(make_text_chunk(sid, float(i * 50), text))
                await _ticks(5)

            # No entity ever established → no correction possible
            synthetic = [
                e for e in interruptions
                if e.payload.get("reason") == "self_correction_detected"
            ]
            assert not synthetic, (
                f"Unrelated chunks with no established entity must not trigger "
                f"synthetic INTERRUPTION. Got: {[e.payload for e in synthetic]}"
            )
        finally:
            blocker.set()
            await rt.stop()

    @pytest.mark.asyncio
    async def test_E_module3_generation_increments_exactly_once(self):
        """
        HIGH-2 requirement E:
        Even under rapid chunk delivery, Module 3 generation increments exactly
        once per detected correction (deduplication preserved end-to-end).
        """
        clock = VirtualClock(start_ms=0.0)
        rt = Runtime(config=TEST_CONFIG, clock=clock)
        router = FastPathRouter(rt, clock)
        await rt.start()
        sid = rt.create_session()
        blocker = asyncio.Event()

        try:
            task_id = await rt.create_background_task(sid, blocker.wait(), "active")
            await _wait_for(lambda: rt.get_task(sid, task_id).status == TaskStatus.RUNNING)

            assert rt.get_current_generation(sid) == 1

            await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight to Tokyo"))
            await _ticks(5)
            # Correction chunk + rapid repeats
            await rt.submit_event(make_text_chunk(sid, 50.0, "change it to Osaka"))
            await rt.submit_event(make_text_chunk(sid, 51.0, "change it to Osaka"))
            await rt.submit_event(make_text_chunk(sid, 52.0, "change it to Osaka"))
            await _ticks(60)

            await _wait_for(lambda: rt.get_current_generation(sid) >= 2, attempts=80)

            # Generation must be exactly 2 — not 3 or 4 from duplicate submissions
            final_gen = rt.get_current_generation(sid)
            assert final_gen == 2, (
                f"Module 3 generation must be exactly 2 after one correction. "
                f"Got {final_gen} — indicates duplicate INTERRUPTION events were processed."
            )
        finally:
            blocker.set()
            await rt.stop()

    def test_correction_submitted_flag_on_hypothesis_state(self):
        """
        Unit test: _correction_submitted starts False, can be set, is not
        cleared by update() — only by object replacement.
        """
        h = HypothesisState(generation=1)
        assert not h._correction_submitted

        h.update("Book a flight to Tokyo")
        assert not h._correction_submitted, "update() must not touch _correction_submitted"

        h._correction_submitted = True
        h.update("change it to Osaka")  # Even correction detection doesn't clear it
        assert h._correction_submitted, "_correction_submitted must survive update()"

        # New hypothesis (object replacement) resets it
        h2 = HypothesisState(generation=2)
        assert not h2._correction_submitted, "Fresh HypothesisState must start with False"

    def test_correction_submitted_reset_on_buffer_dispose(self):
        """
        Unit test: After dispose() + get_or_create(), the new hypothesis has
        _correction_submitted=False even if the old one had it True.
        """
        from module1.fast_path.hypothesis import HypothesisBuffer

        buf = HypothesisBuffer()
        h1 = buf.get_or_create("s1", 1)
        h1.update("Book a flight to Tokyo")
        h1._correction_submitted = True

        buf.dispose("s1")
        h2 = buf.get_or_create("s1", 1)  # same generation, but disposed → fresh
        assert h2 is not h1
        assert not h2._correction_submitted

    def test_correction_submitted_reset_on_generation_change(self):
        """
        Unit test: get_or_create with new generation returns fresh hypothesis
        with _correction_submitted=False (lazy invalidation path).
        """
        from module1.fast_path.hypothesis import HypothesisBuffer

        buf = HypothesisBuffer()
        h1 = buf.get_or_create("s1", 1)
        h1._correction_submitted = True

        # New generation — even without dispose() — produces fresh hypothesis
        h2 = buf.get_or_create("s1", 2)
        assert h2 is not h1
        assert not h2._correction_submitted
