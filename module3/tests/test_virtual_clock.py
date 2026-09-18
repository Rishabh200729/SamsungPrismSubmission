"""
module3/tests/test_virtual_clock.py

Tests for the deterministic VirtualClock.
"""
import asyncio
import pytest

from module3.runtime.clock.virtual_clock import VirtualClock


class TestVirtualClock:
    def test_initial_time(self):
        clock = VirtualClock(start_ms=100.0)
        assert clock.now() == 100.0

    def test_advance_increases_time(self):
        clock = VirtualClock()
        clock.advance(50.0)
        assert clock.now() == 50.0
        clock.advance(25.0)
        assert clock.now() == 75.0

    def test_advance_to(self):
        clock = VirtualClock()
        clock.advance_to(500.0)
        assert clock.now() == 500.0

    def test_advance_to_past_is_noop(self):
        clock = VirtualClock(start_ms=100.0)
        clock.advance_to(50.0)   # In the past, no-op
        assert clock.now() == 100.0

    def test_advance_negative_raises(self):
        clock = VirtualClock()
        with pytest.raises(ValueError):
            clock.advance(-10.0)

    def test_schedule_fires_callback(self):
        clock = VirtualClock()
        fired = []
        clock.schedule(lambda: fired.append(1), at_ms=100.0)
        clock.advance_to(200.0)
        assert fired == [1]

    def test_schedule_fires_in_order(self):
        clock = VirtualClock()
        order = []
        clock.schedule(lambda: order.append("c"), at_ms=300.0)
        clock.schedule(lambda: order.append("a"), at_ms=100.0)
        clock.schedule(lambda: order.append("b"), at_ms=200.0)
        clock.advance_to(400.0)
        assert order == ["a", "b", "c"]

    def test_callback_not_fired_before_time(self):
        clock = VirtualClock()
        fired = []
        clock.schedule(lambda: fired.append(1), at_ms=500.0)
        clock.advance_to(200.0)
        assert fired == []

    def test_multiple_callbacks_at_same_time(self):
        clock = VirtualClock()
        fired = []
        clock.schedule(lambda: fired.append("x"), at_ms=100.0)
        clock.schedule(lambda: fired.append("y"), at_ms=100.0)
        clock.advance_to(100.0)
        assert len(fired) == 2

    async def test_async_advance_fires_callbacks(self):
        clock = VirtualClock()
        fired = []
        clock.schedule(lambda: fired.append("fired"), at_ms=100.0)
        await clock.advance_async(200.0)
        assert fired == ["fired"]

    async def test_sleep_wakes_on_advance(self):
        clock = VirtualClock()
        woke = []

        async def sleeper():
            await clock.sleep(100.0)
            woke.append("awake")

        asyncio.ensure_future(sleeper())
        await asyncio.sleep(0)  # Let sleeper register
        await clock.advance_async(200.0)
        assert woke == ["awake"]

    async def test_run_until_idle_fires_all(self):
        clock = VirtualClock()
        fired = []
        clock.schedule(lambda: fired.append(1), at_ms=100.0)
        clock.schedule(lambda: fired.append(2), at_ms=200.0)
        clock.schedule(lambda: fired.append(3), at_ms=300.0)
        await clock.run_until_idle()
        assert fired == [1, 2, 3]
        assert clock.now() == 300.0

    def test_is_idle_with_no_pending(self):
        clock = VirtualClock()
        assert clock.is_idle()

    def test_is_idle_false_with_pending(self):
        clock = VirtualClock()
        clock.schedule(lambda: None, at_ms=100.0)
        assert not clock.is_idle()

    async def test_deterministic_scenario_ordering(self):
        """
        Replicate the spec's example:
        T=0   TEXT_CHUNK
        T=300 agent starts background work (simulated)
        T=450 INTERRUPTION
        T=700 old TOOL_RESULT
        T=900 new response
        """
        clock = VirtualClock()
        events_order = []

        clock.schedule(lambda: events_order.append(("TEXT_CHUNK", 0)), at_ms=0)
        clock.schedule(lambda: events_order.append(("BACKGROUND_WORK", 300)), at_ms=300)
        clock.schedule(lambda: events_order.append(("INTERRUPTION", 450)), at_ms=450)
        clock.schedule(lambda: events_order.append(("TOOL_RESULT", 700)), at_ms=700)
        clock.schedule(lambda: events_order.append(("FINAL_RESPONSE", 900)), at_ms=900)

        await clock.run_until_idle()

        assert len(events_order) == 5
        # Verify causal ordering
        timestamps = [e[1] for e in events_order]
        assert timestamps == sorted(timestamps)
        assert events_order[0][0] == "TEXT_CHUNK"
        assert events_order[2][0] == "INTERRUPTION"
        assert events_order[3][0] == "TOOL_RESULT"
        assert events_order[4][0] == "FINAL_RESPONSE"
