"""
Regression tests for defects found by auditing PRISM against the REAL FDB-v3 evaluator
(github.com/DanielLin94144/Full-Duplex-Bench, v3/).

Why these matter for the score
------------------------------
`evaluate_pass_rate.py` decides PASS with *multiset matching by function name*
(recall AND precision, no LLM involved).  So

  * a legitimate repeated call that is erased from the log  -> "Missing tools" -> FAIL
  * a corrected call that the parser never sees             -> FAIL
  * 10 of the 100 public scenarios expect the SAME tool twice (travel_15/23/25,
    finance_13/14/22, housing_13/25, ecommerce_22/24), so they were unpassable.

Each test below reproduces one verified defect.
"""

import asyncio
import json
import os
import tempfile
import unittest

from prism import EffectClass, ToolCall, TRPState
from prism.saga_coordinator import SagaCoordinator
from prism.tool_dispatcher import (
    EFFECT_MAP,
    ToolDispatcher,
    register_effect,
)
from prism.trp_gate import TRPGate


def official_extract(log_path: str, room: str, stream_start_time: float = 0.0):
    """VERBATIM copy of the tool-call extraction loop in FDB-v3
    v3/run_tool_benchmark.py ("Step 6: Extract actual tool calls from telemetry").
    Kept identical on purpose so these tests exercise what the organizers' pipeline does."""
    actual_tool_calls = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t_data = json.loads(line)
                    if t_data.get("room") == room:
                        call_data = t_data.get("call")
                        if stream_start_time:
                            if "timestamp_start" in call_data:
                                call_data["timestamp_start"] = round(call_data["timestamp_start"] - stream_start_time, 2)
                            if "timestamp_end" in call_data:
                                call_data["timestamp_end"] = round(call_data["timestamp_end"] - stream_start_time, 2)
                            if "timestamp" in call_data:
                                call_data["timestamp"] = round(call_data["timestamp"] - stream_start_time, 2)
                        actual_tool_calls.append(call_data)
    except Exception:
        pass  # the real pipeline swallows this and keeps whatever it collected so far
    return actual_tool_calls


class DispatcherCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.log_path = tempfile.mktemp(suffix=".log")
        self.executed = []
        self.epoch = 0
        self.saga = SagaCoordinator()

    def tearDown(self):
        for p in (self.log_path, self.log_path + ".audit"):
            if os.path.exists(p):
                os.remove(p)

    def make(self, use_epoch=True, **kw):
        def registry(name, **kwargs):
            self.executed.append((name, kwargs))
            return {"status": "success", "name": name, **kwargs}
        return ToolDispatcher(
            registry_fn=registry, saga=self.saga, tool_log_path=self.log_path,
            room_name="room-1",
            correction_epoch_fn=(lambda: self.epoch) if use_epoch else None, **kw,
        )

    def logged(self):
        return official_extract(self.log_path, "room-1")

    @staticmethod
    def call(name, **args):
        return ToolCall(name=name, args=args, effect_class=EFFECT_MAP[name])

    async def settle(self):
        await asyncio.sleep(0.01)


class TestLegitimateRepeatsSurvive(DispatcherCase):
    """Defects A/B: same tool called twice for two DIFFERENT things must yield two calls."""

    async def test_two_read_only_calls_for_different_orders_both_execute(self):
        d = self.make()
        t1 = asyncio.create_task(d.dispatch(self.call("track_order", order_id="A1")))
        await self.settle()
        t2 = asyncio.create_task(d.dispatch(self.call("track_order", order_id="B2")))
        await self.settle()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        r1, r2 = json.loads(await t1), json.loads(await t2)
        self.assertEqual((r1["status"], r2["status"]), ("success", "success"))
        self.assertEqual([c["args"]["order_id"] for c in self.logged()], ["A1", "B2"])

    async def test_two_mutations_for_different_products_both_logged(self):
        d = self.make()
        t1 = asyncio.create_task(d.dispatch(self.call("add_to_cart", product_id="X1", quantity=1)))
        t2 = asyncio.create_task(d.dispatch(self.call("add_to_cart", product_id="Y2", quantity=1)))
        await self.settle()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await t1, await t2
        self.assertEqual([a["product_id"] for _, a in self.executed], ["X1", "Y2"])
        # previously only the LAST one survived in the log although BOTH had executed
        self.assertEqual([c["args"]["product_id"] for c in self.logged()], ["X1", "Y2"])

    async def test_sequential_chained_same_tool_both_logged(self):
        """Second call issued only after the first committed (chained calls)."""
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)   # turn already confirmed
        await d.dispatch(self.call("update_search_filter", filter_name="alpha", value=1))
        await d.dispatch(self.call("update_search_filter", filter_name="beta", value=2))
        self.assertEqual([c["args"]["filter_name"] for c in self.logged()], ["alpha", "beta"])

    async def test_repair_marker_heard_before_both_calls_does_not_supersede(self):
        """A repair earlier in the utterance says nothing about two later legit calls."""
        self.epoch = 1                       # "wait..." was said BEFORE either call
        d = self.make()
        t1 = asyncio.create_task(d.dispatch(self.call("track_order", order_id="A1")))
        t2 = asyncio.create_task(d.dispatch(self.call("track_order", order_id="B2")))
        await self.settle()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await t1, await t2
        self.assertEqual(len(self.logged()), 2)


class TestRepairsStillSupersede(DispatcherCase):
    """The intended PRISM behaviour must survive: a real repair discards the stale call."""

    async def test_repaired_read_only_call_is_superseded(self):
        d = self.make()
        t1 = asyncio.create_task(d.dispatch(self.call("search_flights", destination="Rome", date="May 1")))
        await self.settle()
        self.epoch += 1                       # user says "wait, actually ..."
        t2 = asyncio.create_task(d.dispatch(self.call("search_flights", destination="Oslo", date="May 1")))
        await self.settle()
        self.assertEqual(json.loads(await t1)["status"], "superseded")
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await t2
        self.assertEqual([c["args"]["destination"] for c in self.logged()], ["Oslo"])

    async def test_repaired_staged_mutation_never_executes(self):
        """Previously BOTH staged modify_autopay calls executed at TRP_CONFIRMED
        (a duplicate state change - the thing the task forbids)."""
        d = self.make()
        t1 = asyncio.create_task(d.dispatch(self.call("modify_autopay", bill_type="water", source_account="checking")))
        await self.settle()
        self.epoch += 1
        t2 = asyncio.create_task(d.dispatch(self.call("modify_autopay", bill_type="gas", source_account="checking")))
        await self.settle()
        self.assertEqual(json.loads(await t1)["status"], "superseded")
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await t2
        self.assertEqual([a["bill_type"] for _, a in self.executed], ["gas"])   # stale one never ran
        self.assertEqual([c["args"]["bill_type"] for c in self.logged()], ["gas"])

    async def test_legacy_mode_without_epoch_keeps_old_same_tool_rule(self):
        d = self.make(use_epoch=False)
        t1 = asyncio.create_task(d.dispatch(self.call("search_flights", destination="A", date="x")))
        await self.settle()
        t2 = asyncio.create_task(d.dispatch(self.call("search_flights", destination="B", date="x")))
        await self.settle()
        self.assertEqual(json.loads(await t1)["status"], "superseded")
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await t2


class TestEarlyCommitEvidence(DispatcherCase):
    """TRP confirmed BEFORE the repair arrived, so the stale call already committed."""

    async def test_executed_stale_mutation_is_not_erased_from_the_trace(self):
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("book_flight", passenger_name="Ana"))    # executes
        self.epoch += 1                                                      # then: "no wait, Bo"
        await d.dispatch(self.call("book_flight", passenger_name="Bo"))
        # Honest trace: both real executions are visible (a real duplicate is a real failure).
        self.assertEqual([c["args"]["passenger_name"] for c in self.logged()], ["Ana", "Bo"])
        with open(self.log_path + ".audit") as f:
            events = [json.loads(l) for l in f if l.strip()]
        self.assertTrue(any(e.get("event") == "stale_mutation_committed_before_repair" for e in events))

    async def test_opt_in_flag_retracts_committed_mutation(self):
        d = self.make(retract_committed_mutations=True)
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("book_flight", passenger_name="Ana"))
        self.epoch += 1
        await d.dispatch(self.call("book_flight", passenger_name="Bo"))
        self.assertEqual([c["args"]["passenger_name"] for c in self.logged()], ["Bo"])

    async def test_read_only_early_commit_is_retracted_by_repair(self):
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("search_flights", destination="Rome", date="May 1"))
        self.epoch += 1
        await d.dispatch(self.call("search_flights", destination="Oslo", date="May 1"))
        self.assertEqual([c["args"]["destination"] for c in self.logged()], ["Oslo"])

    async def test_retraction_touches_only_the_targeted_line(self):
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("track_order", order_id="A1"))
        await d.dispatch(self.call("search_flights", destination="Rome", date="May 1"))
        self.epoch += 1
        await d.dispatch(self.call("search_flights", destination="Oslo", date="May 1"))
        self.assertEqual([c["function"] for c in self.logged()], ["track_order", "search_flights"])


class TestEvaluatorLogContract(DispatcherCase):
    """The tool-call log must stay parseable by the organizers' pipeline."""

    async def test_compensation_goes_to_audit_log_not_tool_call_log(self):
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("modify_autopay", bill_type="water", source_account="checking"))
        await self.saga.compensate_all()
        with open(self.log_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertTrue(all("call" in l for l in lines), "every evaluator-log line needs a 'call' key")
        with open(self.log_path + ".audit") as f:
            self.assertIn("compensation", json.loads(f.readline()))

    async def test_corrected_call_after_rollback_is_visible_to_official_parser(self):
        """commit -> barge-in rollback -> user re-issues corrected call.  With the old
        behaviour the compensation line made the parser raise TypeError and drop the
        corrected call, so the scorer saw only the STALE one."""
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("modify_autopay", bill_type="water", source_account="checking"))
        await self.saga.compensate_all()
        await d.dispatch(self.call("modify_autopay", bill_type="water", source_account="savings"))
        seen = official_extract(self.log_path, "room-1", stream_start_time=1.0)
        self.assertEqual([c["args"]["source_account"] for c in seen], ["checking", "savings"])

    async def test_extra_call_id_key_is_harmless_to_official_parser(self):
        d = self.make()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await d.dispatch(self.call("get_card_benefits", card_type="gold"))
        (call,) = official_extract(self.log_path, "room-1", stream_start_time=1.0)
        self.assertEqual(set(call), {"function", "args", "timestamp_start", "timestamp_end"})


class TestTurnResetDoesNotHang(DispatcherCase):
    async def test_pending_calls_are_resolved_when_user_resumes_speaking(self):
        d = self.make()
        ro = asyncio.create_task(d.dispatch(self.call("search_flights", destination="Rome", date="May 1")))
        mut = asyncio.create_task(d.dispatch(self.call("book_flight", passenger_name="Ana")))
        await self.settle()
        d.reset_for_new_turn()                    # agent does this on every user_speech_started
        done, pending = await asyncio.wait({ro, mut}, timeout=1.0)
        self.assertEqual(len(pending), 0, "dispatch() must not hang after reset_for_new_turn()")
        for t in done:
            self.assertEqual(json.loads(t.result())["status"], "discarded")
        # READ_ONLY calls run speculatively BY DESIGN (no side effects); the state-changing
        # call must never have run, and neither may appear in the evaluator's log.
        self.assertNotIn("book_flight", [n for n, _ in self.executed])
        self.assertEqual(self.logged(), [])


class TestUnknownToolsFailClosed(DispatcherCase):
    def test_unknown_tool_defaults_to_compensable(self):
        self.assertEqual(EFFECT_MAP.get("transfer_funds", EffectClass.COMPENSABLE), EffectClass.COMPENSABLE)
        self.assertEqual(EFFECT_MAP["some_new_tool"], EffectClass.COMPENSABLE)
        self.assertIsNone(EFFECT_MAP.get("some_new_tool"))

    async def test_unknown_tool_is_staged_not_run_speculatively(self):
        d = self.make()
        task = asyncio.create_task(d.dispatch(ToolCall("update_destination", {"place": "x"}, EffectClass.READ_ONLY)))
        await self.settle()
        self.assertEqual(self.executed, [], "an unclassified tool must not run before the turn is confirmed")
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await task
        self.assertEqual(len(self.executed), 1)

    async def test_register_effect_restores_speculative_fast_path(self):
        register_effect("get_route_test_only", EffectClass.READ_ONLY)
        try:
            d = self.make()
            task = asyncio.create_task(d.dispatch(ToolCall("get_route_test_only", {}, EffectClass.READ_ONLY)))
            await self.settle()
            self.assertEqual(len(self.executed), 1, "registered read-only tools still run speculatively")
            await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
            await task
        finally:
            dict.pop(EFFECT_MAP, "get_route_test_only", None)


class TestGateDrivesDispatcher(DispatcherCase):
    """End-to-end: the real TRPGate epoch decides supersede-vs-keep."""

    async def test_spoken_repair_supersedes_but_plain_second_call_does_not(self):
        gate = TRPGate()
        d = ToolDispatcher(
            registry_fn=lambda n, **k: {"status": "success", **k}, saga=self.saga,
            tool_log_path=self.log_path, room_name="room-1",
            correction_epoch_fn=lambda: gate.correction_epoch,
        )
        # 1) real repair: "... Rome ... wait, actually Oslo"
        t1 = asyncio.create_task(d.dispatch(self.call("search_flights", destination="Rome", date="May 1")))
        await self.settle()
        await gate.on_transcript_token("wait, actually make that Oslo", is_final=False)
        t2 = asyncio.create_task(d.dispatch(self.call("search_flights", destination="Oslo", date="May 1")))
        await self.settle()
        self.assertEqual(json.loads(await t1)["status"], "superseded")
        # 2) no repair: two legitimate track_order calls
        t3 = asyncio.create_task(d.dispatch(self.call("track_order", order_id="A1")))
        t4 = asyncio.create_task(d.dispatch(self.call("track_order", order_id="B2")))
        await self.settle()
        await d.on_trp_state_change(TRPState.TRP_CONFIRMED)
        await t2, await t3, await t4
        self.assertEqual(
            sorted((c["function"], tuple(c["args"].values())[0]) for c in self.logged()),
            [("search_flights", "Oslo"), ("track_order", "A1"), ("track_order", "B2")],
        )


if __name__ == "__main__":
    unittest.main()
