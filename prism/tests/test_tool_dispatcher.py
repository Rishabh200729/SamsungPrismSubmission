import unittest
import asyncio
import json
import os
import tempfile
from prism import EffectClass, ToolCall, TRPState
from prism.saga_coordinator import SagaCoordinator
from prism.tool_dispatcher import ToolDispatcher, EFFECT_MAP


class TestToolDispatcher(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        self.temp_log_path = self.temp_log.name
        self.temp_log.close()

        self.mock_db = {
            "flights": [],
            "cart": [],
            "identity": {},
        }
        self.call_history = []

        def mock_registry_call(name: str, **kwargs):
            self.call_history.append((name, kwargs))
            if name == "search_flights":
                return {"status": "success", "destination": kwargs.get("destination"), "date": kwargs.get("date")}
            elif name == "book_flight":
                booking = {"ref": "B999", "passenger": kwargs.get("passenger_name")}
                self.mock_db["flights"].append(booking)
                return {"status": "success", "booking": booking}
            elif name == "add_to_cart":
                item = {"product_id": kwargs.get("product_id"), "qty": kwargs.get("quantity")}
                self.mock_db["cart"].append(item)
                return {"status": "success", "item": item}
            elif name == "update_identity_doc":
                self.mock_db["identity"] = kwargs
                return {"status": "success", "doc": kwargs}
            return {"status": "success"}

        self.saga = SagaCoordinator()
        self.dispatcher = ToolDispatcher(
            registry_fn=mock_registry_call,
            saga=self.saga,
            tool_log_path=self.temp_log_path,
            room_name="test-room-1",
        )

    def tearDown(self):
        if os.path.exists(self.temp_log_path):
            os.remove(self.temp_log_path)

    def test_effect_map_exact_12_tools(self):
        """EFFECT_MAP must classify the 12 tools from benchmark_data_v2.json plus paper aliases."""
        canonical_12 = [
            "search_flights", "get_exchange_rate", "get_card_benefits",
            "search_apartments", "calculate_commute", "search_products", "track_order",
            "book_flight", "modify_autopay", "update_search_filter", "add_to_cart",
            "update_identity_doc"
        ]
        for name in canonical_12:
            self.assertIn(name, EFFECT_MAP, f"Canonical tool {name} missing from EFFECT_MAP")

        # 7 Read only canonical
        read_only = [k for k in canonical_12 if EFFECT_MAP[k] == EffectClass.READ_ONLY]
        self.assertEqual(len(read_only), 7)

        # 4 Compensable canonical
        compensable = [k for k in canonical_12 if EFFECT_MAP[k] == EffectClass.COMPENSABLE]
        self.assertEqual(len(compensable), 4)

        # 1 Irreversible canonical
        irreversible = [k for k in canonical_12 if EFFECT_MAP[k] == EffectClass.IRREVERSIBLE]
        self.assertEqual(len(irreversible), 1)

        # Also verify paper aliases map to valid effect classes
        paper_aliases = [
            "book_ticket", "calculate_currency_exchange", "modify_autopay_source",
            "query_card_benefits", "check_order_status", "update_travel_profile",
            "cancel_pending_action", "process_exchange"
        ]
        for name in paper_aliases:
            self.assertIn(name, EFFECT_MAP, f"Paper alias {name} missing from EFFECT_MAP")

    async def test_self_correction_supersedes_speculative_call(self):
        """User self-correction: earlier call is superseded before TRP; only final call commits."""
        call1 = ToolCall(
            name="search_flights",
            args={"destination": "Paris", "date": "Oct 5"},
            effect_class=EffectClass.READ_ONLY,
        )
        task1 = asyncio.create_task(self.dispatcher.dispatch(call1))
        await asyncio.sleep(0.01)  # allow task to enter provisional cache

        # User self-corrects: Oct 7 instead of Oct 5
        call2 = ToolCall(
            name="search_flights",
            args={"destination": "Paris", "date": "Oct 7"},
            effect_class=EffectClass.READ_ONLY,
        )
        task2 = asyncio.create_task(self.dispatcher.dispatch(call2))
        await asyncio.sleep(0.01)

        # First call should receive superseded sentinel
        res1_str = await task1
        res1 = json.loads(res1_str)
        self.assertEqual(res1.get("status"), "superseded")

        # Confirm TRP
        await self.dispatcher.on_trp_state_change(TRPState.TRP_CONFIRMED)
        res2_str = await task2
        res2 = json.loads(res2_str)
        self.assertEqual(res2.get("status"), "success")
        self.assertEqual(res2.get("date"), "Oct 7")

        # Telemetry log check: only ONE entry written (Oct 7), Oct 5 never written!
        with open(self.temp_log_path, "r") as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        logged_data = json.loads(lines[0])
        self.assertEqual(logged_data["call"]["args"]["date"], "Oct 7")

    async def test_exact_duplicate_mutation_commits_once(self):
        """Duplicate model emissions must not repeat a state-changing action."""
        first = ToolCall(name="add_to_cart", args={"product_id": "P9", "quantity": 2}, effect_class=EffectClass.COMPENSABLE)
        duplicate = ToolCall(name="add_to_cart", args={"product_id": "P9", "quantity": 2}, effect_class=EffectClass.COMPENSABLE)
        first_task = asyncio.create_task(self.dispatcher.dispatch(first))
        await asyncio.sleep(0.01)
        duplicate_result = json.loads(await self.dispatcher.dispatch(duplicate))

        await self.dispatcher.on_trp_state_change(TRPState.TRP_CONFIRMED)
        first_result = json.loads(await first_task)
        self.assertEqual(first_result["status"], "success")
        self.assertEqual(duplicate_result["status"], "superseded")
        self.assertEqual(self.mock_db["cart"], [{"product_id": "P9", "qty": 2}])
        with open(self.temp_log_path, "r") as f:
            self.assertEqual(len([line for line in f if line.strip()]), 1)

    async def test_reset_allows_identical_call_in_next_turn_and_preserves_history(self):
        """Deduplication is turn-scoped, not a cross-turn or telemetry filter."""
        call_args = {"destination": "Rome", "date": "Oct 7"}
        first_task = asyncio.create_task(self.dispatcher.dispatch(ToolCall(name="search_flights", args=call_args, effect_class=EffectClass.READ_ONLY)))
        await asyncio.sleep(0.01)
        await self.dispatcher.on_trp_state_change(TRPState.TRP_CONFIRMED)
        self.assertEqual(json.loads(await first_task)["status"], "success")

        self.dispatcher.reset_for_new_turn()
        second_task = asyncio.create_task(self.dispatcher.dispatch(ToolCall(name="search_flights", args=call_args, effect_class=EffectClass.READ_ONLY)))
        await asyncio.sleep(0.01)
        await self.dispatcher.on_trp_state_change(TRPState.TRP_CONFIRMED)
        self.assertEqual(json.loads(await second_task)["status"], "success")

        self.assertEqual([name for name, _ in self.call_history], ["search_flights", "search_flights"])
        with open(self.temp_log_path, "r") as f:
            self.assertEqual(len([line for line in f if line.strip()]), 2)

    async def test_compensable_mutation_staged_then_committed_with_saga(self):
        """COMPENSABLE mutations stay staged until TRP_CONFIRMED, then commit and register to Saga."""
        call = ToolCall(
            name="book_flight",
            args={"passenger_name": "Bob"},
            effect_class=EffectClass.COMPENSABLE,
        )
        task = asyncio.create_task(self.dispatcher.dispatch(call))
        await asyncio.sleep(0.01)

        # Before TRP: not executed yet!
        self.assertEqual(len(self.mock_db["flights"]), 0)
        self.assertFalse(self.saga.has_committed_mutations)

        # Confirm TRP
        await self.dispatcher.on_trp_state_change(TRPState.TRP_CONFIRMED)
        res_str = await task
        res = json.loads(res_str)
        self.assertEqual(res.get("status"), "success")

        # After TRP: executed and registered in Saga!
        self.assertEqual(len(self.mock_db["flights"]), 1)
        self.assertTrue(self.saga.has_committed_mutations)

    async def test_abort_purges_uncommitted_calls_no_telemetry_pollution(self):
        """On ABORT (barge-in), uncommitted speculative and staged calls are purged with zero telemetry."""
        call1 = ToolCall(
            name="search_flights",
            args={"destination": "Tokyo", "date": "Nov 1"},
            effect_class=EffectClass.READ_ONLY,
        )
        call2 = ToolCall(
            name="add_to_cart",
            args={"product_id": "P555", "quantity": 2},
            effect_class=EffectClass.COMPENSABLE,
        )
        task1 = asyncio.create_task(self.dispatcher.dispatch(call1))
        task2 = asyncio.create_task(self.dispatcher.dispatch(call2))
        await asyncio.sleep(0.01)

        # Trigger ABORTED
        await self.dispatcher.on_trp_state_change(TRPState.ABORTED)

        res1_str = await task1
        res2_str = await task2
        self.assertEqual(json.loads(res1_str).get("status"), "aborted")
        self.assertEqual(json.loads(res2_str).get("status"), "aborted")

        # Telemetry file must be completely empty
        if os.path.exists(self.temp_log_path):
            with open(self.temp_log_path, "r") as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 0)


if __name__ == "__main__":
    unittest.main()
