import unittest
import asyncio
from prism import CompensationEntry
from prism.saga_coordinator import SagaCoordinator


class TestSagaCoordinator(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.saga = SagaCoordinator()
        self.executed = []

    async def _dummy_compensate(self, name: str):
        self.executed.append(name)

    async def test_reverse_order_compensation(self):
        """Compensations must execute in strict LIFO order (Garcia-Molina & Salem 1987)."""
        entry1 = CompensationEntry(
            call_id="call-001",
            tool_name="book_flight",
            compensation_fn=lambda: self._dummy_compensate("cancel_flight"),
            args={"passenger_name": "Alice"},
        )
        entry2 = CompensationEntry(
            call_id="call-002",
            tool_name="add_to_cart",
            compensation_fn=lambda: self._dummy_compensate("remove_from_cart"),
            args={"product_id": "P123", "quantity": 1},
        )
        entry3 = CompensationEntry(
            call_id="call-003",
            tool_name="modify_autopay",
            compensation_fn=lambda: self._dummy_compensate("revert_autopay"),
            args={"bill_type": "electric"},
        )

        self.saga.register(entry1)
        self.saga.register(entry2)
        self.saga.register(entry3)

        self.assertTrue(self.saga.has_committed_mutations)

        compensated = await self.saga.compensate_all()
        self.assertEqual(compensated, ["call-003", "call-002", "call-001"])
        self.assertEqual(
            self.executed,
            ["revert_autopay", "remove_from_cart", "cancel_flight"]
        )
        self.assertFalse(self.saga.has_committed_mutations)

    async def test_fault_tolerant_compensation_isolation(self):
        """Failure in one compensation action must not abort remaining compensations."""
        async def failing_comp():
            raise RuntimeError("Network timeout during refund")

        entry1 = CompensationEntry(
            call_id="call-001",
            tool_name="book_flight",
            compensation_fn=lambda: self._dummy_compensate("cancel_flight"),
        )
        entry2 = CompensationEntry(
            call_id="call-002",
            tool_name="failing_tool",
            compensation_fn=failing_comp,
        )
        entry3 = CompensationEntry(
            call_id="call-003",
            tool_name="add_to_cart",
            compensation_fn=lambda: self._dummy_compensate("remove_from_cart"),
        )

        self.saga.register(entry1)
        self.saga.register(entry2)
        self.saga.register(entry3)

        compensated = await self.saga.compensate_all()
        # call-002 failed, but call-003 and call-001 must have succeeded
        self.assertEqual(compensated, ["call-003", "call-001"])
        self.assertEqual(self.executed, ["remove_from_cart", "cancel_flight"])

    def test_clear_for_new_turn(self):
        """clear_for_new_turn flushes all logged entries."""
        entry = CompensationEntry(
            call_id="call-001",
            tool_name="book_flight",
            compensation_fn=lambda: self._dummy_compensate("cancel"),
        )
        self.saga.register(entry)
        self.assertTrue(self.saga.has_committed_mutations)
        self.saga.clear_for_new_turn()
        self.assertFalse(self.saga.has_committed_mutations)


if __name__ == "__main__":
    unittest.main()
