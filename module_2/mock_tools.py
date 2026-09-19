"""
mock_tools.py
Local stand-ins for real tool APIs, so the loop (call -> execute -> result
-> final response) can be tested end-to-end before real integrations
exist. Swap TOOL_REGISTRY's entries for real implementations (or for
Rishabh's module3/mocks/tools.py, if that's the org-standard mock set)
without touching tool_executor.py or orchestrator.py.
"""

import asyncio
import random
from typing import Dict, Any


async def book_flight(arguments: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.05)
    return {"booking_id": f"FL-{random.randint(1000, 9999)}", "status": "confirmed", **arguments}


async def book_hotel(arguments: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.05)
    return {"booking_id": f"HT-{random.randint(1000, 9999)}", "status": "confirmed", **arguments}


async def get_flight_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"status": random.choice(["on_time", "delayed"]), **arguments}


TOOL_REGISTRY = {
    "book_flight": book_flight,
    "book_hotel": book_hotel,
    "get_flight_status": get_flight_status,
}
