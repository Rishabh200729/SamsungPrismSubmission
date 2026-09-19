"""
module1/tests/conftest.py

Shared fixtures for Module 1 tests.

All tests use VirtualClock for deterministic timing and the real Runtime
(not a mock) to ensure Module 1 integrates correctly with Module 3's actual
cancellation, generation, floor, and output-gate machinery.
"""

from __future__ import annotations

import asyncio
import pytest

from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.clock.virtual_clock import VirtualClock
from module1.fast_path.router import FastPathRouter


@pytest.fixture
def clock():
    return VirtualClock(start_ms=0.0)


@pytest.fixture
def runtime(clock):
    return Runtime(config=TEST_CONFIG, clock=clock)


@pytest.fixture
def router(runtime, clock):
    """FastPathRouter already wired into the runtime."""
    return FastPathRouter(runtime, clock)


@pytest.fixture
async def live(runtime, router):
    """
    Started runtime + router, yielded as (runtime, router).
    Automatically stopped after each test.
    """
    await runtime.start()
    yield runtime, router
    await runtime.stop()


async def _wait_for(predicate, attempts: int = 40) -> None:
    """Spin the event loop until predicate() is True or timeout."""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate(), "wait_for timed out"
