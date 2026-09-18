"""module3/tests/conftest.py — shared pytest fixtures."""
import pytest
import asyncio
from module3.runtime.runtime import Runtime
from module3.runtime.config import TEST_CONFIG
from module3.runtime.clock.virtual_clock import VirtualClock


@pytest.fixture
def virtual_clock():
    return VirtualClock(start_ms=0.0)


@pytest.fixture
async def runtime(virtual_clock):
    rt = Runtime(config=TEST_CONFIG, clock=virtual_clock)
    await rt.start()
    yield rt
    await rt.stop()


@pytest.fixture
def session_id(runtime):
    return runtime.create_session()
