"""Contract tests for competitive and non-competitive interruption events."""

from __future__ import annotations

import asyncio

import pytest

from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.events import InputEventType, make_interruption
from module3.runtime.tasks.lifecycle import TaskStatus


async def _wait_for(predicate, attempts: int = 20) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate()


@pytest.mark.asyncio
async def test_non_competitive_interruption_does_not_cancel():
    runtime = Runtime(config=TEST_CONFIG)
    blocker = asyncio.Event()
    stopped = False
    await runtime.start()
    session_id = runtime.create_session()
    try:
        task_id = await runtime.create_background_task(
            session_id, blocker.wait(), "backchannel-survival"
        )
        await _wait_for(
            lambda: runtime.get_task(session_id, task_id).status == TaskStatus.RUNNING
        )

        await runtime.submit_event(
            make_interruption(session_id, 1, text="uh-huh", competitive=False)
        )
        # stop() waits for the dispatcher loop.  This proves the core handler
        # has completed, rather than merely observing that get() removed the
        # event from the queue.
        await runtime.stop()
        stopped = True

        assert runtime.get_current_generation(session_id) == 1
        assert runtime.get_task(session_id, task_id).status == TaskStatus.RUNNING
    finally:
        blocker.set()
        await asyncio.sleep(0)
        if not stopped:
            await runtime.stop()


@pytest.mark.asyncio
async def test_competitive_interruption_still_cancels():
    runtime = Runtime(config=TEST_CONFIG)
    blocker = asyncio.Event()
    await runtime.start()
    session_id = runtime.create_session()
    try:
        task_id = await runtime.create_background_task(
            session_id, blocker.wait(), "competitive-interruption"
        )
        await _wait_for(
            lambda: runtime.get_task(session_id, task_id).status == TaskStatus.RUNNING
        )

        await runtime.submit_event(make_interruption(session_id, 1, text="stop"))
        await _wait_for(lambda: runtime.get_current_generation(session_id) == 2)

        assert runtime.get_task(session_id, task_id).status in {
            TaskStatus.CANCELLATION_REQUESTED,
            TaskStatus.CANCELLED,
        }
    finally:
        blocker.set()
        await runtime.stop()


@pytest.mark.asyncio
async def test_module1_handler_fires_for_both_interruption_kinds():
    runtime = Runtime(config=TEST_CONFIG)
    received: list[bool] = []

    async def module1_on_interruption(event):
        received.append(event.payload["competitive"])

    runtime.register_handler(InputEventType.INTERRUPTION, module1_on_interruption)
    await runtime.start()
    session_id = runtime.create_session()
    try:
        await runtime.submit_event(make_interruption(session_id, 1, competitive=False))
        await runtime.submit_event(make_interruption(session_id, 2, competitive=True))
        await _wait_for(lambda: len(received) == 2)
    finally:
        await runtime.stop()

    assert received == [False, True]
