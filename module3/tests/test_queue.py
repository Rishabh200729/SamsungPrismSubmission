"""
module3/tests/test_queue.py

Tests for InputQueue and OutputQueue.
"""
import asyncio
import pytest

from module3.runtime.queues import InputQueue, OutputQueue
from module3.runtime.events import make_text_chunk, make_filler, make_final_response
from module3.runtime.events.base import OutputEventType

SESSION = "q-test-session"


class TestInputQueue:
    async def test_put_and_get(self):
        q = InputQueue()
        e = make_text_chunk(SESSION, 0.0, "hello")
        await q.put(e)
        result = await q.get()
        assert result.event_id == e.event_id

    async def test_fifo_ordering(self):
        q = InputQueue()
        events = [make_text_chunk(SESSION, float(i), f"msg{i}") for i in range(5)]
        for e in events:
            await q.put(e)
        for expected in events:
            got = await q.get()
            assert got.event_id == expected.event_id
            q.task_done()

    async def test_qsize(self):
        q = InputQueue()
        assert q.qsize() == 0
        await q.put(make_text_chunk(SESSION, 0.0, "a"))
        await q.put(make_text_chunk(SESSION, 1.0, "b"))
        assert q.qsize() == 2

    async def test_empty(self):
        q = InputQueue()
        assert q.empty()
        await q.put(make_text_chunk(SESSION, 0.0, "x"))
        assert not q.empty()

    async def test_close_raises_on_put(self):
        q = InputQueue()
        await q.close()
        with pytest.raises(RuntimeError, match="closed"):
            await q.put(make_text_chunk(SESSION, 0.0, "x"))

    async def test_concurrent_producers(self):
        q = InputQueue()
        results = []

        async def producer(n: int):
            for i in range(3):
                await q.put(make_text_chunk(SESSION, float(n * 10 + i), f"p{n}_m{i}"))

        async def consumer():
            for _ in range(6):
                e = await q.get()
                results.append(e.event_id)
                q.task_done()

        await asyncio.gather(producer(0), producer(1), consumer())
        assert len(results) == 6

    async def test_put_nowait(self):
        q = InputQueue()
        e = make_text_chunk(SESSION, 0.0, "now")
        q.put_nowait(e)
        result = await q.get()
        assert result.event_id == e.event_id

    async def test_maxsize_backpressure(self):
        q = InputQueue(maxsize=1)
        await q.put(make_text_chunk(SESSION, 0.0, "first"))
        assert q.full()
        with pytest.raises(asyncio.QueueFull):
            q.put_nowait(make_text_chunk(SESSION, 1.0, "second"))

    async def test_drop_oldest_is_opt_in_and_preserves_newest_event(self):
        q = InputQueue(maxsize=1, drop_oldest=True)
        first = make_text_chunk(SESSION, 0.0, "first")
        second = make_text_chunk(SESSION, 1.0, "second")
        assert await q.put(first) is None
        assert await q.put(second) == first
        assert q.dropped_count == 1
        assert (await q.get()).event_id == second.event_id

    async def test_input_queue_nowait_empty_join_and_async_iteration(self):
        q = InputQueue()
        with pytest.raises(asyncio.QueueEmpty):
            q.get_nowait()
        q.put_nowait(make_text_chunk(SESSION, 0, "iterated"))
        await q.close()
        assert q.is_closed
        assert [event.payload["text"] async for event in q] == ["iterated"]
        await q.join()
        with pytest.raises(RuntimeError):
            q.put_nowait(make_text_chunk(SESSION, 1, "closed"))


class TestOutputQueue:
    async def test_emit_and_next(self):
        q = OutputQueue()
        e = make_filler(SESSION, 0.0, "One moment...")
        await q.emit(e)
        result = await q.next_output()
        assert result.event_id == e.event_id

    async def test_drain_all(self):
        q = OutputQueue()
        for i in range(5):
            await q.emit(make_filler(SESSION, float(i), f"filler {i}"))
        results = await q.drain_all()
        assert len(results) == 5
        assert q.empty()

    async def test_total_emitted_counter(self):
        q = OutputQueue()
        assert q.total_emitted == 0
        await q.emit(make_filler(SESSION, 0.0, "x"))
        await q.emit(make_filler(SESSION, 1.0, "y"))
        assert q.total_emitted == 2

    async def test_collect_until_idle(self):
        q = OutputQueue()
        await q.emit(make_filler(SESSION, 0.0, "a"))
        await q.emit(make_filler(SESSION, 1.0, "b"))
        results = await q.collect_until_idle(timeout_ms=50)
        assert len(results) == 2

    async def test_priority_is_stable_and_filtering_keeps_nonmatching_events(self):
        q = OutputQueue()
        normal = make_final_response("other", 1.0, "normal")
        high_first = make_filler(SESSION, 2.0, "first")
        high_second = make_filler(SESSION, 3.0, "second")
        await q.emit(normal, priority=0)
        await q.emit(high_first, priority=5)
        await q.emit(high_second, priority=5)

        filtered = await q.next_output(event_type=OutputEventType.FINAL_RESPONSE)
        assert filtered.event_id == normal.event_id
        assert (await q.next_output()).event_id == high_first.event_id
        assert (await q.next_output()).event_id == high_second.event_id

    async def test_session_filter_timeout_nowait_and_close_iteration(self):
        q = OutputQueue(maxsize=1)
        event = make_filler("a", 0, "a")
        q.emit_nowait(event)
        with pytest.raises(asyncio.QueueFull):
            q.emit_nowait(make_filler("b", 1, "b"))
        with pytest.raises(asyncio.TimeoutError):
            await q.next_output(session_id="b", timeout_ms=1)
        assert (await q.next_output(session_id="a")).event_id == event.event_id
        await q.close()
        with pytest.raises(RuntimeError):
            await q.emit(make_filler("a", 2, "closed"))
        assert [item async for item in q] == []

    async def test_emit_nowait_wakes_waiting_consumer(self):
        q = OutputQueue()
        waiting = asyncio.create_task(q.next_output())
        await asyncio.sleep(0)
        event = make_filler(SESSION, 0, "wake")
        q.emit_nowait(event)
        assert (await waiting).event_id == event.event_id
