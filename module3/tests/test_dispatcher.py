"""
module3/tests/test_dispatcher.py

Tests for the Dispatcher event router.
"""
import asyncio
import pytest

from module3.runtime.dispatcher import Dispatcher
from module3.runtime.events import make_text_chunk, make_interruption
from module3.runtime.events.base import InputEventType

SESSION = "disp-session"


class TestDispatcher:
    async def test_register_and_dispatch(self):
        dispatcher = Dispatcher()
        received = []

        async def handler(event):
            received.append(event.event_id)

        dispatcher.register_handler(InputEventType.TEXT_CHUNK, handler)
        e = make_text_chunk(SESSION, 0.0, "hello")
        await dispatcher.dispatch(e)
        assert len(received) == 1
        assert received[0] == e.event_id

    async def test_handler_must_be_async(self):
        dispatcher = Dispatcher()

        def sync_handler(event):
            pass

        with pytest.raises(TypeError, match="async def"):
            dispatcher.register_handler(InputEventType.TEXT_CHUNK, sync_handler)

    async def test_no_handler_no_error(self):
        dispatcher = Dispatcher()
        e = make_text_chunk(SESSION, 0.0, "hello")
        # Should not raise even with no registered handlers
        await dispatcher.dispatch(e)
        assert dispatcher.dispatch_count == 1

    async def test_multiple_handlers_fan_out(self):
        dispatcher = Dispatcher()
        calls = []

        async def h1(event): calls.append("h1")
        async def h2(event): calls.append("h2")
        async def h3(event): calls.append("h3")

        dispatcher.register_handler(InputEventType.TEXT_CHUNK, h1)
        dispatcher.register_handler(InputEventType.TEXT_CHUNK, h2)
        dispatcher.register_handler(InputEventType.TEXT_CHUNK, h3)

        await dispatcher.dispatch(make_text_chunk(SESSION, 0.0, "x"))
        assert sorted(calls) == ["h1", "h2", "h3"]

    async def test_handler_error_does_not_crash_dispatch(self):
        dispatcher = Dispatcher()
        good_calls = []

        async def bad_handler(event):
            raise RuntimeError("handler error")

        async def good_handler(event):
            good_calls.append(event.event_id)

        dispatcher.register_handler(InputEventType.TEXT_CHUNK, bad_handler)
        dispatcher.register_handler(InputEventType.TEXT_CHUNK, good_handler)

        e = make_text_chunk(SESSION, 0.0, "test")
        await dispatcher.dispatch(e)  # Should not raise
        assert len(good_calls) == 1   # Good handler still ran
        assert dispatcher.error_count == 1

    async def test_wildcard_handler_receives_all_events(self):
        dispatcher = Dispatcher()
        all_events = []

        async def wildcard(event):
            all_events.append(event.event_type.value)

        dispatcher.register_wildcard_handler(wildcard)
        await dispatcher.dispatch(make_text_chunk(SESSION, 0.0, "hello"))
        await dispatcher.dispatch(make_interruption(SESSION, 100.0))

        assert "TEXT_CHUNK" in all_events
        assert "INTERRUPTION" in all_events

    async def test_dispatch_count_increments(self):
        dispatcher = Dispatcher()
        async def noop(event): pass
        dispatcher.register_handler(InputEventType.TEXT_CHUNK, noop)

        for _ in range(5):
            await dispatcher.dispatch(make_text_chunk(SESSION, 0.0, "x"))
        assert dispatcher.dispatch_count == 5

    async def test_unregister_handler(self):
        dispatcher = Dispatcher()
        calls = []

        async def handler(event): calls.append(1)
        dispatcher.register_handler(InputEventType.TEXT_CHUNK, handler)
        dispatcher.unregister_handler(InputEventType.TEXT_CHUNK, handler)

        await dispatcher.dispatch(make_text_chunk(SESSION, 0.0, "x"))
        assert len(calls) == 0

    async def test_registered_types(self):
        dispatcher = Dispatcher()
        async def h(e): pass
        dispatcher.register_handler(InputEventType.TEXT_CHUNK, h)
        dispatcher.register_handler(InputEventType.INTERRUPTION, h)
        types = dispatcher.registered_types()
        assert InputEventType.TEXT_CHUNK in types
        assert InputEventType.INTERRUPTION in types
