"""Every internal event factory preserves its lifecycle payload."""

from module3.runtime.events import internal_events as events
from module3.runtime.events.base import EventCategory, InternalEventType


def test_all_internal_event_factories():
    made = [
        events.make_task_created("s", 1, "t", "work", 1, parent_task_id="p", call_id="c"),
        events.make_task_started("s", 2, "t", 1),
        events.make_task_completed("s", 3, "t", 1),
        events.make_task_cancel_requested("s", 4, "t", 1, "manual"),
        events.make_task_cancelled("s", 5, "t", 1, "waiting"),
        events.make_task_failed("s", 6, "t", 1, "boom"),
        events.make_stale_result_discarded("s", 7, "t", "c", 1, 2),
        events.make_generation_invalidated("s", 8, 1, 2),
        events.make_session_created("s", 9),
        events.make_session_closed("s", 10),
        events.make_queue_overflow("s", 11, "e", "TEXT_CHUNK", "s", 1),
        events.make_duplicate_call_id("s", 12, "call-abc", "book_flight", 1),
    ]
    assert all(event.category == EventCategory.INTERNAL for event in made)
    assert [event.event_type for event in made] == list(InternalEventType)
    assert made[0].payload["task_type"] == "work"
    assert made[3].payload["reason"] == "manual"
    assert made[5].payload["error"] == "boom"
    assert made[-2].payload["dropped_event_id"] == "e"
    assert made[-1].call_id == "call-abc"
