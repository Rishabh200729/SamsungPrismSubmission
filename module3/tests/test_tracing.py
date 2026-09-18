"""Trace recorder export and query coverage."""

import json

from module3.runtime.events import make_text_chunk
from module3.runtime.tracing.recorder import TraceRecorder


def test_recorder_records_queries_exports_and_clear(tmp_path):
    recorder = TraceRecorder(record_wall_clock=True)
    event = make_text_chunk("s1", 10, "hello", task_id="t1", metadata={"kind": "input"})
    entry = recorder.record_event(event, source="test", extra_metadata={"extra": True})
    transition = recorder.record_transition("s1", 20, "TASK_STARTED", task_id="t1", metadata={"phase": "run"})
    recorder.record_transition("s2", 30, "TASK_STARTED", task_id="t2")

    assert (entry.seq, transition.seq) == (1, 2)
    assert entry.wall_clock_ms is not None and entry.metadata["payload_keys"] == ["text", "is_partial"]
    assert recorder.entries_for_session("s1") == [entry, transition]
    assert recorder.entries_for_task("t1") == [entry, transition]
    assert recorder.entries_of_type("TASK_STARTED") == [transition, recorder.entries()[2]]
    assert recorder.first_of_type("TASK_STARTED", "s1") == transition
    assert recorder.count_of_type("TASK_STARTED") == 2

    json_path = tmp_path / "nested" / "trace.json"
    jsonl_path = tmp_path / "nested" / "trace.jsonl"
    recorder.export_json(json_path)
    recorder.export_jsonl(jsonl_path)
    assert json.loads(json_path.read_text())[0]["entry_type"] == "TEXT_CHUNK"
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert [line["seq"] for line in lines] == [1, 2, 3]

    recorder.clear()
    assert recorder.total_entries == 0 and recorder.entries() == []
