"""Per-session call ledger: correlation, generation, and write deduplication."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .manifest import ToolSideEffect

@dataclass
class CallRecord:
    call_id: str
    generation: int
    snapshot_version: int
    tool_name: str
    side_effect: ToolSideEffect
    idempotency_key: str | None
    status: str = "PENDING"

class CallLedger:
    def __init__(self) -> None:
        self._calls: dict[str, CallRecord] = {}
        self._writes: dict[str, str] = {}
    def register(self, record: CallRecord) -> None:
        if record.call_id in self._calls:
            raise ValueError("duplicate call_id")
        if record.side_effect is ToolSideEffect.STATE_MODIFYING:
            if not record.idempotency_key:
                raise ValueError("state-modifying calls require idempotency_key")
            if record.idempotency_key in self._writes:
                raise ValueError("duplicate state-modifying call")
            self._writes[record.idempotency_key] = record.call_id
        self._calls[record.call_id] = record
    def get(self, call_id: str) -> CallRecord | None:
        return self._calls.get(call_id)
    def supersede_before(self, generation: int) -> list[CallRecord]:
        records = [r for r in self._calls.values() if r.generation < generation and r.status == "PENDING"]
        for r in records: r.status = "SUPERSEDED"
        return records
    def finish(self, call_id: str) -> None:
        if call_id in self._calls: self._calls[call_id].status = "COMPLETED"
