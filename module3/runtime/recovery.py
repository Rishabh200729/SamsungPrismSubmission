"""
module3/runtime/recovery.py

Bounded recovery policy — §8 of the orchestration kernel spec.

Recovery tiers (in priority order, lowest-risk first)
------------------------------------------------------
1. READ_ONLY transient failure  → bounded retry with exponential back-off.
2. Validation / schema failure  → clarification request or local argument repair.
3. Stale / superseded call      → discard, no retry.
4. Mutating-call uncertainty    → query status first; never re-issue the write.
5. Full replan                  → only after the active generation is stable
                                   and all lower-risk tiers have been exhausted.

RetryRecord
-----------
Every retry attempt is logged with:
  original_call_id, new_call_id, retry_number, reason,
  idempotency_key, generation, snapshot_version.

Usage::

    policy = RecoveryPolicy(max_read_retries=3, base_backoff_ms=200.0)

    result = await policy.handle_read_failure(
        call_fn=my_tool_coro,
        call_id="c1",
        session_id=sid,
        generation=current_gen,
        snapshot_version=sv,
        reason="timeout",
    )

    ok = await policy.handle_write_uncertainty(
        status_fn=check_booking_status,
        call_id="c1",
        session_id=sid,
        generation=current_gen,
        snapshot_version=sv,
        idempotency_key="idem-123",
    )
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RetryRecord — audit trail for every retry attempt
# ---------------------------------------------------------------------------

class RecoveryReason(str, Enum):
    TRANSIENT_FAILURE = "transient_failure"
    VALIDATION_ERROR = "validation_error"
    STALE_SUPERSEDED = "stale_superseded"
    WRITE_UNCERTAINTY = "write_uncertainty"
    REPLAN = "replan"


@dataclass
class RetryRecord:
    """
    Immutable audit record for one retry attempt.

    Stored in RecoveryPolicy.retry_log for evaluation / trace export.
    """
    original_call_id: str
    new_call_id: str
    retry_number: int
    reason: RecoveryReason
    idempotency_key: str | None
    generation: int
    snapshot_version: int
    timestamp_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RecoveryPolicy
# ---------------------------------------------------------------------------

class RecoveryPolicy:
    """
    Stateless-by-design recovery policy with an internal retry audit log.

    The log is append-only and available for inspection by the evaluator.
    """

    def __init__(
        self,
        max_read_retries: int = 3,
        base_backoff_ms: float = 200.0,
        backoff_multiplier: float = 2.0,
        max_backoff_ms: float = 5000.0,
        clock_now: Callable[[], float] | None = None,
    ) -> None:
        self._max_read_retries = max_read_retries
        self._base_backoff_ms = base_backoff_ms
        self._backoff_multiplier = backoff_multiplier
        self._max_backoff_ms = max_backoff_ms
        self._now = clock_now or (lambda: 0.0)
        self.retry_log: list[RetryRecord] = []

    # ------------------------------------------------------------------
    # Tier 1: READ_ONLY transient failure
    # ------------------------------------------------------------------

    async def handle_read_failure(
        self,
        call_fn: Callable[[], Awaitable[Any]],
        *,
        call_id: str,
        session_id: str,
        generation: int,
        snapshot_version: int,
        reason: str = "transient_failure",
        idempotency_key: str | None = None,
    ) -> Any:
        """
        Retry a read-only call with exponential back-off.

        Returns the successful result or re-raises the last exception after
        max_read_retries attempts are exhausted.
        """
        backoff_ms = self._base_backoff_ms
        last_exc: Exception | None = None
        for attempt in range(1, self._max_read_retries + 2):  # +1 for the initial attempt
            if attempt > 1:
                new_call_id = str(uuid.uuid4())
                record = RetryRecord(
                    original_call_id=call_id,
                    new_call_id=new_call_id,
                    retry_number=attempt - 1,
                    reason=RecoveryReason.TRANSIENT_FAILURE,
                    idempotency_key=idempotency_key,
                    generation=generation,
                    snapshot_version=snapshot_version,
                    timestamp_ms=self._now(),
                    metadata={"reason": reason},
                )
                self.retry_log.append(record)
                logger.info(
                    "Recovery[READ]: retry=%d call_id=%s -> %s session=%s",
                    attempt - 1, call_id, new_call_id, session_id,
                )
                await asyncio.sleep(min(backoff_ms, self._max_backoff_ms) / 1000.0)
                backoff_ms *= self._backoff_multiplier
            try:
                return await call_fn()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Recovery[READ]: attempt=%d failed: %r session=%s call_id=%s",
                    attempt, exc, session_id, call_id,
                )
                if attempt > self._max_read_retries:
                    break
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Tier 2: Validation / schema failure
    # ------------------------------------------------------------------

    async def handle_validation_failure(
        self,
        *,
        call_id: str,
        session_id: str,
        generation: int,
        snapshot_version: int,
        error: str,
        repaired_args: dict[str, Any] | None = None,
        clarification_fn: Callable[[], Awaitable[Any]] | None = None,
    ) -> Any | None:
        """
        Handle a schema/validation error.

        If repaired_args are provided the repair is logged as a retry.
        Otherwise, if clarification_fn is provided it is called to request
        clarification from the user.
        Returns the result of the clarification_fn or None.
        """
        record = RetryRecord(
            original_call_id=call_id,
            new_call_id=str(uuid.uuid4()),
            retry_number=1,
            reason=RecoveryReason.VALIDATION_ERROR,
            idempotency_key=None,
            generation=generation,
            snapshot_version=snapshot_version,
            timestamp_ms=self._now(),
            metadata={"validation_error": error, "repaired": repaired_args is not None},
        )
        self.retry_log.append(record)
        logger.info(
            "Recovery[VALIDATION]: call_id=%s repaired=%s session=%s error=%s",
            call_id, repaired_args is not None, session_id, error,
        )
        if clarification_fn is not None:
            return await clarification_fn()
        return None

    # ------------------------------------------------------------------
    # Tier 3: Stale / superseded call
    # ------------------------------------------------------------------

    def handle_stale_call(
        self,
        *,
        call_id: str,
        session_id: str,
        generation: int,
        snapshot_version: int,
        reason: str = "superseded",
    ) -> None:
        """
        Record and discard a stale/superseded call. No retry is issued.
        """
        record = RetryRecord(
            original_call_id=call_id,
            new_call_id=call_id,  # same — no new call
            retry_number=0,
            reason=RecoveryReason.STALE_SUPERSEDED,
            idempotency_key=None,
            generation=generation,
            snapshot_version=snapshot_version,
            timestamp_ms=self._now(),
            metadata={"reason": reason},
        )
        self.retry_log.append(record)
        logger.info(
            "Recovery[STALE]: call_id=%s discarded (no retry) session=%s reason=%s",
            call_id, session_id, reason,
        )

    # ------------------------------------------------------------------
    # Tier 4: Mutating-call uncertainty (idempotent status check)
    # ------------------------------------------------------------------

    async def handle_write_uncertainty(
        self,
        status_fn: Callable[[], Awaitable[str]],
        *,
        call_id: str,
        session_id: str,
        generation: int,
        snapshot_version: int,
        idempotency_key: str,
    ) -> str:
        """
        When a write call timed out and may have committed, query its status
        before deciding whether to re-issue.

        Returns the status string from status_fn (e.g. "COMMITTED", "PENDING",
        "NOT_FOUND").  The caller must NOT re-issue the write if status is
        "COMMITTED".
        """
        record = RetryRecord(
            original_call_id=call_id,
            new_call_id=str(uuid.uuid4()),
            retry_number=1,
            reason=RecoveryReason.WRITE_UNCERTAINTY,
            idempotency_key=idempotency_key,
            generation=generation,
            snapshot_version=snapshot_version,
            timestamp_ms=self._now(),
            metadata={"action": "status_check_before_reissue"},
        )
        self.retry_log.append(record)
        logger.info(
            "Recovery[WRITE_UNCERTAINTY]: querying status before reissue call_id=%s session=%s",
            call_id, session_id,
        )
        status = await status_fn()
        logger.info(
            "Recovery[WRITE_UNCERTAINTY]: status=%s call_id=%s session=%s",
            status, call_id, session_id,
        )
        return status

    # ------------------------------------------------------------------
    # Tier 5: Full replan
    # ------------------------------------------------------------------

    async def handle_replan(
        self,
        replan_fn: Callable[[], Awaitable[Any]],
        *,
        call_id: str,
        session_id: str,
        generation: int,
        snapshot_version: int,
        reason: str = "exhausted_lower_tiers",
    ) -> Any:
        """
        Trigger a full replan as the last resort.

        Should only be called after lower-risk tiers have failed and the
        generation is confirmed stable (no pending interruption).
        """
        record = RetryRecord(
            original_call_id=call_id,
            new_call_id=str(uuid.uuid4()),
            retry_number=1,
            reason=RecoveryReason.REPLAN,
            idempotency_key=None,
            generation=generation,
            snapshot_version=snapshot_version,
            timestamp_ms=self._now(),
            metadata={"reason": reason},
        )
        self.retry_log.append(record)
        logger.info(
            "Recovery[REPLAN]: triggering full replan call_id=%s session=%s reason=%s",
            call_id, session_id, reason,
        )
        return await replan_fn()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def retries_for_call(self, original_call_id: str) -> list[RetryRecord]:
        return [r for r in self.retry_log if r.original_call_id == original_call_id]

    def total_retries(self) -> int:
        return len(self.retry_log)
