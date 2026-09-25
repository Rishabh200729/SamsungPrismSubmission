"""
prism/saga_coordinator.py
ADR 03 / ADR 04 (shared)

Saga Rollback Coordinator — Garcia-Molina & Salem (1987).
Maintains a per-session append-only log of committed mutations with their
compensation handlers.  On abort (barge-in or REPAIRING timeout) executes
compensations in reverse-dependency order.

Independence: no LiveKit imports. Fully unit-testable with mock compensation
closures; no audio or network required.
"""

from __future__ import annotations

import logging

from prism import CompensationEntry

log = logging.getLogger("prism.saga")


class SagaCoordinator:
    """
    Per-session saga log.

    Lifecycle:
        session starts → SagaCoordinator()
        tool committed → register(entry)
        barge-in/abort → await compensate_all()
        turn boundary  → clear_for_new_turn()
    """

    def __init__(self) -> None:
        # Append-only log of committed mutations this turn.
        # Never modified by more than one coroutine simultaneously
        # (asyncio single-thread guarantee).
        self._log: List[CompensationEntry] = []

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register(self, entry: CompensationEntry) -> None:
        """
        Called by ToolDispatcher immediately after a COMPENSABLE or IRREVERSIBLE
        mutation is committed.  Appends entry to the log.
        """
        self._log.append(entry)
        log.debug(
            "saga: registered compensation for %s (call_id=%s)",
            entry.tool_name, entry.call_id[:8],
        )

    # -----------------------------------------------------------------------
    # Compensation
    # -----------------------------------------------------------------------

    async def compensate_all(self) -> list[str]:
        """
        Execute compensation for every un-compensated entry in REVERSE insertion
        order (Garcia-Molina & Salem 1987 reverse-dependency rule).

        Returns list of compensated call_ids for audit logging.
        Does NOT raise on individual compensation failure — logs error and continues
        so that a failing C_i doesn't prevent earlier ones from running.
        """
        compensated_ids: list[str] = []
        pending = [e for e in reversed(self._log) if not e.compensated]

        if not pending:
            log.debug("saga: compensate_all called but nothing to compensate")
            return compensated_ids

        log.info("saga: compensating %d committed mutation(s) in reverse order", len(pending))

        for entry in pending:
            try:
                await entry.compensation_fn()
                entry.compensated = True
                compensated_ids.append(entry.call_id)
                log.info("saga: compensated %s (call_id=%s)", entry.tool_name, entry.call_id[:8])
            except Exception as exc:
                # Log and continue — don't let one failure block others
                log.error(
                    "saga: compensation for %s (call_id=%s) FAILED: %s",
                    entry.tool_name, entry.call_id[:8], exc,
                )

        return compensated_ids

    # -----------------------------------------------------------------------
    # Turn boundary
    # -----------------------------------------------------------------------

    def clear_for_new_turn(self) -> None:
        """
        Called by lk_agent_tool.py at the start of each new user turn.
        Previous turn's committed mutations are no longer eligible for rollback.
        """
        if self._log:
            log.debug("saga: clearing %d log entries for new turn", len(self._log))
        self._log.clear()

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    @property
    def has_committed_mutations(self) -> bool:
        """True if any un-compensated COMPENSABLE/IRREVERSIBLE calls are logged."""
        return any(not e.compensated for e in self._log)

    def __repr__(self) -> str:
        return (
            f"SagaCoordinator(entries={len(self._log)}, "
            f"pending={sum(1 for e in self._log if not e.compensated)})"
        )
