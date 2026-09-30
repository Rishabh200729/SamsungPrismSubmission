"""
prism/tool_dispatcher.py
ADR 03 — Two-Phase Transactional Tooling (TPTT)

Adapts Atomix (arXiv:2602.14849) to the voice/turn-taking domain.

What Atomix provides (our base):
  - Effect classification (bufferable / externalized-reversible / irreversible)
  - Speculative isolation (bufferable effects held until commit)
  - Saga-style compensation on abort
  - Commit gating ("delay until safety predicate satisfied")

Our voice-domain contribution (not in Atomix):
  - Safety predicate = disfluency-aware TRP gate (not epoch/frontier)
  - Integration with LiveKit VAD / streaming transcript
  - REPAIRING state suppresses commit; editing terms from human speech drive gating

EFFECT_MAP verified against:
  - benchmark_data_v2.json  (100 scenarios, 12 unique function names)
  - mock_apis.py            (MockAPIRegistry.FUNCTIONS, 12 entries)
  - lk_agent_tool.py        (AssistantFnc methods, 12 methods)

Independence: no LiveKit imports.  Pass in a callable for registry.call().
Fully unit-testable with mocked registry and mocked SagaCoordinator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional

from prism import CompensationEntry, EffectClass, ToolCall, TRPState
from prism.saga_coordinator import SagaCoordinator

log = logging.getLogger("trax.tool_dispatcher")


# ---------------------------------------------------------------------------
# Verified Effect Classification Map
# Real benchmark tool names from Lin et al. (arXiv:2604.04847 Table 1) + mock_apis.py
# ---------------------------------------------------------------------------

class _CanonicalEffectMap(dict):
    """
    Primary effect map holding the real benchmark tool names and their effect classes.
    Supports transparent resolution of runtime aliases without polluting the primary table.
    """
    _CODE_ALIASES: dict[str, str] = {
        "book_flight": "book_ticket",
        "get_exchange_rate": "calculate_currency_exchange",
        "modify_autopay": "modify_autopay_source",
        "get_card_benefits": "query_card_benefits",
        "track_order": "check_order_status",
        "update_identity_doc": "update_travel_profile",
        "search_products": "search_apartments",   # READ_ONLY
        "add_to_cart": "book_ticket",             # COMPENSABLE
    }

    def __getitem__(self, key: str) -> EffectClass:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self._CODE_ALIASES:
            return super().__getitem__(self._CODE_ALIASES[key])
        return EffectClass.COMPENSABLE

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or key in self._CODE_ALIASES

    def get(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key) or key in self._CODE_ALIASES:
            return self[key]
        return default


EFFECT_MAP: dict[str, EffectClass] = _CanonicalEffectMap({
    # READ_ONLY (Atomix: bufferable queries)
    "search_flights":              EffectClass.READ_ONLY,   # destination, date
    "calculate_currency_exchange": EffectClass.READ_ONLY,   # amount, from_currency, to_currency
    "query_card_benefits":         EffectClass.READ_ONLY,   # card_type
    "search_apartments":           EffectClass.READ_ONLY,   # city, bedrooms, max_price, pets_allowed
    "check_order_status":          EffectClass.READ_ONLY,   # order_id
    "calculate_commute":           EffectClass.READ_ONLY,   # origin_address, destination_address, mode (12th from mock_apis.py)

    # COMPENSABLE (Atomix: externalized-reversible mutations)
    "book_ticket":                 EffectClass.COMPENSABLE,   # passenger_name
    "modify_autopay_source":       EffectClass.COMPENSABLE,   # bill_type, source_account
    "update_search_filter":        EffectClass.COMPENSABLE,   # filter_name, value
    "cancel_pending_action":       EffectClass.COMPENSABLE,   # Paper Table 1 compensation primitive
    "process_exchange":            EffectClass.COMPENSABLE,   # Paper Table 1 exchange primitive

    # IRREVERSIBLE (Atomix: irreversible, require explicit TRP gate)
    "update_travel_profile":       EffectClass.IRREVERSIBLE,  # doc_type, doc_number / travel profile
})


def register_effect(tool_name: str, effect: EffectClass) -> None:
    """Register or override the effect classification for a tool name."""
    EFFECT_MAP[tool_name] = effect



# Sentinel JSON returned to the model when a speculative call is superseded
# before TRP fires (e.g., user corrects destination mid-utterance and model
# re-emits the tool with corrected params).
_SUPERSEDED = json.dumps({"status": "superseded", "note": "earlier call cancelled by correction"})
_ABORTED    = json.dumps({"status": "aborted",    "note": "call cancelled by user barge-in"})


class ToolDispatcher:
    """
    Intercepts every tool call from the model and applies transactional gating.

    Wire-up (lk_agent_tool.py)::

        saga       = SagaCoordinator()
        dispatcher = ToolDispatcher(registry_fn=registry.call, saga=saga)
        gate       = TRPGate()
        gate.add_state_listener(dispatcher.on_trp_state_change)

    Then in AssistantFnc tool methods, replace::

        result = registry.call("search_flights", ...)

    with::

        result_str = await self.dispatcher.dispatch(ToolCall(
            name="search_flights", args={...}, effect_class=EFFECT_MAP["search_flights"]
        ))
        result = json.loads(result_str)
    """

    def __init__(
        self,
        registry_fn: Callable[..., Any],    # registry.call(name, **kwargs) → dict
        saga:        SagaCoordinator,
        tool_log_path: str = "/tmp/agent_tool_calls.log",
        room_name:     str = "unknown",
        correction_epoch_fn: Optional[Callable[[], int]] = None,
        retract_committed_mutations: bool = False,
    ) -> None:
        self._registry     = registry_fn
        self._saga         = saga
        self._log_path     = tool_log_path
        self._audit_log_path = tool_log_path + ".audit"
        self._room_name    = room_name
        self._correction_epoch_fn = correction_epoch_fn
        self._retract_committed_mutations = retract_committed_mutations

        # Atomix §3: two effect stores
        self._provisional_cache: dict[str, ToolCall] = {}    # call_id → ToolCall (READ_ONLY)
        self._staged_queue:      list[ToolCall]       = []   # COMPENSABLE / IRREVERSIBLE
        self._committed_history: list[ToolCall]       = []   # committed calls this turn

        # Per-turn deduplication: tracks (name, sorted-args-json) of every dispatched call
        # Prevents double-dispatch when model emits the same tool twice in one turn.
        self._turn_call_history: list[tuple[str, str]] = []

        # Current gate state — updated by on_trp_state_change
        self._current_state: TRPState = TRPState.LISTENING

    def _should_supersede(self, earlier: ToolCall, new_call: ToolCall) -> bool:
        if earlier.name != new_call.name:
            return False
        if self._correction_epoch_fn is not None:
            return new_call.issue_epoch > earlier.issue_epoch
        return True

    # -----------------------------------------------------------------------
    # Primary dispatch — called by AssistantFnc tool methods
    # -----------------------------------------------------------------------

    async def dispatch(self, call: ToolCall) -> str:
        """
        Gate a model tool call according to the Atomix-adapted transactional model.

        Returns a JSON string to the model (same contract as the original tool methods).
        Blocks until TRP_CONFIRMED or ABORTED, which ensures the model waits for
        the user to finish speaking before receiving tool results.
        """
        if self._correction_epoch_fn is not None:
            call.issue_epoch = self._correction_epoch_fn()
        else:
            call.issue_epoch = 0

        effect = EFFECT_MAP[call.name]
        call.effect_class = effect

        # Create per-call resolution future
        loop = asyncio.get_event_loop()
        call._resolution_future = loop.create_future()

        log.info(
            "dispatcher: intercepted %s(%s) [%s] — current state=%s, epoch=%d",
            call.name, _abbrev_args(call.args), effect.name, self._current_state.name, call.issue_epoch,
        )

        # ── Per-turn deduplication ──────────────────────────────────────────
        # If we've already dispatched the exact same (name, args) this turn, skip.
        call_sig = (call.name, json.dumps(call.args, sort_keys=True))
        if call_sig in self._turn_call_history:
            log.info(
                "dispatcher: DEDUP — %s(%s) already dispatched this turn, discarding duplicate",
                call.name, _abbrev_args(call.args),
            )
            call._resolution_future.set_result(_SUPERSEDED)
            return await call._resolution_future
        self._turn_call_history.append(call_sig)

        if effect == EffectClass.READ_ONLY:
            await self._handle_read_only(call)
        else:
            await self._handle_mutating(call)

        # Block until TRP_CONFIRMED or ABORTED resolves this call's future
        result_json: str = await call._resolution_future
        return result_json

    async def _handle_read_only(self, call: ToolCall) -> None:
        """
        Speculative execution: start API call immediately, hold result in
        provisional cache.
        """
        # Supersede earlier provisional calls if repair marker heard
        for cid, existing in list(self._provisional_cache.items()):
            if self._should_supersede(existing, call):
                log.info(
                    "dispatcher: %s (epoch %d) superseded earlier provisional (epoch %d)",
                    call.name, call.issue_epoch, existing.issue_epoch,
                )
                if existing.speculative_task and not existing.speculative_task.done():
                    existing.speculative_task.cancel()
                if not existing._resolution_future.done():
                    existing._resolution_future.set_result(_SUPERSEDED)
                self._provisional_cache.pop(cid, None)

        # Retract earlier committed read-only calls if repaired
        for committed in list(self._committed_history):
            if self._should_supersede(committed, call) and committed.effect_class == EffectClass.READ_ONLY:
                log.info(
                    "dispatcher: retracting early committed read-only call %s (%s)",
                    committed.name, committed.call_id,
                )
                self._retract_telemetry(committed.call_id)
                self._committed_history.remove(committed)

        # Start speculative API call in background
        call.speculative_task = asyncio.create_task(
            self._run_api(call.name, call.args),
            name=f"speculative_{call.name}_{call.call_id[:8]}",
        )
        self._provisional_cache[call.call_id] = call

        log.debug("dispatcher: %s speculative task started", call.name)

        if self._current_state == TRPState.TRP_CONFIRMED:
            log.debug("dispatcher: TRP already confirmed, committing %s immediately", call.name)
            asyncio.create_task(self._commit_read_only(call))

    async def _handle_mutating(self, call: ToolCall) -> None:
        """
        Staged queue: COMPENSABLE and IRREVERSIBLE mutations are never executed
        while the user is speaking or in REPAIRING state.
        """
        # Supersede earlier staged calls if repair marker heard
        for staged in list(self._staged_queue):
            if self._should_supersede(staged, call):
                log.info(
                    "dispatcher: %s (epoch %d) superseded earlier staged %s (epoch %d)",
                    call.name, call.issue_epoch, staged.name, staged.issue_epoch,
                )
                if not staged._resolution_future.done():
                    staged._resolution_future.set_result(_SUPERSEDED)
                self._staged_queue.remove(staged)

        # Handle earlier committed mutations if repair marker heard
        for committed in list(self._committed_history):
            if self._should_supersede(committed, call) and committed.effect_class != EffectClass.READ_ONLY:
                if self._retract_committed_mutations:
                    log.info("dispatcher: retracting early committed mutation %s (%s)", committed.name, committed.call_id)
                    self._retract_telemetry(committed.call_id)
                    self._committed_history.remove(committed)
                else:
                    log.info("dispatcher: logging stale committed mutation event in audit log for %s", committed.name)
                    try:
                        import os
                        with open(self._audit_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "room": self._room_name,
                                "event": "stale_mutation_committed_before_repair",
                                "call_id": committed.call_id,
                                "tool": committed.name,
                                "args": committed.args,
                            }) + "\n")
                    except OSError as exc:
                        log.error("dispatcher: failed to write audit log: %s", exc)

        self._staged_queue.append(call)
        log.info(
            "dispatcher: %s [%s] staged — awaiting TRP_CONFIRMED before execution",
            call.name, call.effect_class.name,
        )

        if self._current_state == TRPState.TRP_CONFIRMED:
            log.debug("dispatcher: TRP already confirmed, executing staged %s immediately", call.name)
            asyncio.create_task(self._commit_mutating(call))

    # -----------------------------------------------------------------------
    # TRP state transition handler — wired as gate listener
    # -----------------------------------------------------------------------

    async def on_trp_state_change(self, new_state: TRPState) -> None:
        self._current_state = new_state

        if new_state == TRPState.TRP_CONFIRMED:
            await self._commit_all()

        elif new_state == TRPState.ABORTED:
            await self._abort_all()

        elif new_state == TRPState.REPAIRING:
            log.info("dispatcher: state=REPAIRING — all commits suspended")

    # -----------------------------------------------------------------------
    # Commit path (TRP_CONFIRMED)
    # -----------------------------------------------------------------------

    async def _commit_all(self) -> None:
        """Commit everything in provisional_cache then staged_queue."""
        log.info(
            "dispatcher: TRP_CONFIRMED — committing %d speculative + %d staged",
            len(self._provisional_cache), len(self._staged_queue),
        )

        provisional = list(self._provisional_cache.values())
        staged      = list(self._staged_queue)
        self._provisional_cache.clear()
        self._staged_queue.clear()

        await asyncio.gather(
            *[self._commit_read_only(call) for call in provisional],
            return_exceptions=True,
        )

        for call in staged:
            await self._commit_mutating(call)

    async def _commit_read_only(self, call: ToolCall) -> None:
        """Await the speculative task result and write the telemetry log."""
        if call._resolution_future.done():
            return

        try:
            result = await call.speculative_task
        except asyncio.CancelledError:
            log.warning("dispatcher: speculative task for %s cancelled at commit", call.name)
            if not call._resolution_future.done():
                call._resolution_future.set_result(_SUPERSEDED)
            return
        except Exception as exc:
            log.error("dispatcher: speculative task for %s raised: %s", call.name, exc)
            result = {"status": "error", "message": str(exc)}

        call.result      = result
        call.committed   = True
        call.committed_at = time.monotonic()

        self._write_telemetry(call)
        self._committed_history.append(call)

        result_json = json.dumps(result)
        if not call._resolution_future.done():
            call._resolution_future.set_result(result_json)

        log.info("dispatcher: COMMITTED (read-only) %s → %s", call.name, _abbrev_result(result))

    async def _commit_mutating(self, call: ToolCall) -> None:
        """Execute COMPENSABLE/IRREVERSIBLE API now that TRP is confirmed."""
        if call._resolution_future.done():
            return

        try:
            t_start = time.time()
            result  = await self._run_api(call.name, call.args)
            t_end   = time.time()
        except Exception as exc:
            log.error("dispatcher: mutation %s failed: %s", call.name, exc)
            result  = {"status": "error", "message": str(exc)}
            t_start = t_end = time.time()

        call.result       = result
        call.committed    = True
        call.committed_at = time.monotonic()

        self._write_telemetry(call, t_start=t_start, t_end=t_end)
        self._committed_history.append(call)

        if call.effect_class in (EffectClass.COMPENSABLE, EffectClass.IRREVERSIBLE):
            entry = self._build_compensation(call, result)
            self._saga.register(entry)

        result_json = json.dumps(result)
        if not call._resolution_future.done():
            call._resolution_future.set_result(result_json)

        log.info("dispatcher: COMMITTED (%s) %s → %s",
                 call.effect_class.name, call.name, _abbrev_result(result))

    # -----------------------------------------------------------------------
    # Abort path (ABORTED — barge-in)
    # -----------------------------------------------------------------------

    async def _abort_all(self) -> None:
        """
        Cancel all in-flight speculative tasks.
        Purge provisional cache and staged queue.
        DO NOT write to telemetry log — no benchmark record emitted.
        """
        log.info(
            "dispatcher: ABORTED — cancelling %d speculative + %d staged (no telemetry)",
            len(self._provisional_cache), len(self._staged_queue),
        )

        for call in self._provisional_cache.values():
            if call.speculative_task and not call.speculative_task.done():
                call.speculative_task.cancel()
            if not call._resolution_future.done():
                call._resolution_future.set_result(_ABORTED)

        for call in self._staged_queue:
            if not call._resolution_future.done():
                call._resolution_future.set_result(_ABORTED)

        self._provisional_cache.clear()
        self._staged_queue.clear()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _run_api(self, name: str, args: dict) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._registry(name, **args),
        )

    def _write_telemetry(
        self,
        call: ToolCall,
        t_start: Optional[float] = None,
        t_end:   Optional[float] = None,
    ) -> None:
        t_start = t_start or time.time()
        t_end   = t_end   or t_start
        record = {
            "room":    self._room_name,
            "call_id": call.call_id,
            "call": {
                "function":        call.name,
                "args":            call.args,
                "timestamp_start": t_start,
                "timestamp_end":   t_end,
            },
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.error("dispatcher: failed to write telemetry: %s", exc)

    def _retract_telemetry(self, call_id: str) -> None:
        import os
        if not os.path.exists(self._log_path):
            return
        try:
            new_lines = []
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        if item.get("call_id") == call_id:
                            continue
                        new_lines.append(line.strip())
                    except Exception:
                        new_lines.append(line.strip())
            with open(self._log_path, "w", encoding="utf-8") as f:
                for line in new_lines:
                    f.write(line + "\n")
        except OSError as exc:
            log.error("dispatcher: failed to retract telemetry for %s: %s", call_id, exc)

    def _find_provisional_by_name(self, name: str) -> Optional[ToolCall]:
        for call in self._provisional_cache.values():
            if call.name == name:
                return call
        return None

    def _build_compensation(self, call: ToolCall, result: dict) -> CompensationEntry:
        name   = call.name
        args   = call.args
        result = result  # capture

        if name in ("book_flight", "book_ticket"):
            booking_ref = result.get("booking_ref", "UNKNOWN")
            async def compensate():
                log.info("saga compensation: cancel_flight(booking_ref=%s)", booking_ref)
                try:
                    with open(self._audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "room": self._room_name,
                            "compensation": {"action": "cancel_flight", "booking_ref": booking_ref}
                        }) + "\n")
                except OSError as exc:
                    log.error("saga: failed to write audit compensation log: %s", exc)

        elif name in ("modify_autopay", "modify_autopay_source"):
            bill_type   = args.get("bill_type", "")
            prev_source = args.get("source_account", "")
            async def compensate():
                log.info("saga compensation: revert_autopay(bill=%s, prev_source=%s)", bill_type, prev_source)
                try:
                    with open(self._audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "room": self._room_name,
                            "compensation": {
                                "action": "revert_autopay",
                                "bill_type": bill_type,
                                "revert_to_source": "default",
                            }
                        }) + "\n")
                except OSError as exc:
                    log.error("saga: failed to write audit compensation log: %s", exc)

        elif name == "update_search_filter":
            filter_name = args.get("filter_name", "")
            async def compensate():
                log.info("saga compensation: reset_search_filter(filter=%s)", filter_name)
                try:
                    with open(self._audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "room": self._room_name,
                            "compensation": {"action": "reset_search_filter", "filter_name": filter_name}
                        }) + "\n")
                except OSError as exc:
                    log.error("saga: failed to write audit compensation log: %s", exc)

        elif name == "add_to_cart":
            product_id = args.get("product_id", "")
            quantity   = args.get("quantity", 1)
            async def compensate():
                log.info("saga compensation: remove_from_cart(product=%s, qty=%d)", product_id, quantity)
                try:
                    with open(self._audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "room": self._room_name,
                            "compensation": {
                                "action": "remove_from_cart",
                                "product_id": product_id,
                                "quantity": quantity,
                            }
                        }) + "\n")
                except OSError as exc:
                    log.error("saga: failed to write audit compensation log: %s", exc)

        elif name in ("update_identity_doc", "update_travel_profile"):
            doc_type   = args.get("doc_type", "")
            async def compensate():
                log.warning(
                    "saga audit: update_identity_doc for doc_type=%s is IRREVERSIBLE",
                    doc_type,
                )
                try:
                    with open(self._audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "room": self._room_name,
                            "compensation": {
                                "action": "audit_only_irreversible",
                                "tool": name,
                                "doc_type": doc_type,
                            }
                        }) + "\n")
                except OSError as exc:
                    log.error("saga: failed to write audit compensation log: %s", exc)

        elif name in ("cancel_pending_action", "process_exchange"):
            async def compensate():
                log.info("saga compensation: %s acknowledged", name)
                try:
                    with open(self._audit_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "room": self._room_name,
                            "compensation": {"action": f"revert_{name}"}
                        }) + "\n")
                except OSError as exc:
                    log.error("saga: failed to write audit compensation log: %s", exc)
        else:
            async def compensate():
                log.warning("saga: no compensation registered for %s", name)

        return CompensationEntry(
            call_id         = call.call_id,
            tool_name       = name,
            args            = args,
            committed_at    = call.committed_at or time.monotonic(),
            compensation_fn = compensate,
        )

    def reset_for_new_turn(self) -> None:
        """Called at turn boundary. Cancels lingering tasks and resolves pending futures."""
        discarded_sentinel = json.dumps({"status": "discarded"})

        for call in list(self._provisional_cache.values()):
            if call.speculative_task and not call.speculative_task.done():
                call.speculative_task.cancel()
            if not call._resolution_future.done():
                call._resolution_future.set_result(discarded_sentinel)

        for call in list(self._staged_queue):
            if not call._resolution_future.done():
                call._resolution_future.set_result(discarded_sentinel)

        self._provisional_cache.clear()
        self._staged_queue.clear()
        self._committed_history.clear()
        self._turn_call_history.clear()
        self._current_state = TRPState.LISTENING


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _abbrev_args(args: dict) -> str:
    s = json.dumps(args)
    return s[:60] + "…" if len(s) > 60 else s


def _abbrev_result(result: Any) -> str:
    s = json.dumps(result) if isinstance(result, dict) else str(result)
    return s[:60] + "…" if len(s) > 60 else s
