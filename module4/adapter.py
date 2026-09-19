"""
module4/adapter.py — Module 4 integration adapter for Module 3's real runtime.

>>> READ THIS FIRST: there is a genuine race condition here, not a hypothetical one <<<

module3/runtime/dispatcher/dispatcher.py fans an event out to every registered handler for
that event type via `asyncio.gather(...)` — "Multiple handlers per event type are supported
(fan-out)... called concurrently" (dispatcher.py's own docstring). If Module 4 registers a
handler on `END_OF_TURN` that decides "this is ambiguous, emit CLARIFICATION" while Module 2
*independently* registers its own `END_OF_TURN` handler that decides "reasoning looks fine,
emit TOOL_CALL" (see MODULE3_API_CONTRACT.md's Module2ReasoningAdapter example), both fire.
There is no built-in mechanism in this dispatcher for one handler's decision to suppress
another's — `emit_output`'s "Central Authorization Gate" rejects stale generations and
duplicate idempotency keys, it does not know what "ambiguous" means. You would ship a
prototype where the agent BOTH asks a clarifying question AND fires a tool call off
incomplete information, which is worse for the Task Completion (40%) and Interruption
Recovery (35%) scores than not gating at all — and it's exactly the kind of thing that would
only show up during a live demo with real timing, not in isolated unit tests.

**The fix**, and the reason this file exposes `evaluate_ambiguity()` as a plain importable
method, not just a handler: Module 2's END_OF_TURN handler should call
`Module4Adapter.evaluate_ambiguity(session_id, text)` SYNCHRONOUSLY, in-line, before it
decides whether to emit a TOOL_CALL. That's a single extra `await` in Module 2's existing
handler, it's race-free by construction (no fan-out involved), and it doesn't require any
change to Module 3's dispatcher. This is a decision Module 2's owner needs to sign off on —
raise it before day 8's integration pass, not during it.

We ALSO register a defensive END_OF_TURN handler below (`on_end_of_turn_defensive`) that
emits CLARIFICATION on its own if ambiguity is detected, purely as a fallback for the case
where Module 2 does NOT adopt the direct-call path. It is strictly worse than the direct
call (races exactly as described above) — treat it as a safety net, not the real
integration, and delete it once Module 2 calls `evaluate_ambiguity()` directly.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Callable, Optional

from module3.runtime.events import InputEventType, make_clarification
from module3.runtime.interfaces import IRuntimeClient

from module4.clarifier import ClarifierVerdict, GroundingPipeline
from module4.visual_buffer import CaptionResult, SessionVisualBuffer

logger = logging.getLogger("module4.adapter")


class Module4Adapter:
    """Wraps GroundingPipeline + SessionVisualBuffer behind Module 3's IRuntimeClient.

    Parameters
    ----------
    runtime: any object satisfying IRuntimeClient (the real Runtime, or a test double).
    caption_fn_async: async callable (frame_b64, frame_index) -> (caption, objects, confidence).
        Defaults to Module 3's own `mock_frame_lookup` mock — see module docstring on why
        that beats an ad hoc fake for cross-team testing. Swap for a real VLM call for the
        final submission.
    """

    def __init__(
        self,
        runtime: IRuntimeClient,
        caption_fn_async: Optional[Callable[[str, int], Any]] = None,
        min_confidence: float = 0.5,
    ):
        self.runtime = runtime
        self._caption_fn_async = caption_fn_async or self._default_mock_caption_fn_async
        self.buffer = SessionVisualBuffer(min_confidence=min_confidence)
        self.pipeline = GroundingPipeline(self.buffer, min_confidence=min_confidence)
        # Session-scoped manifest cache — raw dicts as delivered in TOOL_MANIFEST's
        # payload["tools"], NOT module3.runtime.tools.ToolDefinition instances (importing
        # that would violate the boundary rule in MODULE3_API_CONTRACT.md).
        self._manifests: dict[str, list[dict]] = {}
        self._active_tool: dict[str, Optional[dict]] = {}

    @staticmethod
    async def _default_mock_caption_fn_async(frame_b64: str, frame_index: int) -> CaptionResult:
        from module3.mocks.tools import mock_frame_lookup
        result = await mock_frame_lookup(frame_index=frame_index, latency_ms=0.0)
        return result.data["description"], result.data["objects_detected"], result.data["confidence"]

    # -- installation --------------------------------------------------------------

    def install(self) -> None:
        """Register handlers. Call once after Runtime.start()."""
        self.runtime.register_handler(InputEventType.VIDEO_FRAME, self.on_video_frame)
        self.runtime.register_handler(InputEventType.TOOL_MANIFEST, self.on_tool_manifest)
        # Defensive fallback only — see module docstring. Comment out once Module 2 calls
        # evaluate_ambiguity() directly, to remove the race entirely rather than merely
        # tolerate it.
        self.runtime.register_handler(InputEventType.END_OF_TURN, self.on_end_of_turn_defensive)

    # -- handlers -------------------------------------------------------------------

    async def on_video_frame(self, event) -> None:
        payload = event.payload
        caption, objects, confidence = await self._caption_fn_async(
            payload["frame_b64"], payload["frame_index"]
        )
        self.buffer.add_frame_precomputed(
            session_id=event.session_id,
            frame_index=payload["frame_index"],
            caption=caption,
            salient_objects=objects,
            confidence=confidence,
            timestamp_ms=event.timestamp_ms,
        )

    async def on_tool_manifest(self, event) -> None:
        tools = event.payload.get("tools", [])
        self._manifests[event.session_id] = tools
        if tools:
            self._active_tool[event.session_id] = tools[0]

    async def on_end_of_turn_defensive(self, event) -> None:
        text = event.payload.get("final_text") or ""
        if not text:
            return
        verdict = self.evaluate_ambiguity(event.session_id, text)
        if verdict.ambiguous:
            logger.warning(
                "session=%s: Module 4's DEFENSIVE end-of-turn handler fired ambiguity=%s. "
                "If Module 2 also emitted a TOOL_CALL for this same turn, you have the race "
                "described in this file's module docstring — check the trace.",
                event.session_id, verdict.source,
            )
            await self.runtime.emit_output(
                make_clarification(
                    event.session_id,
                    self.runtime.clock.now(),
                    **verdict.to_clarification_kwargs(),
                ),
                priority=8,
            )

    # -- the API Module 2 should actually call --------------------------------------

    def evaluate_ambiguity(self, session_id: str, text: str) -> ClarifierVerdict:
        """Synchronous, race-free ambiguity check. Call this from INSIDE Module 2's own
        END_OF_TURN handler, before deciding to emit a TOOL_CALL:

            from module4.adapter import Module4Adapter
            ...
            verdict = module4_adapter.evaluate_ambiguity(event.session_id, final_text)
            if verdict.ambiguous:
                await runtime.emit_output(make_clarification(
                    event.session_id, runtime.clock.now(), **verdict.to_clarification_kwargs()
                ))
                return  # do NOT also emit a tool call this turn
            # ... proceed with tool selection / reasoning as normal
        """
        active_tool = self._active_tool.get(session_id)
        return self.pipeline.evaluate(session_id, text, active_tool)

    def drop_session(self, session_id: str) -> None:
        """Call from Module 3's session-close path once that hook exists — see
        visual_buffer.py's module docstring for why this matters for the session-scoped-
        memory constraint (§6 of the guide)."""
        self.buffer.drop_session(session_id)
        self._manifests.pop(session_id, None)
        self._active_tool.pop(session_id, None)
