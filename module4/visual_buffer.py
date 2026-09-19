"""
module4/visual_buffer.py — Session-scoped sliding-window frame buffer.

Design source: the Meta patent cited in the team's build plan ("Multimodal dialog state
tracking and action prediction for assistant systems") uses a sliding-window visual-state
buffer to resolve coreference ("this", "that") against recently seen frames. Cited as
prior-art / engineering pattern only, per the build plan's own caveat — this is our
implementation, not the patent's.

Field names here are deliberately matched to what Module 3 actually defines, not to a
generic guess:
  - `VideoFramePayload` (module3/runtime/events/input_events.py) carries `frame_b64`,
    `width`, `height`, `frame_index` — there is no `frame_id`, so `frame_index` is the
    identity we key on.
  - `mock_frame_lookup()` (module3/mocks/tools.py) — Module 3's existing frame-analysis
    mock — returns `{"description": str, "objects_detected": list[str], "confidence":
    float}`. The default `caption_fn` below returns that exact shape so swapping in the
    mock (or, later, a real captioner/VLM) is a one-line change, not a rewrite.

Hard constraint from Theme_5_Guide.pdf §6: "State Scope: Session-scoped memory only (no
cross-session caching)." `drop_session()` MUST be wired to Module 3's session-close path
(`Runtime.close_session` / the SESSION_CLOSED internal event) or this buffer leaks across
sessions for the lifetime of the process. Module 3's `close_session()` is public API
(IRuntimeClient) but does not currently call out into other modules on close — if there's
no session-lifecycle hook exposed yet, that's a real gap, not just a Module 4 TODO. Raise
it with whoever owns runtime.py; don't quietly assume it'll be handled.

Deliberately no real vision embeddings (no CLIP/SigLIP) — see README "Known gaps" for why,
and for what to upgrade first if you have spare time after core integration is stable.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

CaptionResult = tuple[str, list[str], float]   # (caption, objects_detected, confidence)
CaptionFn = Callable[[str, int], CaptionResult]  # (frame_b64, frame_index) -> CaptionResult


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    caption: str
    salient_objects: list[str] = field(default_factory=list)
    confidence: float = 0.0


def _default_caption_fn(frame_b64: str, frame_index: int) -> CaptionResult:
    """No-op stand-in so the module runs standalone without a vision backend wired up.
    Replace with a call to module3.mocks.tools.mock_frame_lookup (or a real VLM call)
    in production — see Module4Adapter's `caption_fn` parameter."""
    return f"[uncaptioned frame #{frame_index}]", [], 0.0


class SessionVisualBuffer:
    """Fixed-size, per-session sliding window of recent frames."""

    def __init__(self, window_size: int = 6, staleness_ms: float = 15_000,
                 min_confidence: float = 0.5, caption_fn: CaptionFn = _default_caption_fn):
        self.window_size = window_size
        self.staleness_ms = staleness_ms
        self.min_confidence = min_confidence
        self.caption_fn = caption_fn
        self._buffers: dict[str, deque[FrameRecord]] = {}

    def add_frame(self, session_id: str, frame_b64: str, frame_index: int,
                  timestamp_ms: Optional[float] = None) -> FrameRecord:
        """Synchronous path: runs self.caption_fn inline. Used by standalone tests /
        callers that don't need an async captioner. Module4Adapter's real (async) path
        uses `add_frame_precomputed` instead — see that method's docstring."""
        caption, objects, confidence = self.caption_fn(frame_b64, frame_index)
        return self.add_frame_precomputed(session_id, frame_index, caption, objects,
                                           confidence, timestamp_ms)

    def add_frame_precomputed(self, session_id: str, frame_index: int, caption: str,
                               salient_objects: list[str], confidence: float,
                               timestamp_ms: Optional[float] = None) -> FrameRecord:
        """Use this when captioning already happened elsewhere (e.g. Module4Adapter awaits
        an async caption_fn before calling in) — avoids running self.caption_fn a second
        time on the same frame."""
        buf = self._buffers.setdefault(session_id, deque(maxlen=self.window_size))
        rec = FrameRecord(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else time.time() * 1000,
            caption=caption,
            salient_objects=salient_objects,
            confidence=confidence,
        )
        buf.append(rec)
        return rec

    def recent_frames(self, session_id: str, max_age_ms: Optional[float] = None) -> list[FrameRecord]:
        buf = self._buffers.get(session_id)
        if not buf:
            return []
        cutoff = None
        if max_age_ms is not None:
            cutoff = time.time() * 1000 - max_age_ms
        return [f for f in buf if cutoff is None or f.timestamp_ms >= cutoff]

    def latest_frame(self, session_id: str) -> Optional[FrameRecord]:
        buf = self._buffers.get(session_id)
        return buf[-1] if buf else None

    def is_stale(self, session_id: str) -> bool:
        latest = self.latest_frame(session_id)
        if latest is None:
            return True
        return (time.time() * 1000 - latest.timestamp_ms) > self.staleness_ms

    def is_low_confidence(self, session_id: str) -> bool:
        latest = self.latest_frame(session_id)
        return latest is None or latest.confidence < self.min_confidence

    def drop_session(self, session_id: str) -> None:
        """MUST be called on session end — see module docstring."""
        self._buffers.pop(session_id, None)
