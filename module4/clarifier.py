"""
module4/clarifier.py — Multimodal ambiguity resolution: "ambiguous vs. answerable".

Research grounding (full citations in module4/README.md):

  [1] Yang et al., "Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for
      Egocentric Intent Disambiguation", AAAI 2026 (arXiv:2511.08971). Decomposes
      ambiguity into THREE independent sub-problems rather than one monolithic VLM
      judgment call: (a) text clarifier — underspecified language, (b) vision clarifier —
      actionable camera-quality guidance, not just a re-ask, (c) cross-modal clarifier —
      grounding deictic references to specific objects. We copy this three-way split
      directly (TextClarifier / VisionClarifier / CrossModalClarifier below). We do NOT
      copy their 3D-pointing-gesture geometry — this task's I/O contract gives PNG frames
      + text, no gesture/pose stream, so that part of the paper doesn't transfer.

  [2] "BLaVe-CoT: Consistency-Aware VQA for Blind and Low Vision Users" (arXiv:2509.06010).
      Key idea taken: track multiple plausible groundings with scores rather than forcing
      one, and treat a small top-1/top-2 margin as itself the ambiguity signal (not just an
      absolute confidence threshold). Used in CrossModalClarifier.MARGIN_THRESHOLD below.
      We do NOT take their BLIP-2/PolyFormer segmentation pipeline.

Honest limitation: per the build plan, Module 4's actual scored contribution is "a unified
schema-enforcement + ambiguity-clarification layer evaluated under the hidden multimodal
1.5x condition" — the novelty claim is in the integration + evaluation angle, not in any one
clarifier being independently novel. Don't oversell the individual clarifiers on the
innovation-highlights slide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from module4.visual_buffer import FrameRecord

# Deictic / anaphoric markers signalling "this reference needs grounding against a frame".
# Deliberately a small, explicit, unit-testable list rather than an ML classifier — at a
# 10-day clock, a regex you can debug under a demo beats a black-box you can't. Expand as
# you see false negatives against your own test scenarios.
_DEICTIC_PATTERN = re.compile(
    r"\b(this|that|these|those|it|here|there|the (one|button|thing|part|screen))\b",
    re.IGNORECASE,
)


class AmbiguitySource(str, Enum):
    LINGUISTIC = "linguistic"
    VISUAL_QUALITY = "visual_quality"
    CROSS_MODAL_REFERENCE = "cross_modal_reference"


@dataclass
class GroundingCandidate:
    label: str
    score: float
    source_frame_index: Optional[int] = None


@dataclass
class ClarifierVerdict:
    ambiguous: bool
    source: Optional[AmbiguitySource] = None
    question: Optional[str] = None
    guidance: Optional[str] = None          # only set for VISUAL_QUALITY
    missing_slots: list[str] = field(default_factory=list)
    candidates: list[GroundingCandidate] = field(default_factory=list)
    confidence: float = 1.0
    debug_reason: str = ""

    def to_clarification_kwargs(self) -> dict[str, Any]:
        """Shape this verdict into kwargs for module3.runtime.events.make_clarification.
        ClarificationPayload has no ambiguity_source/guidance fields of its own, so we
        carry them in `context` — that field is an open dict specifically for this."""
        assert self.ambiguous, "only call this on an ambiguous verdict"
        return {
            "text": self.question or "Could you clarify that?",
            "missing_slots": self.missing_slots,
            "context": {
                "ambiguity_source": self.source.value if self.source else None,
                "guidance": self.guidance,
                "candidates": [c.label for c in self.candidates],
                "debug_reason": self.debug_reason,
            },
        }


class TextClarifier:
    """[1]'s text clarifier, driven off the LIVE tool manifest rather than a generic
    slot-filler. `tool` here is a raw dict shaped like Module 3's `ToolDefinition`
    (module3/runtime/tools/manifest.py): {"name", "input_schema", "side_effect", ...}.
    We read this as a plain dict rather than importing ToolDefinition, per
    MODULE3_API_CONTRACT.md's boundary rule (module3.runtime.tools is off-limits to
    external modules) — the raw dicts arrive via TOOL_MANIFEST's payload["tools"] anyway."""

    def evaluate(self, text: str, active_tool: Optional[dict]) -> ClarifierVerdict:
        if not active_tool:
            return ClarifierVerdict(ambiguous=False, debug_reason="no active tool to check slots against")

        schema = active_tool.get("input_schema", {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = [slot for slot in required if slot.lower() not in text.lower()]

        if not missing:
            return ClarifierVerdict(ambiguous=False, debug_reason="all required slots present in utterance")

        # One crisp question at a time, not a batch — Full-Duplex-Bench-v3's failure
        # taxonomy flags long clarification turns as a self-correction/rollback failure
        # mode, and a short turn keeps the fast-path filler + clarification pair tight.
        slot_name = missing[0]
        desc = properties.get(slot_name, {}).get("description", slot_name)
        return ClarifierVerdict(
            ambiguous=True,
            source=AmbiguitySource.LINGUISTIC,
            question=f"Quick check — what's the {desc}?",
            missing_slots=missing,
            confidence=0.9,
            debug_reason=f"required slot(s) missing for tool={active_tool.get('name')}: {missing}",
        )


class VisionClarifier:
    """[1]'s vision clarifier: give actionable camera guidance, not a bare re-ask, when
    the *frame* is the problem rather than the language. Uses FrameRecord.confidence,
    which — once you wire the real `caption_fn` — comes straight from
    module3.mocks.tools.mock_frame_lookup's `confidence` field (or a real VLM's)."""

    def evaluate(self, frame: Optional[FrameRecord], buffer_is_stale: bool,
                 min_confidence: float) -> ClarifierVerdict:
        if frame is None or buffer_is_stale:
            return ClarifierVerdict(
                ambiguous=True,
                source=AmbiguitySource.VISUAL_QUALITY,
                question="Could you point the camera at what you're asking about?",
                guidance="No recent frame available — hold the camera on the item for a moment.",
                confidence=0.95,
                debug_reason="no frame, or frame older than staleness window",
            )
        if frame.confidence < min_confidence or not frame.salient_objects:
            return ClarifierVerdict(
                ambiguous=True,
                source=AmbiguitySource.VISUAL_QUALITY,
                question="I'm not getting a clear view — can you get a bit closer?",
                guidance="Move closer and keep the item centered in frame.",
                confidence=1.0 - frame.confidence,
                debug_reason=f"frame confidence {frame.confidence:.2f} below threshold {min_confidence}",
            )
        return ClarifierVerdict(ambiguous=False, debug_reason="frame confidence sufficient")


class CrossModalClarifier:
    """[1]'s deictic-grounding clarifier + [2]'s multiple-valid-grounding margin check.
    Tracks ALL plausible candidate objects for a reference like "this"/"that" rather than
    committing to the single best guess, and flags ambiguity when the top two are too
    close to call."""

    MARGIN_THRESHOLD = 0.15   # top1.score - top2.score below this = toss-up

    def _score_candidates(self, frames: list[FrameRecord]) -> list[GroundingCandidate]:
        # Placeholder recency-weighted scoring. This is the seam where a real grounding
        # model (CLIP-style similarity between the referring expression and each detected
        # object) plugs in — see README "Known gaps" for what to upgrade first.
        candidates: list[GroundingCandidate] = []
        for rank, frame in enumerate(reversed(frames)):
            recency_weight = max(0.3, 1.0 - 0.15 * rank)
            for obj in frame.salient_objects:
                candidates.append(GroundingCandidate(
                    label=obj, score=recency_weight * frame.confidence,
                    source_frame_index=frame.frame_index,
                ))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def evaluate(self, text: str, frames: list[FrameRecord]) -> ClarifierVerdict:
        if not _DEICTIC_PATTERN.search(text):
            return ClarifierVerdict(ambiguous=False, debug_reason="no deictic reference in utterance")

        candidates = self._score_candidates(frames)
        if not candidates:
            return ClarifierVerdict(
                ambiguous=True,
                source=AmbiguitySource.CROSS_MODAL_REFERENCE,
                question="Which one do you mean?",
                confidence=0.8,
                debug_reason="deictic reference present but no grounded candidates in buffer",
            )

        top1, top2 = candidates[0], (candidates[1] if len(candidates) > 1 else None)
        margin = top1.score - top2.score if top2 else 1.0

        if margin < self.MARGIN_THRESHOLD:
            return ClarifierVerdict(
                ambiguous=True,
                source=AmbiguitySource.CROSS_MODAL_REFERENCE,
                question=f"Did you mean the {top1.label} or the {top2.label}?",
                candidates=candidates[:3],
                confidence=0.6,
                debug_reason=f"top-2 margin too small ({margin:.2f} < {self.MARGIN_THRESHOLD})",
            )

        return ClarifierVerdict(
            ambiguous=False,
            candidates=candidates[:1],
            confidence=top1.score,
            debug_reason=f"resolved '{text}' -> '{top1.label}' with margin {margin:.2f}",
        )


class GroundingPipeline:
    """Cheap-first, short-circuiting pipeline. Order matters for latency: the linguistic
    check is near-free string ops; vision/cross-modal checks are where a real model call
    would live, so we only pay for those once the cheap check hasn't already resolved the
    turn. Emit a fast-path filler BEFORE calling this (§3.2 item 5's "behind a
    conversational acknowledgment") — this pipeline itself is not guaranteed sub-100ms
    once a real vision backend is wired in."""

    def __init__(self, buffer, min_confidence: float = 0.5):
        self.buffer = buffer
        self.min_confidence = min_confidence
        self.text_clarifier = TextClarifier()
        self.vision_clarifier = VisionClarifier()
        self.cross_modal_clarifier = CrossModalClarifier()

    def evaluate(self, session_id: str, text: str, active_tool: Optional[dict]) -> ClarifierVerdict:
        v = self.text_clarifier.evaluate(text, active_tool)
        if v.ambiguous:
            return v

        if _DEICTIC_PATTERN.search(text):
            latest = self.buffer.latest_frame(session_id)
            stale = self.buffer.is_stale(session_id)
            v = self.vision_clarifier.evaluate(latest, stale, self.min_confidence)
            if v.ambiguous:
                return v

            frames = self.buffer.recent_frames(session_id)
            # NOTE: previously this method threw away `v` here and returned a fresh
            # "all clarifiers passed" verdict at the bottom, silently discarding
            # CrossModalClarifier's resolved `candidates` (the actual grounding result
            # callers need, e.g. to log which object "that" was resolved to). Caught by
            # test_cross_modal_resolved_single_object in tests/test_clarifier.py.
            v = self.cross_modal_clarifier.evaluate(text, frames)
            return v

        return ClarifierVerdict(ambiguous=False, debug_reason="all clarifiers passed")
