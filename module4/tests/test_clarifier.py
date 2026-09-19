"""Unit tests for the clarifier pipeline in isolation — no Module 3 runtime needed.
Run: python -m pytest module4/tests/test_clarifier.py -v
"""

from module4.clarifier import AmbiguitySource, GroundingPipeline
from module4.visual_buffer import SessionVisualBuffer

SESSION = "sess-unit"

TOOL_SET_POWER = {
    "name": "set_power",
    "input_schema": {
        "required": ["device"],
        "properties": {"device": {"description": "device name"}},
    },
    "side_effect": "STATE_MODIFYING",
}


def make_buffer_with_frame(objects, confidence=0.9):
    buf = SessionVisualBuffer(min_confidence=0.5)
    buf.add_frame_precomputed(SESSION, frame_index=0, caption="test frame",
                               salient_objects=objects, confidence=confidence)
    return buf


def test_linguistic_ambiguity_missing_slot():
    pipeline = GroundingPipeline(SessionVisualBuffer())
    verdict = pipeline.evaluate(SESSION, "turn it on", TOOL_SET_POWER)
    assert verdict.ambiguous
    assert verdict.source == AmbiguitySource.LINGUISTIC
    assert "device" in verdict.missing_slots


def test_linguistic_slot_present_not_ambiguous():
    pipeline = GroundingPipeline(SessionVisualBuffer())
    verdict = pipeline.evaluate(SESSION, "turn on the device please", TOOL_SET_POWER)
    assert not verdict.ambiguous


def test_visual_ambiguity_no_frame():
    pipeline = GroundingPipeline(SessionVisualBuffer())
    verdict = pipeline.evaluate(SESSION, "turn this on", active_tool=None)
    assert verdict.ambiguous
    assert verdict.source == AmbiguitySource.VISUAL_QUALITY
    assert verdict.guidance  # actionable camera guidance, not a bare re-ask


def test_visual_ambiguity_low_confidence_frame():
    buf = make_buffer_with_frame(objects=["red button"], confidence=0.2)
    pipeline = GroundingPipeline(buf, min_confidence=0.5)
    verdict = pipeline.evaluate(SESSION, "press this", active_tool=None)
    assert verdict.ambiguous
    assert verdict.source == AmbiguitySource.VISUAL_QUALITY


def test_cross_modal_ambiguity_two_similar_objects():
    buf = make_buffer_with_frame(objects=["left lamp", "right lamp"], confidence=0.9)
    pipeline = GroundingPipeline(buf)
    verdict = pipeline.evaluate(SESSION, "turn that on", active_tool=None)
    assert verdict.ambiguous
    assert verdict.source == AmbiguitySource.CROSS_MODAL_REFERENCE
    assert len(verdict.candidates) >= 2


def test_cross_modal_resolved_single_object():
    buf = make_buffer_with_frame(objects=["red button"], confidence=0.9)
    pipeline = GroundingPipeline(buf)
    verdict = pipeline.evaluate(SESSION, "press that", active_tool=None)
    assert not verdict.ambiguous
    assert verdict.candidates[0].label == "red button"


def test_no_deictic_reference_skips_vision_checks_entirely():
    # No frame in buffer at all, but no deictic reference either -> should NOT trigger
    # visual ambiguity, proving the pipeline short-circuits before paying for vision checks.
    pipeline = GroundingPipeline(SessionVisualBuffer())
    verdict = pipeline.evaluate(SESSION, "what's the weather like", active_tool=None)
    assert not verdict.ambiguous


def test_to_clarification_kwargs_shape():
    pipeline = GroundingPipeline(SessionVisualBuffer())
    verdict = pipeline.evaluate(SESSION, "turn it on", TOOL_SET_POWER)
    kwargs = verdict.to_clarification_kwargs()
    assert set(kwargs.keys()) == {"text", "missing_slots", "context"}
    assert kwargs["context"]["ambiguity_source"] == "linguistic"
