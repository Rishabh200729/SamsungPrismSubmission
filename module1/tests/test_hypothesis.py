"""
module1/tests/test_hypothesis.py

Unit tests for HypothesisBuffer and HypothesisState.
No runtime dependency — pure data-structure tests.
"""

from __future__ import annotations

import pytest
from module1.fast_path.hypothesis import (
    HypothesisBuffer,
    HypothesisState,
    _extract_entities,
    _extract_intent,
    _detect_correction,
)


class TestHypothesisState:
    def test_initial_state(self):
        h = HypothesisState(generation=1)
        assert h.intent is None
        assert h.entities == {}
        assert h.confidence == 0.0
        assert h.chunk_count == 0
        assert not h.is_correction

    def test_update_accumulates_text(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight from Seoul")
        assert "Seoul" in h.accumulated_text
        assert h.chunk_count == 1

    def test_update_extracts_entities(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight from Seoul to Tokyo")
        assert h.entities.get("origin", "").lower() == "seoul"
        assert h.entities.get("destination", "").lower() == "tokyo"

    def test_update_extracts_intent(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight")
        assert h.intent == "book_flight"

    def test_confidence_grows_with_chunks(self):
        h = HypothesisState(generation=1)
        h.update("word1")
        c1 = h.confidence
        h.update("word2")
        assert h.confidence > c1

    def test_confidence_caps_below_1_before_finalize(self):
        h = HypothesisState(generation=1)
        for i in range(20):
            h.update(f"word{i}")
        assert h.confidence < 1.0

    def test_finalize_sets_confidence_to_1(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight")
        h.finalize("Book a flight from Seoul to Tokyo")
        assert h.confidence == 1.0

    def test_finalize_with_authoritative_text_re_extracts(self):
        h = HypothesisState(generation=1)
        h.update("something random")
        h.finalize("Book a flight from Seoul to Tokyo")
        assert h.entities.get("destination", "").lower() == "tokyo"

    def test_is_correction_set_when_destination_changes(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight from Seoul to Tokyo")
        assert not h.is_correction  # first update, no previous entities
        h.update("actually change destination to Osaka")
        assert h.is_correction

    def test_is_correction_not_set_on_first_chunk(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight to Tokyo")
        assert not h.is_correction


class TestHypothesisBuffer:
    def test_get_or_create_returns_new_for_unknown_session(self):
        buf = HypothesisBuffer()
        h = buf.get_or_create("s1", 1)
        assert h.generation == 1
        assert h.chunk_count == 0

    def test_get_or_create_returns_same_object_same_gen(self):
        buf = HypothesisBuffer()
        h1 = buf.get_or_create("s1", 1)
        h1.update("hello")
        h2 = buf.get_or_create("s1", 1)
        assert h2 is h1
        assert h2.chunk_count == 1

    def test_get_or_create_creates_fresh_on_gen_change(self):
        buf = HypothesisBuffer()
        h1 = buf.get_or_create("s1", 1)
        h1.update("old text")
        h2 = buf.get_or_create("s1", 2)  # generation advanced
        assert h2 is not h1
        assert h2.chunk_count == 0
        assert h2.generation == 2

    def test_dispose_clears_state(self):
        buf = HypothesisBuffer()
        h = buf.get_or_create("s1", 1)
        h.update("something")
        buf.dispose("s1")
        assert buf.peek("s1") is None

    def test_get_or_create_after_dispose_creates_fresh(self):
        buf = HypothesisBuffer()
        h1 = buf.get_or_create("s1", 1)
        h1.update("something")
        buf.dispose("s1")
        h2 = buf.get_or_create("s1", 1)
        assert h2 is not h1
        assert h2.chunk_count == 0

    def test_session_isolation(self):
        buf = HypothesisBuffer()
        h_a = buf.get_or_create("session-a", 1)
        h_b = buf.get_or_create("session-b", 1)
        h_a.update("text for A")
        assert h_b.chunk_count == 0


class TestEntityExtraction:
    def test_origin_extracted(self):
        e = _extract_entities("Book a flight from Seoul to Tokyo")
        assert e.get("origin", "").lower() == "seoul"

    def test_destination_extracted(self):
        e = _extract_entities("Book a flight from Seoul to Tokyo")
        assert e.get("destination", "").lower() == "tokyo"

    def test_correction_destination_overrides(self):
        # "change it to Osaka" should override earlier Tokyo
        e = _extract_entities("Book a flight to Tokyo. Actually change it to Osaka")
        assert e.get("destination", "").lower() == "osaka"

    def test_intent_book_flight(self):
        assert _extract_intent("Book a flight") == "book_flight"

    def test_intent_search(self):
        assert _extract_intent("Find me a hotel") == "search"

    def test_intent_cancel(self):
        assert _extract_intent("Cancel that") == "cancel"

    def test_intent_none(self):
        assert _extract_intent("hello") is None


class TestCorrectionDetection:
    def test_no_correction_if_prev_empty(self):
        assert not _detect_correction({}, {"destination": "Tokyo"})

    def test_no_correction_if_same_value(self):
        assert not _detect_correction(
            {"destination": "Tokyo"}, {"destination": "Tokyo"}
        )

    def test_correction_if_value_changed(self):
        assert _detect_correction(
            {"destination": "Tokyo"}, {"destination": "Osaka"}
        )

    def test_case_insensitive(self):
        assert not _detect_correction(
            {"destination": "tokyo"}, {"destination": "Tokyo"}
        )


# ---------------------------------------------------------------------------
# Candidate A — Inverted 'instead' pattern regression tests
# Before fix: "No, Tokyo instead" → {} (correction missed)
# After fix:  "No, Tokyo instead" → {destination: Tokyo} (correction detected)
# ---------------------------------------------------------------------------

class TestInvertedInsteadPattern:
    """
    Regression tests for the inverted 'instead' pattern (Candidate A fix).
    City BEFORE 'instead' is a common natural speech pattern.
    """

    def test_city_instead_extracts_destination(self):
        e = _extract_entities("No, Tokyo instead")
        assert e.get("destination", "").lower() == "tokyo", (
            "CANDIDATE A: 'No, Tokyo instead' must extract destination=Tokyo"
        )

    def test_osaka_instead(self):
        e = _extract_entities("Osaka instead")
        assert e.get("destination", "").lower() == "osaka"

    def test_inverted_instead_triggers_correction(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight to Seoul")
        h.update("No, Tokyo instead")
        assert h.is_correction, (
            "CANDIDATE A: 'No, Tokyo instead' after Seoul must trigger is_correction"
        )
        assert h.entities.get("destination", "").lower() == "tokyo"

    def test_monday_instead_is_not_city(self):
        e = _extract_entities("Monday instead")
        assert e.get("destination") is None, "Calendar words must not be extracted as cities"

    def test_friday_instead_is_not_city(self):
        e = _extract_entities("Friday instead")
        assert e.get("destination") is None

    def test_first_instead_is_not_city(self):
        e = _extract_entities("First instead")
        assert e.get("destination") is None, "'First' must be blocked by plausibility guard"

    def test_book_instead_is_not_city(self):
        e = _extract_entities("Book instead")
        assert e.get("destination") is None


# ---------------------------------------------------------------------------
# Candidate B — Multi-word destination extraction regression tests
# Before fix: "to Los Angeles" → destination='Los' (truncated)
# After fix:  "to Los Angeles" → destination='Los Angeles' (full)
# ---------------------------------------------------------------------------

class TestMultiWordDestination:
    """
    Regression tests for multi-word destination extraction (Candidate B fix).
    Destination patterns now use _CITY_2W to capture Los Angeles, New York, etc.
    """

    def test_los_angeles_extracted_fully(self):
        e = _extract_entities("Book a flight to Los Angeles")
        assert e.get("destination", "").lower() == "los angeles", (
            "CANDIDATE B: destination must be 'Los Angeles', not truncated 'Los'"
        )

    def test_new_york_extracted_fully(self):
        e = _extract_entities("Book a flight to New York")
        assert e.get("destination", "").lower() == "new york"

    def test_san_francisco_extracted_fully(self):
        e = _extract_entities("fly from Seoul to San Francisco")
        assert e.get("destination", "").lower() == "san francisco"

    def test_origin_and_2w_destination(self):
        e = _extract_entities("fly from San Francisco to Los Angeles")
        assert e.get("origin", "").lower() == "san francisco"
        assert e.get("destination", "").lower() == "los angeles"

    def test_change_to_2w_city(self):
        e = _extract_entities("change it to Los Angeles")
        assert e.get("destination", "").lower() == "los angeles"

    def test_make_it_2w_city(self):
        e = _extract_entities("make it New York")
        assert e.get("destination", "").lower() == "new york"

    def test_2w_correction_detected(self):
        h = HypothesisState(generation=1)
        h.update("Book a flight to Tokyo")
        h.update("change it to Los Angeles")
        assert h.is_correction
        assert h.entities.get("destination", "").lower() == "los angeles"

    def test_1w_city_unaffected(self):
        """Single-word cities still work correctly after the upgrade."""
        e = _extract_entities("Book a flight to Tokyo")
        assert e.get("destination", "").lower() == "tokyo"

    def test_to_book_flight_not_extracted(self):
        """'Book' is in blocklist — 'to Book Flight' must not be a destination."""
        e = _extract_entities("to Book Flight")
        assert e.get("destination") is None

    def test_to_first_class_not_extracted(self):
        """'First' is in blocklist — 'to First Class' must not be a destination."""
        e = _extract_entities("to First Class")
        assert e.get("destination") is None

    def test_same_2w_city_no_false_correction(self):
        """Same 2W city in both chunks must not trigger is_correction."""
        h = HypothesisState(generation=1)
        h.update("Book a flight to Los Angeles")
        h.update("change it to Los Angeles")
        assert not h.is_correction, (
            "Same multi-word destination must not trigger correction"
        )
