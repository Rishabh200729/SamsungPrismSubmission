"""
module1/fast_path/hypothesis.py

Lightweight, disposable, generation-stamped provisional hypothesis.

Tracks only what Module 1 needs for:
- Provisional intent / entity evidence from incoming chunks
- Correction / repair detection (slot value changes mid-utterance)
- Per-hypothesis confidence
- The generation at which this hypothesis was created

Design rules:
- No LLM calls. All extraction is keyword-based and O(n) in text length.
- Disposable: HypothesisBuffer disposes the state on generation change.
- Replaceable: the extraction logic (extract_entities / extract_intent)
  is a private implementation detail; swap it without touching the router.

CORRECTION FALSE-POSITIVE HARDENING
-------------------------------------
Self-correction detection has two independent guards before triggering
a synthetic INTERRUPTION:

Guard 1 — Extraction quality: entities are only extracted when they appear
  in a CORRECTION CONTEXT, not whenever a capitalised word appears after
  "to" or "from". Specifically:
  - "change it to X", "make it X", "actually change ... to X", "switch to X"
    are correction-context patterns (high specificity).
  - "to X" alone (generic) only fires for the destination slot, and only
    when X is followed by a known sentence-ending boundary or another
    preposition that isolates it.
  - "Monday", "Actually", "Book", "Change" alone do NOT capture city entities
    because they either fail the capitalisation+boundary rules or are matched
    by more specific patterns that require explicit slot-modification verbs.

Guard 2 — Router guard: `hypothesis.is_correction` is True only if an entity
  VALUE CHANGED from a previously observed value. The router additionally
  requires `has_active_work` before submitting any INTERRUPTION. This means
  even if extraction produces a false positive entity, no INTERRUPTION fires
  unless real background tasks are running.

Together: false positives require BOTH guards to fail simultaneously, making
them extremely unlikely without genuine slot-revision language.

CITY EXTRACTION CONSTRAINTS
-----------------------------
_CITY_1W captures `[A-Z][a-zA-Z]+`. To prevent "Monday", "Actually", "Book"
from being treated as cities, the following additional constraints apply:

1. Correction patterns require explicit slot-modification verbs:
   "change ... to <City>", "make it <City>", "switch to <City>".
   These verbs do not co-occur with calendar words or filler verbs.

2. Generic destination "to <City>" only fires when the word following "to"
   is NOT a known calendar term or common non-city capitalized word.
   A blocklist of common false positives is checked before accepting the match.

3. Origin "from <City>" uses a context-bounded lookahead: the city is only
   accepted if followed by another preposition (to/on/via) or end-of-phrase.
   This prevents "from Seoul to" capturing "Seoul to".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# HypothesisState — the data object
# ---------------------------------------------------------------------------

@dataclass
class HypothesisState:
    """
    Compact generation-stamped provisional hypothesis for one session.

    Fields
    ------
    generation        : int
        Session generation at the time this hypothesis was created.
        Used for lazy staleness detection.
    intent            : str | None
        Lightweight best-guess intent (e.g. "book_flight", "search").
    entities          : dict[str, str]
        Slot/entity map extracted from accumulated text.
        e.g. {"origin": "Seoul", "destination": "Tokyo"}
    confidence        : float
        Rough signal strength, 0.0–1.0. Increments with each chunk.
    chunk_count       : int
        Number of TEXT_CHUNK increments applied.
    accumulated_text  : str
        All text seen so far in this utterance.
    is_correction     : bool
        True if a slot value changed between consecutive chunks.
        Cleared when the hypothesis is finalized.
    _prev_entities    : dict[str, str]
        Snapshot of entities before the last update, used internally
        to detect per-update corrections.
    _correction_submitted : bool
        HIGH-2 guard: set to True immediately after Module 1 submits a
        synthetic competitive INTERRUPTION for this hypothesis. Prevents
        a subsequent TEXT_CHUNK on the same hypothesis (before the generation
        advances) from submitting a duplicate synthetic INTERRUPTION.

        Reset automatically when this hypothesis is replaced — either by
        `HypothesisBuffer.dispose()` (eager, on INTERRUPTION handler) or by
        `HypothesisBuffer.get_or_create()` returning a new object when the
        generation changes (lazy). A fresh HypothesisState always starts
        with _correction_submitted=False.
    """
    generation: int = 1
    intent: Optional[str] = None
    entities: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    chunk_count: int = 0
    accumulated_text: str = ""
    is_correction: bool = False
    _prev_entities: dict[str, str] = field(default_factory=dict, repr=False)
    _correction_submitted: bool = field(default=False, repr=False)

    def update(self, text: str) -> None:
        """
        Append a new text chunk and refresh evidence.

        Correction detection: if any entity value changes from a previously
        observed non-None value, is_correction is set to True.
        """
        self._prev_entities = dict(self.entities)
        if self.accumulated_text:
            self.accumulated_text += " " + text.strip()
        else:
            self.accumulated_text = text.strip()
        self.chunk_count += 1

        # Re-extract from the full accumulated text
        new_entities = _extract_entities(self.accumulated_text)
        new_intent = _extract_intent(self.accumulated_text)

        # Detect slot-value correction
        self.is_correction = _detect_correction(self._prev_entities, new_entities)

        self.entities = new_entities
        self.intent = new_intent
        # Confidence grows with chunk count, caps at 0.9 (finalize sets 1.0)
        self.confidence = min(0.9, self.chunk_count * 0.2)

    def finalize(self, final_text: str | None = None) -> None:
        """
        Mark the hypothesis complete (END_OF_TURN received).
        If final_text is provided, re-extract from it (authoritative transcript).
        """
        if final_text:
            self.accumulated_text = final_text.strip()
            self.entities = _extract_entities(self.accumulated_text)
            self.intent = _extract_intent(self.accumulated_text)
        self.confidence = 1.0
        # is_correction reflects whether a correction happened mid-utterance;
        # it is preserved intentionally so the router can check it after finalize.


# ---------------------------------------------------------------------------
# HypothesisBuffer — per-session manager
# ---------------------------------------------------------------------------

class HypothesisBuffer:
    """
    Per-session hypothesis lifecycle manager.

    Usage::

        buf = HypothesisBuffer()
        h = buf.get_or_create(session_id, current_gen)
        h.update(chunk_text)

        # On interruption:
        buf.dispose(session_id)

        # Lazy staleness on next chunk:
        h = buf.get_or_create(session_id, new_gen)
        # → always returns a fresh hypothesis if generation changed
    """

    def __init__(self) -> None:
        self._states: dict[str, HypothesisState | None] = {}

    def get_or_create(self, session_id: str, current_generation: int) -> HypothesisState:
        """
        Return the existing hypothesis for the session if it is still current,
        otherwise create a fresh one stamped with current_generation.

        This is the lazy-invalidation path: if the runtime incremented the
        generation (due to interruption) but Module 1 hasn't yet explicitly
        disposed the hypothesis, this call catches the staleness.
        """
        existing = self._states.get(session_id)
        if existing is None or existing.generation != current_generation:
            existing = HypothesisState(generation=current_generation)
            self._states[session_id] = existing
        return existing

    def dispose(self, session_id: str) -> None:
        """
        Eagerly dispose the hypothesis for this session.

        Called on INTERRUPTION (competitive) so that the next TEXT_CHUNK
        starts with a clean slate even if get_or_create hasn't been called yet.
        """
        self._states[session_id] = None

    def peek(self, session_id: str) -> HypothesisState | None:
        """Return the current state without creating one. Used in tests."""
        return self._states.get(session_id)


# ---------------------------------------------------------------------------
# Non-city word blocklist
# ---------------------------------------------------------------------------
# Words that are capitalised in natural speech but are NOT city names.
# Used to reject false-positive "to X" destination matches.
# Deliberately small — only common false-positive categories:
#   calendar, common verbs, filler words.
# Keep in sync with extraction tests in test_hypothesis.py.

_NON_CITY_WORDS: frozenset[str] = frozenset({
    # Calendar / time
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march",
    "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "today", "tomorrow",
    "morning", "afternoon", "evening", "night",
    # Common filler / functional words that appear capitalised
    "actually", "book", "change", "check", "find", "make",
    "reserve", "search", "show", "stop", "cancel", "go",
    "get", "yes", "no", "ok", "okay", "please", "sure",
    "help", "use", "need", "want",
    # Common English words that can appear capitalised after a comma
    "the", "a", "an", "it", "that", "this", "my", "your",
    "i", "me", "we", "you",
    # Travel domain words that are NOT city names
    "flight", "flights", "hotel", "hotels", "train", "trains",
    "taxi", "car", "bus", "ticket", "tickets", "seat", "seats",
    "class", "first", "second", "third", "last", "next",
    "previous", "one", "two", "three",
    # Verbs / adjectives that can appear sentence-initial capitalised
    "wait", "right", "instead", "original", "another", "other",
})


def _is_plausible_city(word: str) -> bool:
    """
    True if `word` could be a city name.

    A word is NOT a plausible city if it appears in the blocklist of
    common non-city capitalised words.
    """
    return word.lower() not in _NON_CITY_WORDS


def _is_plausible_city_match(candidate: str) -> bool:
    """
    True if `candidate` (1 or 2 words extracted by _CITY_2W patterns) could be
    a real city name.

    For 2-word candidates (e.g. 'Los Angeles', 'New York'), BOTH words must be
    individually plausible — neither word may be in the non-city blocklist.
    For 1-word candidates, defers to _is_plausible_city().

    This guard eliminates false positives introduced by the _CITY_2W upgrade:
    - 'First Class'  → First is blocked → False
    - 'Book Flight'  → Book is blocked  → False
    - 'Los Angeles'  → both pass        → True
    - 'New York'     → both pass        → True
    """
    parts = candidate.split()
    return all(_is_plausible_city(p) for p in parts)


# ---------------------------------------------------------------------------
# Extraction helpers — deterministic, O(n), no LLM
# ---------------------------------------------------------------------------

# Patterns for travel-domain entities (matches hackathon mock tools).
# Deliberately simple; replace with a real NLU call if a slow-path model
# is available. These patterns run on every TEXT_CHUNK so they MUST be fast.

# A single title-case word (city names are typically one or two words).
_CITY_1W = r'([A-Z][a-zA-Z]+)'
# Two-word city variant for "New York", "Los Angeles" etc.
_CITY_2W = r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'

# ---------------------------------------------------------------------------
# Origin patterns — all require explicit "from <City>" structure
# with context boundary to prevent "from Seoul to" → "Seoul to".
# ---------------------------------------------------------------------------

_ORIGIN_PATTERNS = [
    # "from Seoul to" — city bounded by next preposition
    re.compile(
        r'\bfrom\s+' + _CITY_2W + r'(?=\s+(?:to|on|via|at|and)\b|[,.]|\s*$)',
        re.IGNORECASE,
    ),
    # Fallback: "from Seoul" at end of phrase (no trailing preposition)
    re.compile(r'\bfrom\s+' + _CITY_1W + r'\b', re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Destination — CORRECTION patterns (high specificity)
#
# These require explicit slot-modification verbs:
#   "change it to X", "make it X", "switch to X", etc.
# They will NOT match "change your mind", "actually book it", or
# "actually check the schedule" because:
# - "change your mind": "your" is not a capitalised city-word
# - "actually book it": "actually" pattern requires go/make/change + city
# - "book it": "it" is not capitalised
# ---------------------------------------------------------------------------

_DEST_CORRECTION_PATTERNS = [
    # "change it to X" / "change to X"  (Candidate B: upgraded to _CITY_2W)
    re.compile(r'\bchange\s+(?:it\s+)?to\s+' + _CITY_2W + r'\b', re.IGNORECASE),
    # "make it X"  (Candidate B: upgraded to _CITY_2W)
    re.compile(r'\bmake\s+it\s+' + _CITY_2W + r'\b', re.IGNORECASE),
    # "actually go to X" / "actually make it X" / "actually change it to X"
    re.compile(
        r'\bactually\s+(?:go\s+to|make\s+it|change\s+(?:it\s+)?to)\s+' + _CITY_2W + r'\b',
        re.IGNORECASE,
    ),
    # "switch it to X" / "switch to X"  (Candidate B: upgraded to _CITY_2W)
    re.compile(r'\bswitch\s+(?:it\s+)?to\s+' + _CITY_2W + r'\b', re.IGNORECASE),
    # "instead go to X" / "instead of going to X" / "instead X"  (forward)
    re.compile(r'\binstead\s+(?:go\s+to\s+|of\s+going\s+to\s+)?' + _CITY_2W + r'\b', re.IGNORECASE),
    # Candidate A — inverted instead: "Tokyo instead" / "No, Tokyo instead"
    # Natural speech pattern where the city precedes 'instead' rather than follows it.
    # Plausibility guard: _is_plausible_city() applied at extraction time.
    re.compile(r'\b' + _CITY_2W + r'\s+instead\b', re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Destination — GENERIC patterns (lower specificity)
#
# "to X" is a very common false-positive trigger. It is only used when no
# correction pattern matched, and the matched word passes _is_plausible_city().
# ---------------------------------------------------------------------------

_DEST_GENERIC_PATTERNS = [
    # "destination is X" / "destination to X"  (Candidate B: upgraded to _CITY_2W)
    re.compile(r'\bdestination\s+(?:is\s+|to\s+)?' + _CITY_2W + r'\b', re.IGNORECASE),
    # Generic "to X" — lower specificity, applied after correction patterns
    # Candidate B: upgraded to _CITY_2W to capture Los Angeles, New York etc.
    re.compile(r'\bto\s+' + _CITY_2W + r'\b', re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Date patterns (explicit, not city)
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    re.compile(
        r'\bon\s+((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday'
        r'|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?))',
        re.IGNORECASE,
    ),
    re.compile(r'\b(tomorrow|today|next\s+\w+)', re.IGNORECASE),
]


def _extract_entities(text: str) -> dict[str, str]:
    """Return best-guess entity dict from text."""
    entities: dict[str, str] = {}

    # Origin
    for pat in _ORIGIN_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip()
            if _is_plausible_city_match(candidate):
                entities["origin"] = candidate
            break

    # Destination: correction patterns take priority over generic 'to X'.
    # Within each group, the LAST match wins (most recent text wins).
    dest_val: str | None = None
    for pat in _DEST_CORRECTION_PATTERNS:
        for m in pat.finditer(text):
            candidate = m.group(1).strip()
            if _is_plausible_city_match(candidate):
                dest_val = candidate
    if dest_val is None:
        for pat in _DEST_GENERIC_PATTERNS:
            for m in pat.finditer(text):
                candidate = m.group(1).strip()
                if _is_plausible_city_match(candidate):
                    dest_val = candidate
    if dest_val is not None:
        entities["destination"] = dest_val

    # Date
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            entities["date"] = m.group(1).strip()
            break

    return entities


def _extract_intent(text: str) -> str | None:
    """Return best-guess intent label from text."""
    lower = text.lower()
    if any(w in lower for w in ("book", "reserve", "flight", "ticket")):
        return "book_flight"
    if any(w in lower for w in ("search", "find", "look", "show")):
        return "search"
    if any(w in lower for w in ("cancel", "stop", "never mind", "forget")):
        return "cancel"
    if any(w in lower for w in ("change", "update", "modify", "actually", "instead")):
        return "correction"
    return None


def _detect_correction(prev: dict[str, str], curr: dict[str, str]) -> bool:
    """
    True if any entity value changed from a previously observed non-None value.

    This is the primary signal for TEXT_CHUNK-detected self-corrections.
    A correction requires:
    1. The entity was previously observed (prev has the key with a value)
    2. The new value is different (case-insensitive)

    This guard is necessary but not sufficient — the router also requires
    active work before submitting a synthetic INTERRUPTION (guard 2).
    """
    for key, new_val in curr.items():
        old_val = prev.get(key)
        if old_val is not None and old_val.lower() != new_val.lower():
            return True
    return False
