"""
nlu.py
Rule-based intent/slot extractor for Module 2's Module3Adapter.

v3 — fixes a real bug found during integration testing, not just a
polish pass. The old `_match_intent` picked the tool with the highest
raw keyword-overlap score against the tool's own NAME tokens. That is
unsound whenever two tools share a word: "book_flight" and
"flight_search" both contain "flight", so "book a flight to Boston"
scored book_flight (matches "book" + "flight" = 2) higher than
flight_search (matches "flight" only = 1) — even though the user hasn't
given a flight_id yet and book_flight can't actually be called. The
fix below decides intent the way a human dispatcher would: which tool
can I actually call right now, not which tool's name I happen to have
said the most words of.

Design, in order of precedence:
  1. READINESS FIRST. For every tool whose name shares at least one
     token with the utterance, check whether its required slots are
     resolvable from THIS utterance. A tool with all required slots
     filled outranks a same-or-higher-scoring tool that isn't ready —
     this is what makes "book a flight to Boston" route to
     flight_search (destination resolvable) instead of book_flight
     (flight_id not resolvable from a bare city name).
  2. FULL-NAME-MATCH TIE-BREAK. Among tools tied on readiness, prefer
     the one whose name tokens are ALL present in the utterance (a
     "complete" mention) over one that only partially overlaps. This
     stops a single shared word ("flight") from being treated as
     equally strong evidence as a full match ("book" + "flight" both
     present).
  3. KEYWORD SCORE as the final tie-break, using tool-NAME tokens only.
     Description text is deliberately excluded from scoring — a tool's
     own description can legitimately mention another tool's name
     (e.g. "Book a flight returned by flight_search."), and scoring
     against that free text reintroduces exactly the collision bug
     this rewrite fixes.

Everything below is still plain regex/keyword matching — no LLM call —
consistent with the rest of Module 2's fallback-mode design. Swap this
out for a real LLM-backed extractor by implementing IntentExtractor;
the interface is unchanged from v2.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from tool_manifest import ToolManifest

# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")

_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                                          # 2026-10-01
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b",
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b",
]
_RELATIVE_DATE_WORDS = ("today", "tomorrow", "day after tomorrow", "tonight")
# Weekday/relative-date words must never be captured as a person's name —
# "Book it for Monday" is a date, not a passenger called "Monday".
_NON_NAME_WORDS = frozenset(_RELATIVE_DATE_WORDS) | {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_QUANTITY_PATTERN = re.compile(r"\b(\d+)\s*(?:seats?|people|persons|guests?|nights?|rooms?)\b")

_PLACE_PREPOSITIONS = ("to", "in", "at", "from", "near")
_NAME_MARKERS = ("for", "passenger", "name is", "under the name", "traveler")

# Words that legitimately follow "to"/"for" but are never place or person
# names — prevents "for Friday"/"to Cancel" style false extractions.
_NON_ENTITY_WORDS = _NON_NAME_WORDS | {
    "cancel", "book", "search", "flight", "flights", "hotel", "hotels",
    "class", "first", "second", "one", "two", "three",
    "me", "it", "that", "this", "the", "a", "an", "no", "yes", "my", "our",
    "on", "at", "in", "for", "from", "by", "with", "and", "or",
}

# Extend as your team's scenarios reveal more phrasing variety.
_SYNONYM_MAP = {"reserve": "book", "find": "search", "check": "get", "lookup": "search"}


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _stem(word: str) -> str:
    """Crude plural stemmer — good enough to match tool-name tokens
    ("flight") against user text ("flights") without pulling in a real
    NLP dependency."""
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


class IntentExtractor:
    def extract(self, text: str, manifest: ToolManifest) -> Tuple[Optional[str], Dict[str, Any]]:
        raise NotImplementedError


class RuleBasedExtractor(IntentExtractor):
    def extract(self, text: str, manifest: ToolManifest) -> Tuple[Optional[str], Dict[str, Any]]:
        normalized = self._normalize(text.lower())
        intent = self._match_intent(normalized, text, manifest)
        if not intent:
            return None, {}
        return intent, self._extract_slots(text, intent, manifest)

    # ------------------------------------------------------------------
    # Intent selection — see module docstring for why this isn't a bare
    # argmax over keyword overlap.
    # ------------------------------------------------------------------

    def _normalize(self, text_lower: str) -> str:
        for syn, canon in _SYNONYM_MAP.items():
            text_lower = re.sub(rf"\b{syn}\b", canon, text_lower)
        return text_lower

    def _name_tokens(self, tool_name: str) -> set:
        return {_stem(t) for t in tool_name.lower().split("_")}

    def _keyword_score(self, tool_name: str, stemmed_text_words: set) -> int:
        """Overlap against the tool's own NAME tokens only — never its
        description. See module docstring, point 3. `stemmed_text_words`
        must already be stemmed (see `_match_intent`) — stemming only one
        side of the comparison silently produces zero matches."""
        return len(self._name_tokens(tool_name) & stemmed_text_words)

    def _is_ready(self, tool_name: str, text: str, manifest: ToolManifest) -> bool:
        """True if every required slot for this tool can be resolved
        from the utterance text alone. This intentionally does NOT
        consult prior-turn state — Module 2's SlotTracker.update()
        already merges across turns; this check only decides which
        tool the CURRENT utterance is evidence for."""
        required = manifest.required_slots_for(tool_name)
        if not required:
            return True
        slots = self._extract_slots(text, tool_name, manifest)
        return all(r in slots for r in required)

    def _match_intent(self, normalized_text: str, raw_text: str,
                       manifest: ToolManifest) -> Optional[str]:
        text_words = {_stem(w) for w in _words(normalized_text)}
        candidates: List[Tuple[bool, bool, int, str]] = []

        for name in manifest.tools:
            score = self._keyword_score(name, text_words)
            if score == 0:
                continue
            full_match = score >= len(self._name_tokens(name))
            ready = self._is_ready(name, raw_text, manifest)
            candidates.append((ready, full_match, score, name))

        if not candidates:
            return None

        # Prefer full_match > score > ready, so a tool with full keyword
        # overlap isn't displaced by an unrelated tool with fewer required slots.
        candidates.sort(key=lambda c: (c[1], c[2], c[0]), reverse=True)
        return candidates[0][3]

    # ------------------------------------------------------------------
    # Slot extraction
    # ------------------------------------------------------------------

    def _extract_slots(self, text: str, intent: str, manifest: ToolManifest) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        required = manifest.required_slots_for(intent)
        target_slots = required or ("destination", "date", "passenger_name")

        date_val = self._find_date(text)
        if date_val:
            for r in target_slots:
                if "date" in r.lower():
                    slots[r] = date_val

        m = _QUANTITY_PATTERN.search(text.lower())
        if m:
            for r in target_slots:
                if any(k in r.lower() for k in ("seat", "guest", "night", "room", "count", "quantity")):
                    slots[r] = int(m.group(1))

        name_val = self._find_person_name(text)
        if name_val:
            for r in target_slots:
                if "passenger" in r.lower() or (r.lower().endswith("name")
                                                 and "device" not in r.lower()):
                    slots.setdefault(r, name_val)

        place = self._find_place(text)
        if not place:
            # Standalone city or destination reply (e.g. user just answers "Delhi" or "Chandigarh")
            words = [w for w in _words(text) if w not in _NON_ENTITY_WORDS and not self._find_date(w)]
            if 1 <= len(words) <= 2:
                place = " ".join(words).title()
        if place:
            for r in target_slots:
                if r.lower() in ("destination", "city", "location", "origin") and r not in slots:
                    slots[r] = place

        return slots

    def _find_date(self, text: str) -> Optional[str]:
        for pat in _DATE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(0)
        text_lower = text.lower()
        for word in _RELATIVE_DATE_WORDS:
            if word in text_lower:
                return word  # resolve to an actual date downstream if needed
        return None

    def _find_place(self, text: str) -> Optional[str]:
        # Multi-word place names (e.g. "New York", "Los Angeles", "delhi", "chandigarh")
        for prep in _PLACE_PREPOSITIONS:
            for m in re.finditer(rf"\b{prep}\s+([a-zA-Z]+)(?:\s+([a-zA-Z]+))?", text, re.IGNORECASE):
                w1 = m.group(1).strip()
                w2 = m.group(2)
                if w1.lower() in _NON_ENTITY_WORDS:
                    continue
                if w2 and w2.strip().lower() not in _NON_ENTITY_WORDS:
                    return f"{w1} {w2.strip()}".title()
                return w1.title()
        return None

    def _find_person_name(self, text: str) -> Optional[str]:
        for marker in _NAME_MARKERS:
            m = re.search(
                rf"\b{re.escape(marker)}\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", text, re.IGNORECASE
            )
            if m:
                candidate = m.group(1).strip()
                if candidate.lower() not in _NON_ENTITY_WORDS:
                    return candidate.title()
        return None
