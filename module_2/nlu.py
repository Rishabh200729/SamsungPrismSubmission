"""
nlu.py
Still a rule-based placeholder (no real LLM call) — but meaningfully less
brittle than v1: handles synonyms, multi-word place names, numeric
quantities, and common relative-date phrasing. Swap for a real LLM-backed
extractor by implementing IntentExtractor.
"""

import re
from typing import Dict, Any, Tuple, Optional
from tool_manifest import ToolManifest

_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                                          # 2026-10-01
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b",
]
_RELATIVE_DATE_WORDS = ("today", "tomorrow", "day after tomorrow")

_QUANTITY_PATTERN = r"\b(\d+)\s*(?:seats?|people|persons|guests?|nights?|rooms?)\b"

_PLACE_PREPOSITIONS = ("to", "in", "at", "from", "near")

# Extend as your team's scenarios reveal more phrasing variety.
_SYNONYM_MAP = {"reserve": "book", "find": "search", "check": "get"}


class IntentExtractor:
    def extract(self, text: str, manifest: ToolManifest) -> Tuple[Optional[str], Dict[str, Any]]:
        raise NotImplementedError


class RuleBasedExtractor(IntentExtractor):
    def extract(self, text: str, manifest: ToolManifest) -> Tuple[Optional[str], Dict[str, Any]]:
        normalized = self._normalize(text.lower())
        intent = self._match_intent(normalized, manifest)
        if not intent:
            return None, {}
        return intent, self._extract_slots(text, intent, manifest)

    def _normalize(self, text_lower: str) -> str:
        for syn, canon in _SYNONYM_MAP.items():
            text_lower = re.sub(rf"\b{syn}\b", canon, text_lower)
        return text_lower

    def _match_intent(self, text_lower: str, manifest: ToolManifest) -> Optional[str]:
        best, best_score = None, 0
        for name in manifest.tools:
            score = sum(1 for k in name.lower().split("_") if k in text_lower)
            if score > best_score:
                best, best_score = name, score
        return best

    def _extract_slots(self, text: str, intent: str, manifest: ToolManifest) -> Dict[str, Any]:
        slots: Dict[str, Any] = {}
        required = manifest.required_slots_for(intent)

        date_val = self._find_date(text)
        if date_val:
            for r in required:
                if "date" in r.lower():
                    slots[r] = date_val

        m = re.search(_QUANTITY_PATTERN, text.lower())
        if m:
            for r in required:
                if any(k in r.lower() for k in ("seat", "guest", "night", "room", "count", "quantity")):
                    slots[r] = int(m.group(1))

        place = self._find_place(text)
        if place:
            for r in required:
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
        # Multi-word place names (e.g. "New York", "Los Angeles"), tried in
        # decreasing preposition specificity so "to" doesn't shadow "from".
        for prep in _PLACE_PREPOSITIONS:
            m = re.search(rf"\b{prep}\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", text)
            if m:
                return m.group(1)
        return None
