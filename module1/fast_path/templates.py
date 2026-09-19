"""
module1/fast_path/templates.py

Non-committal filler templates for fast-path speech.

Design rules:
- Never assert facts (no "Your flight to Tokyo is being booked").
- Never make promises (no "I'll have that done in a second").
- Prefer brevity: under 6 words is ideal for the spoken channel.
- Select randomly within a category to avoid sounding robotic.
"""

from __future__ import annotations

import random

FILLER_TEMPLATES: dict[str, list[str]] = {
    # Emitted on END_OF_TURN while slow-path reasoning starts
    "acknowledgment": [
        "Got it!",
        "Sure thing.",
        "On it.",
        "Working on that.",
        "Let me check.",
        "One moment.",
    ],
    # Emitted during a long THINKING pause (if throttle allows)
    "thinking": [
        "Let me look into that.",
        "Give me just a second.",
        "Still working on it.",
        "Checking now.",
    ],
    # Emitted after a competitive interruption (barge-in)
    "barge_in_ack": [
        "Sure, let me adjust.",
        "Got it, changing that now.",
        "Okay, switching gears.",
        "Understood, let me redo that.",
    ],
    # Emitted after a self-correction is detected
    "correction_ack": [
        "Sure, let me adjust.",
        "Got it, updating that.",
        "Of course, let me change that.",
        "No problem, I'll update it.",
    ],
}


def select_filler(category: str) -> str:
    """
    Select a random filler from the given category.

    Falls back to the acknowledgment category if the category is unknown.
    """
    templates = FILLER_TEMPLATES.get(category, FILLER_TEMPLATES["acknowledgment"])
    return random.choice(templates)
