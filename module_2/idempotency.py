"""
idempotency.py
CHANGED FROM idempotency_ledger.py: Module 3's output gate now does the
actual exactly-once ENFORCEMENT (dedup by idempotency_key, staleness
rejection by generation). We don't need our own ledger of ISSUED /
PREPARED / COMMITTED anymore — that whole state machine lives in Module 3.

What's still ours: choosing a GOOD idempotency_key. Module 3 will derive
one for us if we omit it, but a deterministic key we control means retries
of the *same logical action* (same session, same intent, same filled
slots) collapse to the same key on purpose — which is exactly what we want
when a correction turn re-proposes a call that hasn't changed.
"""

import hashlib
import json
from typing import Dict, Any


def make_idempotency_key(session_id: str, generation: int, tool_name: str,
                          arguments: Dict[str, Any], attempt: int = 0) -> str:
    """
    Deterministic across retries of the identical action within the same
    generation AND the same attempt count. A NEW generation (post-
    interruption) naturally produces a different key, which is correct —
    Module 3 fences old generations separately anyway.

    attempt matters for a case generation alone doesn't cover: a call that
    genuinely FAILS (not interrupted) and the user asks to retry with the
    same slots. Without attempt in the key, that retry produces the exact
    same key as the failed attempt, and Module 3's gate — which dedupes by
    idempotency_key regardless of whether the prior attempt succeeded —
    would likely reject it as a duplicate, silently blocking a legitimate
    retry. The caller increments attempt each time a call fails (see
    SlotTracker.increment_retry).
    """
    stable_args = json.dumps(arguments, sort_keys=True)
    raw = f"{session_id}:{generation}:{tool_name}:{stable_args}:{attempt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
