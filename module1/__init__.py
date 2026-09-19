"""
module1 — Fast-Path Control Layer for PRISM

This module owns:
- Fast-path event classification (TEXT_CHUNK, END_OF_TURN, INTERRUPTION)
- Lightweight provisional hypothesis tracking
- Interruption / correction detection policy
- Filler / silence / acknowledgement policy
- Synthetic competitive INTERRUPTION submission for detected corrections
- Fast-path timing instrumentation

Module 1 does NOT own:
- Physical task cancellation        → Module 3 (request_cancellation)
- Generation increment              → Module 3 (CancellationCoordinator)
- State invalidation                → Module 3 (_handle_interruption)
- Call supersession                 → Module 3 (CallLedger)
- Stale-result rejection            → Module 3 (output gate)
- TOOL_RESULT handling              → Module 3 / Module 2
- Tool orchestration                → Module 2
"""

from .fast_path.router import FastPathRouter

__all__ = ["FastPathRouter"]
