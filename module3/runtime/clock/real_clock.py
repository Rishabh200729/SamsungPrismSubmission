"""
module3/runtime/clock/real_clock.py

Real-time clock for production use.
"""

from __future__ import annotations

import asyncio
import time

from .base import Clock


class RealClock(Clock):
    """Wall-clock time implementation."""

    def now(self) -> float:
        return time.monotonic() * 1000.0

    async def sleep(self, duration_ms: float) -> None:
        await asyncio.sleep(duration_ms / 1000.0)
