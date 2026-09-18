"""
module3/runtime/clock/base.py

Abstract clock protocol.

All clock implementations must satisfy this interface. The runtime
accepts any Clock implementation — real or virtual — making it easy
to switch between production (real time) and test (deterministic) modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable


class Clock(ABC):
    """Abstract base clock for the PRISM runtime."""

    @abstractmethod
    def now(self) -> float:
        """Return current time in milliseconds."""
        ...

    @abstractmethod
    async def sleep(self, duration_ms: float) -> None:
        """Async sleep for duration_ms milliseconds."""
        ...
