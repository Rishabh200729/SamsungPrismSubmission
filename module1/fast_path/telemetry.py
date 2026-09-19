"""
module1/fast_path/telemetry.py

Fast-path timing instrumentation.

Records decision latency per handler invocation.
Stores measurements in-memory; exports a summary dict on request.

Design rules:
- Measurements are wall-clock deltas from clock.now().
- No I/O, no blocking, no asyncio overhead.
- The summary() method computes p50/p95/max lazily on demand.
"""

from __future__ import annotations

from typing import Any


class FastPathTelemetry:
    """
    Collects fast-path timing measurements.

    Usage::

        t0 = clock.now()
        # ... do work ...
        telemetry.record("fast_path_decision_ms", clock.now() - t0)

        report = telemetry.summary()
        # -> {"fast_path_decision_ms": {"count": N, "p50": ..., "p95": ..., "max": ...}}
    """

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def record(self, metric: str, value_ms: float) -> None:
        self._data.setdefault(metric, []).append(value_ms)

    def summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for metric, values in self._data.items():
            if not values:
                continue
            sv = sorted(values)
            n = len(sv)
            result[metric] = {
                "count": n,
                "p50": sv[n // 2],
                "p95": sv[int(n * 0.95)],
                "max": sv[-1],
                "min": sv[0],
            }
        return result

    def reset(self) -> None:
        """Clear all measurements. Useful between test runs."""
        self._data.clear()

    def count(self, metric: str) -> int:
        """Return the number of recorded measurements for a metric."""
        return len(self._data.get(metric, []))
