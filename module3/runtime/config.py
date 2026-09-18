"""
module3/runtime/config.py

Runtime configuration.

All tunable parameters live here. The Runtime constructor accepts
a RuntimeConfig instance, making it trivial to create test configurations
without touching core runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """
    Configuration for a PRISM runtime instance.

    Attributes:
        deterministic_mode: When True, VirtualClock is used and all
            asyncio.sleep() calls in mock tools must go through the clock.
        input_queue_maxsize: 0 = unbounded (default for hackathon).
            Set > 0 to enable backpressure.
        output_queue_maxsize: Same as above.
        session_max_duration_ms: Maximum wall-clock time per scenario
            (120,000ms = 120s per challenge spec). Informational only;
            enforcement is the runner's responsibility.
        record_wall_clock: Whether to record real wall-clock time in
            trace entries alongside virtual time.
        log_level: Logging level string ("DEBUG", "INFO", "WARNING").
    """
    deterministic_mode: bool = True
    input_queue_maxsize: int = 0
    input_queue_drop_oldest: bool = False
    output_queue_maxsize: int = 0
    session_max_duration_ms: float = 120_000.0
    setup_budget_ms: float = 300_000.0
    record_wall_clock: bool = False
    log_level: str = "WARNING"

    # Stale-result protection
    enable_stale_result_protection: bool = True

    # Cancellation behavior
    cancel_on_interrupt: bool = True
    require_tool_manifest: bool = False


DEFAULT_CONFIG = RuntimeConfig()

TEST_CONFIG = RuntimeConfig(
    deterministic_mode=True,
    record_wall_clock=False,
    log_level="WARNING",
)

PRODUCTION_CONFIG = RuntimeConfig(
    deterministic_mode=False,
    record_wall_clock=True,
    log_level="INFO",
    require_tool_manifest=True,
)
