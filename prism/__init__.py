"""
prism/__init__.py
Shared data types for the TRAX voice-transactional layer.

Design basis: Atomix (arXiv:2602.14849) effect taxonomy adapted to the voice domain.
All 12 tool names and classifications verified against benchmark_data_v2.json and mock_apis.py.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional


# ---------------------------------------------------------------------------
# TRP State Machine
# ---------------------------------------------------------------------------

class TRPState(Enum):
    """
    Transition Relevance Place state for the current user turn.
    Emitted by trp_gate.py.  Consumed by tool_dispatcher.py and barge_in.py.

    Transition map:
        LISTENING   ──(editing term detected)──► REPAIRING
        LISTENING   ──(silence ≥ 300 ms + complete)──► TRP_CONFIRMED
        REPAIRING   ──(silence ≥ 900 ms OR reparans complete)──► TRP_CONFIRMED
        REPAIRING   ──(new editing term)──► REPAIRING  (stay, reset timer)
        TRP_CONFIRMED ──(next turn)──► LISTENING
        ANY ──(barge-in while agent speaking)──► ABORTED
    """
    LISTENING      = auto()   # default: user speaking, no disfluency signal
    REPAIRING      = auto()   # editing term or syntactic incompletion detected
    TRP_CONFIRMED  = auto()   # utterance syntactically/pragmatically complete
    ABORTED        = auto()   # barge-in: purge everything


# ---------------------------------------------------------------------------
# Effect Classification  (Atomix §3.5 taxonomy, voice-domain adapted)
# ---------------------------------------------------------------------------

class EffectClass(Enum):
    """
    Atomix effect taxonomy applied to FDB-v3 mock APIs.

    READ_ONLY     → Atomix 'bufferable': execute speculatively, hold in provisional cache,
                    never write to telemetry log until TRP_CONFIRMED.
    COMPENSABLE   → Atomix 'externalized-reversible': execute only on TRP_CONFIRMED,
                    register compensation handler C_i in SagaCoordinator.
    IRREVERSIBLE  → Atomix 'irreversible': require explicit TRP gate before execution,
                    no programmatic rollback — emit audit entry on abort.
    """
    READ_ONLY    = auto()
    COMPENSABLE  = auto()
    IRREVERSIBLE = auto()


# ---------------------------------------------------------------------------
# Tool Call record
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """
    Represents one intercepted tool invocation from the realtime model.
    Created by ToolDispatcher.dispatch() for every tool the model emits.
    """
    name:          str              # exact function name as in benchmark_data_v2.json
    args:          dict             # raw kwargs from the model
    effect_class:  EffectClass
    call_id:       str              = field(default_factory=lambda: str(uuid.uuid4()))
    issued_at:     float            = field(default_factory=time.monotonic)
    issue_epoch:   int              = 0

    # Populated after speculative execution starts (READ_ONLY only)
    speculative_task:   Optional[asyncio.Task]   = field(default=None, repr=False)

    # Resolution future: awaited by dispatch(), resolved by on_trp_state_change()
    _resolution_future: Optional[asyncio.Future] = field(default=None, repr=False)

    # Populated on commit
    committed:     bool  = False
    committed_at:  Optional[float] = None
    result:        Optional[Any]   = None    # the API response dict
    correction_epoch: Optional[int] = None   # repair epoch when the model emitted the call


# ---------------------------------------------------------------------------
# Compensation Entry
# ---------------------------------------------------------------------------

@dataclass
class CompensationEntry:
    """
    A registered compensation handler for one committed mutation.
    Stored in SagaCoordinator._log after every COMPENSABLE/IRREVERSIBLE commit.
    Garcia-Molina & Salem (1987): executed in reverse insertion order on abort.
    """
    call_id:          str
    tool_name:        str
    compensation_fn:  Callable[[], Coroutine]   # zero-arg async callable (closure)
    args:             dict  = field(default_factory=dict)
    committed_at:     float = field(default_factory=time.time)
    compensated:      bool  = False
