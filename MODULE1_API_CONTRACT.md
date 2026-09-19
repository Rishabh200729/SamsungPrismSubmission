# Module 1 — Fast-Path Control Layer: API Contract & Integration Specification

## 1. Public API Surface & Allowed Imports

External components, harness tests, and integrating modules must interact with Module 1 strictly via its public exports:

```python
# Canonical import
from module1 import FastPathRouter

# Optional domain and telemetry inspection types
from module1.fast_path.hypothesis import HypothesisBuffer, HypothesisState
from module1.fast_path.telemetry import FastPathTelemetry
from module1.fast_path.templates import FILLER_TEMPLATES, select_filler
```

> ⚠️ **Boundary Invariant:** External modules must NOT import or invoke private extraction functions (`_extract_entities`, `_extract_intent`, `_detect_correction`, `_is_plausible_city_match`) or private handler callbacks (`_on_text_chunk`, `_on_end_of_turn`, `_on_interruption`).

---

## 2. Component Specifications

### 2.1 `FastPathRouter`

`FastPathRouter` is the primary entry point and orchestrator of the Module 1 fast path.

#### Constructor

```python
FastPathRouter(runtime: Runtime, clock: Clock) -> None
```

- **Parameters:**
  - `runtime` (`module3.runtime.Runtime`): Active Module 3 runtime instance.
  - `clock` (`module3.runtime.clock.Clock`): The clock used by the runtime (`VirtualClock` in tests, `SystemClock` in production).
- **Behavior & Invariants:**
  - Automatically registers three coroutine handlers on the runtime:
    - `InputEventType.TEXT_CHUNK` $\to$ `self._on_text_chunk`
    - `InputEventType.END_OF_TURN` $\to$ `self._on_end_of_turn`
    - `InputEventType.INTERRUPTION` $\to$ `self._on_interruption`
  - Initializes an isolated `HypothesisBuffer`.
  - Initializes in-memory `FastPathTelemetry`.
  - Sets up an empty session dictionary `_last_ack_ms` for interruption ack throttling.

#### Properties

- `telemetry` $\to$ `FastPathTelemetry`:
  Provides access to timing latency distributions.
- `hypothesis_buffer` $\to$ `HypothesisBuffer`:
  Provides access to the per-session provisional hypothesis manager.

---

### 2.2 `HypothesisBuffer` & `HypothesisState`

#### `HypothesisState`

Represents an ephemeral, generation-stamped provisional state for a single dialogue turn.

```python
@dataclass
class HypothesisState:
    generation: int = 1
    intent: str | None = None
    entities: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    chunk_count: int = 0
    accumulated_text: str = ""
    is_correction: bool = False
    _prev_entities: dict[str, str] = field(default_factory=dict, repr=False)
    _correction_submitted: bool = field(default=False, repr=False)
```

- **Methods:**
  - `update(text: str) -> None`:
    Appends chunk text to `accumulated_text`, re-extracts entities and intent, compares current entities against `_prev_entities`, and updates `is_correction`. Caps provisional confidence at $0.9$.
  - `finalize(final_text: str | None = None) -> None`:
    Completes the hypothesis. If `final_text` is provided, performs authoritative re-extraction. Sets `confidence = 1.0`.

#### `HypothesisBuffer`

Manages lifecycle and memory cleanup across sessions.

- **Methods:**
  - `get_or_create(session_id: str, current_generation: int) -> HypothesisState`:
    Returns active hypothesis if existing generation equals `current_generation`; otherwise lazily discards stale state and instantiates a new `HypothesisState(generation=current_generation)`.
  - `dispose(session_id: str) -> None`:
    Eagerly invalidates and clears the session state (`_states[session_id] = None`). Called upon competitive barge-in.
  - `peek(session_id: str) -> HypothesisState | None`:
    Non-mutating read of the current session hypothesis. Returns `None` if uninitialized or disposed.

---

### 2.3 `FastPathTelemetry`

Thread-safe, synchronous latency collector for fast-path processing.

- **Methods:**
  - `record(metric: str, value_ms: float) -> None`:
    Appends a millisecond timing delta to the named metric series.
  - `summary() -> dict[str, dict[str, Any]]`:
    Computes summary statistics lazily. Returns dictionary with schema:
    ```json
    {
      "<metric_name>": {
        "count": int,
        "p50": float,
        "p95": float,
        "max": float,
        "min": float
      }
    }
    ```
  - `reset() -> None`:
    Clears all metric collections.
  - `count(metric: str) -> int`:
    Returns total sample count for a given metric.

---

## 3. Event Contract & Invariants

### 3.1 Consumed Events

Module 1 expects standard Module 3 event schemas:

```python
# 1. TEXT_CHUNK
BaseEvent(
    session_id: str,
    timestamp_ms: float,
    event_type: InputEventType.TEXT_CHUNK,
    category: EventCategory.INPUT,
    payload: {"text": str}
)

# 2. END_OF_TURN
BaseEvent(
    session_id: str,
    timestamp_ms: float,
    event_type: InputEventType.END_OF_TURN,
    category: EventCategory.INPUT,
    payload: {"final_text": str | None}
)

# 3. INTERRUPTION
BaseEvent(
    session_id: str,
    timestamp_ms: float,
    event_type: InputEventType.INTERRUPTION,
    category: EventCategory.CONTROL,
    payload: {
        "competitive": bool,     # True = barge-in/cancel; False = backchannel
        "reason": str,           # e.g., "barge_in", "self_correction_detected"
        "text": str | None
    }
)
```

### 3.2 Emitted Events

#### `OutputEventType.FILLER` (Egress via `emit_output`)
```python
BaseEvent(
    session_id: str,
    timestamp_ms: float,
    event_type: OutputEventType.FILLER,
    category: EventCategory.OUTPUT,
    generation: None,            # MUST BE None (race-safe bypass)
    payload: {"text": str}
)
```
- **Priority:** Always emitted with `priority=10`.
- **Generation Invariant:** `generation` field is explicitly `None`. Any non-None generation stamp is a contract violation.

#### Synthetic `InputEventType.INTERRUPTION` (Ingress via `submit_event`)
```python
BaseEvent(
    session_id: str,
    timestamp_ms: float,
    event_type: InputEventType.INTERRUPTION,
    category: EventCategory.CONTROL,
    payload: {
        "text": str,
        "reason": "self_correction_detected",
        "competitive": True
    }
)
```
- **Lane Guarantee:** Handled via Module 3 control lane (unbounded, prioritized over data lane).

---

## 4. Integration Guarantees & Concurrency Rules

1. **Sub-Millisecond Execution:** Handlers never await external services, databases, or LLMs. Computation is purely local.
2. **Exactly-Once Correction Submission:** For any continuous sequence of revision chunks, `_correction_submitted` ensures at most **one** synthetic `INTERRUPTION` event is submitted before generation advancement.
3. **No Phantom Cancellations:** Module 1 never calls `submit_event(make_interruption(...))` unless `ctx.task_registry.active_tasks()` confirms running background tasks.
4. **Ordering-Independent Floor Consistency:** Module 1 never transitions floor to `LISTENING` unless current state is verified as `FloorState.INTERRUPTED`.
5. **Decoupled Throttling:** Turn-end thinking fillers and competitive barge-in acks do not block or suppress each other.
