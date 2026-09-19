# Module 1 — Fast-Path Control Layer: Developer Overview

## Five-Minute Understanding

Module 1 is the lightweight, deterministic, real-time control plane of the agent system (PRISM GenAI Hackathon 3.0 — Theme 05: *Interruptible Real-Time Agents*). It processes incoming conversational text chunks in sub-millisecond time, detects user self-corrections mid-utterance, emits non-committal floor-holding acknowledgement fillers, and triggers canonical competitive interruptions when active work must be aborted.

- **What Module 1 does:** Evaluates incremental streaming tokens (`TEXT_CHUNK`), tracks provisional entity hypotheses, detects mid-utterance slot revisions, manages turn boundaries (`END_OF_TURN`), issues floor-holding speech fillers, and requests interruption when corrections collide with running background work.
- **Events it consumes:**
  - `InputEventType.TEXT_CHUNK` (data lane)
  - `InputEventType.END_OF_TURN` (data lane)
  - `InputEventType.INTERRUPTION` (control lane)
- **Outputs it produces:**
  - `OutputEventType.FILLER` (emitted via `runtime.emit_output()` with `generation=None`)
  - Synthetic `InputEventType.INTERRUPTION` (submitted via `runtime.submit_event()` with `competitive=True` and `reason="self_correction_detected"`)
- **How it detects corrections:** Deterministic, regex-based slot extraction against travel domain entities (`origin`, `destination`, `date`) and modification verbs. When an entity value changes between consecutive chunks within an utterance, `is_correction` is flagged. If active background tasks exist in Module 3's task registry, Module 1 submits a synthetic competitive interruption.
- **How it interacts with Module 3:** Module 1 registers asynchronous event handlers on the Module 3 `Runtime`. It is purely a *client* of Module 3: it never cancels tasks directly, never increments generations, and never touches session storage internals. Module 3 executes all physical cancellations, generation epoch bumps, and state fencing.
- **What it absolutely must NOT do:**
  - ❌ MUST NOT call `request_cancellation()`, `cancel_task()`, or `increment_generation()`.
  - ❌ MUST NOT call `state_store.invalidate()` or mutate session state directly.
  - ❌ MUST NOT stamp fillers with generation epochs (must use `generation=None`).
  - ❌ MUST NOT handle or consume `TOOL_RESULT` events.
  - ❌ MUST NOT call LLMs or perform blocking I/O on the fast path.
  - ❌ MUST NOT interact with `WorkerScheduler` or propose speculative tool actions.

---

## 1. Purpose

In human-agent conversational systems, barge-in interruptions and self-repairs occur naturally: a user begins asking for a flight to Tokyo, then abruptly pivots ("No, make it Osaka instead"). Standard monolithic LLM agent architectures cannot handle this safely:
1. Slow reasoning models take seconds to process new tokens.
2. Stale background queries continue running, wasting API budget and risking corrupted bookings.
3. Silence during processing causes the user to assume the agent crashed or did not hear.

Module 1 fulfills Theme 05's **Fast-Path Interruption Handling** requirements:
- **Fast Event Classification:** Evaluates incoming events in sub-millisecond time without waiting for LLM inference.
- **Correction Detection:** Detects mid-utterance slot value changes on incremental streaming chunks.
- **Provisional Hypothesis Tracking:** Maintains disposable, generation-stamped entity hypotheses during speech.
- **Floor-Holding Speech:** Emits low-latency, non-committal fillers ("One moment.", "Got it, changing that now.") to maintain conversational presence without committing to unverified facts.
- **Interruption Signalling:** Converts detected self-corrections into structured, competitive `INTERRUPTION` events routed into Module 3's high-priority control lane.
- **Fast-Path Telemetry:** Records decision latencies (p50, p95, max) for monitoring and evaluation.

---

## 2. Ownership Boundary

### Explicit Responsibilities

| Dimension | Owned by Module 1 | Owned by Module 3 (Runtime Kernel) | Owned by Module 2 / Module 4 |
|---|---|---|---|
| **Event Classification** | Fast-path routing of `TEXT_CHUNK`, `END_OF_TURN`, `INTERRUPTION` | Two-lane priority queueing, dispatch event loop | Slow-path reasoning, tool result interpretation, multimodal fusion |
| **Hypothesis Tracking** | Provisional, disposable entity slots (`HypothesisBuffer`) | None (unaware of domain hypotheses) | Authoritative conversation history & context |
| **Correction Detection** | Deterministic slot-change detection + active-work check | None | Semantic ambiguity resolution, slot repair planning |
| **Speech Policy** | Non-committal filler selection & ack throttling | Central output gate authorization & egress | Final answer generation, tool call proposals |
| **Cancellation Mechanics** | Synthetic `INTERRUPTION` submission only | Physical `asyncio.Task.cancel()`, monotonic generation epoch bump, call supersession, state store invalidation | Cooperative yielding at cancellation points |
| **Floor State** | Drives `LISTENING -> THINKING` on turn end; attempts `INTERRUPTED -> LISTENING` post-ack | Legal transition enforcement, forced `INTERRUPTED` on barge-in | None |
| **Tool Execution** | None (does NOT touch tools or `TOOL_RESULT`) | Tool manifest verification, idempotency ledger | Tool proposal, argument generation, execution |

### Why These Boundaries Exist

1. **Kernel Centralization vs. Edge Policy:** Concurrency control, generation epochs, and task lifecycles require a single source of truth. If Module 1 directly cancelled tasks or bumped generations, race conditions with Module 2's reasoning loop or Module 4's perceptual streams would be unavoidable.
2. **Untrusted Proposals:** Module 1 treats itself as an event producer. By submitting an `INTERRUPTION` event to `runtime.submit_event()`, it allows Module 3 to record the causal audit trace, notify all registered listeners concurrently, and maintain deterministic ordering under virtual clocks.
3. **Separation of Speed and Deliberation:** Module 1 must respond in < 1ms. It cannot afford LLM inference, disk I/O, or complex graph resolution. Deep reasoning is cleanly delegated to Module 2.

---

## 3. Architecture

### System Topology & Event Flow

```text
Incoming Events (Audio ASR / User Stream)
                   |
                   v
    +------------------------------+
    | Module 3 PriorityInputQueue  |
    |  - Control Lane (Urgent)     |
    |  - Data Lane (Streaming)     |
    +--------------+---------------+
                   |
                   v
         Module 3 Dispatcher
                   |
         +---------+---------+
         | (Concurrently)    |
         v                   v
+------------------+  +---------------------------------------+
|  FastPathRouter  |  | Module 3 Runtime Kernel               |
|    (Module 1)    |  |  - SessionManager / StateStore        |
+--------+---------+  |  - CancellationCoordinator            |
         |            |  - TaskRegistry / OutputGate          |
         |            +---------------------------------------+
         |
         +--> HypothesisBuffer (provisional entities, generation-stamped)
         +--> Correction Detection (slot change + active-work check)
         +--> Filler / Ack Policy (_ACK_THROTTLE_MS & filler templates)
         |
         +--> Emits: runtime.emit_output(make_filler(...), priority=10)
         |            [generation=None to prevent gate staleness races]
         |
         +--> Emits: runtime.submit_event(make_interruption(competitive=True))
                      |
                      v
              Module 3 Control Lane
                      |
                      v
         Runtime._handle_interruption
           * Physical cancel active tasks (asyncio.Task.cancel)
           * Increment generation epoch (N -> N+1)
           * state_store.invalidate(INTERRUPTED)
           * call_ledger.supersede_before(N+1)
           * Emits CANCELLATION outputs
                      |
                      +-------------------+
                      |                   |
                      v                   v
              +---------------+   +---------------+
              |   Module 2    |   |   Module 4    |
              | (Reasoning /  |   | (Multimodal   |
              |  Tool Calls)  |   |  Perception)  |
              +---------------+   +---------------+
```

### Module Cross-Communication Interfaces

- **Module 1 <-> Module 3:** Module 1 registers handlers via `runtime.register_handler()`. It consumes events passed by the dispatcher. It emits fillers via `runtime.emit_output()` and synthetic interruptions via `runtime.submit_event()`.
- **Module 1 <-> Module 2:** Strictly decoupled. Module 1 does not call Module 2. When Module 1 detects a correction and submits an `INTERRUPTION`, Module 3 cancels Module 2's in-flight task. Module 2 picks up the revised utterance at the next `END_OF_TURN`.
- **Module 1 <-> Module 4:** Strictly decoupled. When Module 1 triggers an interruption, Module 3 cancels active perceptual reasoning tasks. Module 1 does not ingest `VIDEO_FRAME` or `AUDIO_WAV`.

---

## 4. Event Flow

### Matrix of Behaviors

| Scenario | Input Event | Module 1 Fast-Path Action | Module 3 Kernel Action | Output Produced | State / Generation Ownership |
|---|---|---|---|---|---|
| **A. TEXT_CHUNK (no correction)** | `TEXT_CHUNK(text="book flight to Tokyo")` | Appends text to provisional hypothesis; updates slots (`destination="Tokyo"`); no correction flagged | Dispatches chunk to registered data-lane handlers | *None* (stay silent while user speaks) | Generation unchanged. Hypothesis owned by Module 1. |
| **B. END_OF_TURN** | `END_OF_TURN(final_text=...)` | Finalizes hypothesis (`confidence=1.0`); transitions floor `LISTENING -> THINKING`; checks filler throttle (`can_emit_filler`) | Dispatches event to slow-path reasoners (Module 2) | `FILLER` ("One moment.", "Working on that.") if throttle permits | Generation unchanged. Floor updated in Module 3 `StateStore`. |
| **C. Competitive INTERRUPTION** | `INTERRUPTION(competitive=True, reason=...)` | Disposes hypothesis eagerly; checks ack throttle (`_can_emit_ack`); emits ack filler; conditionally updates floor `INTERRUPTED -> LISTENING` | Physical task cancellation (`Task.cancel`); generation epoch incremented ($N \to N+1$); floor invalidated to `INTERRUPTED`; pending calls superseded | `FILLER` ("Sure, let me adjust.") + `CANCELLATION` events from Module 3 | Generation incremented by Module 3. Session state invalidated by Module 3. |
| **D. Non-Competitive INTERRUPTION (Backchannel)** | `INTERRUPTION(competitive=False, text="uh-huh")` | Detects `competitive=False`; immediately returns (no-op); preserves active hypothesis; no filler emitted | Skips cancellation; preserves running tasks; preserves generation epoch | *None* | Generation and state remain completely unchanged. |
| **E. TEXT_CHUNK with Correction** | `TEXT_CHUNK(text="No, Osaka instead")` | Detects slot change (`Tokyo` $\to$ `Osaka`); verifies active work in `task_registry`; sets `_correction_submitted=True`; submits synthetic competitive `INTERRUPTION` | Receives synthetic `INTERRUPTION` on control lane; triggers full cancellation sequence (as in Scenario C) | Synthetic `INTERRUPTION` submitted; subsequently produces ack `FILLER` and `CANCELLATION` events | Generation incremented by Module 3. Hypothesis reset on interruption dispatch. |
| **F. Repeated Correction Chunks** | Rapid consecutive `TEXT_CHUNK`s revising slots | First chunk submits synthetic `INTERRUPTION` and sets `_correction_submitted=True`; subsequent chunks on same hypothesis skip submission | Module 3 processes first interruption; `_interrupt_guard` prevents duplicate re-entrant cancellation within same cycle | Exactly ONE synthetic `INTERRUPTION` emitted per correction episode | Module 3 strictly guards generation increments; Module 1 guards submission. |
| **G. Stale Results Arriving Late** | `TOOL_RESULT` from cancelled generation $N$ | **Ignored** (Module 1 registers NO handler for `TOOL_RESULT`) | Output gate rejects any output stamped with generation $N < N_{curr}$; `is_stale_result()` flags result | *None* (silently dropped or logged to trace) | Module 3 owns output gate authorization. |

---

## 5. FastPathRouter

### Implementation Details (`module1/fast_path/router.py`)

`FastPathRouter` is the single entry point for Module 1.

#### Constructor

```python
class FastPathRouter:
    def __init__(self, runtime: "Runtime", clock: "Clock") -> None:
        self._runtime = runtime
        self._clock = clock
        self._hypothesis = HypothesisBuffer()
        self._telemetry = FastPathTelemetry()
        self._last_ack_ms: dict[str, float] = {}

        # Register handlers against Module 3 runtime
        runtime.register_handler(InputEventType.TEXT_CHUNK, self._on_text_chunk)
        runtime.register_handler(InputEventType.END_OF_TURN, self._on_end_of_turn)
        runtime.register_handler(InputEventType.INTERRUPTION, self._on_interruption)
```

#### Synchronous vs. Asynchronous Boundaries

- **Synchronous & Deterministic:**
  - Token parsing, regex entity extraction, intent detection, and slot comparison (`HypothesisState.update()`).
  - Active task inspection (`ctx.task_registry.active_tasks()`).
  - Throttle checks (`_can_emit_ack()` and `state_store.can_emit_filler()`).
  - Hypothesis disposal (`self._hypothesis.dispose()`).
  - Telemetry recording (`self._telemetry.record()`).
- **Asynchronous Yield Points:**
  - `await self._runtime.submit_event(...)`: Submitting synthetic competitive interruptions to the runtime.
  - `await self._runtime.emit_output(filler_event, priority=10)`: Emitting acknowledgement fillers to the output gate.

#### Why Module 1 Uses Canonical Runtime Methods

- **Uses `runtime.submit_event()`:** Submitting synthetic interruptions through the standard event transport guarantees that the event is logged in Module 3's append-only causal trace, routed to the priority control lane, and dispatched to all modules consistently.
- **Uses `runtime.emit_output()`:** Emitting fillers through the central output gate ensures that session validity, rate limiting, and output serialization are maintained.
- **Does NOT use `request_cancellation()`:** Physical cancellation and generation advancement are kernel operations owned solely by Module 3's `_handle_interruption`. Calling it directly would bypass the control lane and cause race conditions with other registered listeners.
- **Does NOT use `WorkerScheduler`:** Speculative action proposal is explicitly disabled (`SPECULATION_ENABLED = False`). Module 1 focuses exclusively on control flow and speech acts.

---

## 6. HypothesisBuffer & HypothesisState

### Implementation Details (`module1/fast_path/hypothesis.py`)

`HypothesisBuffer` manages ephemeral, per-session conversational state across turns.

#### `HypothesisState` Fields

```python
@dataclass
class HypothesisState:
    generation: int = 1
    intent: Optional[str] = None
    entities: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    chunk_count: int = 0
    accumulated_text: str = ""
    is_correction: bool = False
    _prev_entities: dict[str, str] = field(default_factory=dict, repr=False)
    _correction_submitted: bool = field(default=False, repr=False)
```

#### Lifecycle & Invalidation Mechanics

1. **Session Isolation:** `HypothesisBuffer` stores states in a private dictionary keyed by `session_id`. No state or memory leaks across sessions.
2. **Lazy Staleness Invalidation:** When `get_or_create(session_id, current_generation)` is called, it compares the stored hypothesis generation against `current_generation`. If Module 3 incremented the generation due to an interruption, the old hypothesis is discarded and a fresh `HypothesisState` is instantiated.
3. **Eager Invalidation:** Upon receiving a competitive `INTERRUPTION`, `_on_interruption` calls `self._hypothesis.dispose(session_id)`, immediately clearing the stored state.
4. **HIGH-2 Guard (`_correction_submitted`):** When a mid-utterance correction triggers a synthetic interruption, `_correction_submitted` is set to `True` before yielding to `submit_event()`. Any subsequent streaming chunk arriving before the generation advances is prevented from submitting duplicate interruptions.
5. **Provisional vs. Authoritative:** Hypotheses are strictly provisional guides for fast-path decisions. They are not stored in Module 3's authoritative `StateStore`. Authoritative slot values are extracted by Module 2 only after `END_OF_TURN` is reached.

#### Connection to Chronological Hypothesis Processing

Following the principles of Chronological Thinking (arXiv:2510.05150), human speech is linear and causal. Streaming hypotheses must represent the cumulative prefix of an utterance, updating beliefs monotonically until a repair or boundary token occurs.

---

## 7. Correction Detection

### Deterministic Mechanism

Module 1 does not use non-deterministic LLMs for fast-path classification. Entity and correction extraction is deterministic, keyword- and regex-based, executing in $\mathcal{O}(N)$ time relative to text length (< 0.1ms).

#### Two-Tier Pattern Hierarchy

1. **Destination Correction Patterns (High Specificity):**
   - Matches explicit slot-modification verbs:
     - `change [it] to <City>`
     - `make it <City>`
     - `actually [go to / make it / change it to] <City>`
     - `switch [it] to <City>`
     - `instead [go to / of going to] <City>` (forward instead)
     - `<City> instead` (Candidate A fix: inverted instead, e.g., "Tokyo instead", "No, Tokyo instead")
2. **Destination Generic Patterns (Lower Specificity):**
   - `destination [is / to] <City>`
   - `to <City>` (fallback, applied only if no correction pattern matched)
3. **Multi-Word City Handling (Candidate B fix):**
   - Uses `_CITY_2W = r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'` across all destination patterns to capture multi-word cities (e.g., "Los Angeles", "New York", "San Francisco").
4. **Plausibility Filters & Non-City Blocklist:**
   - To prevent false positives from capitalized non-city words, `_is_plausible_city_match()` verifies that **every** word in a candidate match does not appear in `_NON_CITY_WORDS`:
     - Calendar terms: `monday`, `tuesday`, `today`, `tomorrow`, `morning`, etc.
     - Action verbs & fillers: `actually`, `book`, `change`, `check`, `find`, `make`, `reserve`, `stop`, `cancel`, `please`.
     - Domain nouns & modifiers: `flight`, `hotel`, `ticket`, `class`, `first`, `second`, `third`, `seat`.
     - Pronouns & articles: `the`, `this`, `that`, `you`, `i`.

#### Dual-Guard Protection Against False Interruptions

A synthetic competitive `INTERRUPTION` is emitted if and only if **BOTH** guards pass:
- **Guard 1 (Semantic Revision):** An existing entity slot had a non-empty value, and the new text changed that slot's value to a different valid entity (`_detect_correction() == True`).
- **Guard 2 (Active Work Required):** The session has running background tasks in Module 3's `TaskRegistry` (`bool(list(ctx.task_registry.active_tasks())) == True`). If no work is in-flight, no interruption is triggered because there is nothing to abort.

#### Concrete Classification Examples

| User Utterance | Detected As | Rationale |
|---|---|---|
| `"No, Tokyo instead"` | **TRUE CORRECTION** | Inverted instead pattern matches "Tokyo"; slot changes from previous destination |
| `"Actually, make it Osaka"` | **TRUE CORRECTION** | Correction verb pattern `make it Osaka` extracts destination "Osaka" |
| `"Change the destination to Los Angeles"` | **TRUE CORRECTION** | Multi-word pattern captures "Los Angeles"; both words plausible |
| `"actually book it"` | **NOT CORRECTION** | "it" is not a city; "book" is in blocklist; no slot modification |
| `"change your mind"` | **NOT CORRECTION** | "your" is in blocklist; not a city name |
| `"Book Monday"` | **NOT CORRECTION** | "Monday" is in non-city blocklist; rejected by plausibility check |
| `"I changed my mind"` | **NOT CORRECTION** | No destination slot specified; no entity extracted |
| `"to First Class"` | **NOT CORRECTION** | "First" and "Class" are in blocklist; rejected by `_is_plausible_city_match` |

> ⚠️ **Limitation Notice:** This rule-based extractor is tailored for travel-domain slot repairs (flights, destinations, dates). Complex, non-standard linguistic repairs without keyword markers are deferred to slow-path models.

---

## 8. Filler & Acknowledgement Policy

### Filler Types & Purpose

Fillers are brief vocalizations emitted to manage conversational floor and reassure the user that the system is responsive. Module 1 implements three distinct filler paths:

1. **Turn-End Acknowledgement (`acknowledgment`):** Emitted upon `END_OF_TURN` while slow-path reasoning begins ("Got it!", "One moment.", "Working on that.").
2. **Competitive Interruption Acknowledgement (`barge_in_ack` / `correction_ack`):** Emitted immediately upon a competitive barge-in or detected self-correction ("Sure, let me adjust.", "Got it, changing that now.").
3. **Non-Competitive Backchannel:** **Zero fillers emitted.** Emitting a filler for a non-competitive backchannel ("uh-huh") would interrupt the user while they are speaking.

### Non-Committal Template Design

All templates in `module1/fast_path/templates.py` strictly adhere to the non-committal rule:
- ❌ **Forbidden:** Factual claims or commitments ("Booking your flight to Tokyo now", "I found 3 seats").
- ✅ **Allowed:** Procedural acknowledgements ("Working on that", "Checking now", "Sure, let me adjust").
- **Rationale:** Fillers are emitted before reasoning or tool execution occurs. If a filler makes a factual promise that fails later, user trust is destroyed. Fillers are **not** authoritative task outputs.

### Dual-Throttle Mechanism

To prevent verbal stuttering, Module 1 uses two completely independent throttle windows:
- **Normal Filler Throttle (`state_store.filler_window_ms = 3000ms`):** Governs standard `END_OF_TURN` thinking fillers. Enforced in Module 3's `StateStore`.
- **Dedicated Interruption Ack Throttle (`_ACK_THROTTLE_MS = 500ms`):** Owned by Module 1 in `FastPathRouter._last_ack_ms`. Ensures that a competitive interruption acknowledgement is **never blocked** by an `END_OF_TURN` filler emitted moments before (Fix HIGH-1).

### Race-Safe `generation=None` Semantics

When emitting fillers via `runtime.emit_output()`, Module 1 **omits the generation stamp** (`generation=None`):

```python
filler_event = make_filler(
    session_id=session_id,
    timestamp_ms=timestamp_ms,
    text=text,
    # generation intentionally omitted -> defaults to None
)
await self._runtime.emit_output(filler_event, priority=10)
```

**Why this is necessary:**
When a competitive interruption occurs, `_on_interruption` (Module 1) and `_handle_interruption` (Module 3) run concurrently via `asyncio.gather()`. If Module 1 attempted to stamp the filler with `runtime.get_current_generation()`, it could read generation $N$ right before Module 3 advanced the epoch to $N+1$. When the filler reached the output gate, the gate would reject it as stale ($N \neq N+1$). By using `generation=None`, the filler bypasses generation staleness fencing. This is semantically correct because an acknowledgement is an immediate speech act, not a cached task result.

---

## 9. Floor Management

Module 1 interacts with Module 3's authoritative 6-state floor machine:
$$\text{LISTENING} \longleftrightarrow \text{THINKING} \longleftrightarrow \text{SPEAKING} \longleftrightarrow \text{INTERRUPTED} \longleftrightarrow \text{WAITING\_FOR\_USER / TOOL}$$

### Exact Transitions Driven by Module 1

1. **`LISTENING -> THINKING` on `END_OF_TURN`:**
   - When the user finishes speaking, Module 1 invokes `ctx.state_store.update_floor(FloorState.THINKING)`.
   - If the floor is already in another state (e.g., already interrupted), `ValueError` is safely caught and ignored.
2. **`INTERRUPTED -> LISTENING` Post-Interruption (Hardened Ordering-Independence):**
   - In `_on_interruption`, Module 1 inspects `ctx.state_store.snapshot.floor_state`.
   - If the floor is **already** `INTERRUPTED` (Module 3's `invalidate()` has already executed), Module 1 transitions the floor to `LISTENING`, preparing the agent to receive the user's new instructions.
   - If the floor is **not yet** `INTERRUPTED` (Module 1 executed before Module 3's invalidate step), Module 1 **defers**: it leaves the floor alone, allowing Module 3's `invalidate()` to set `INTERRUPTED`. The subsequent incoming `TEXT_CHUNK` will legally transition the floor.

---

## 10. Generation and Cancellation Contract

### Architectural Concepts

- **Generation ($1, 2, 3\dots$):** Answers *"Is this conversational work still current?"* Monotonically incremented on every competitive interruption.
- **`call_id`:** Answers *"Which exact tool invocation is this?"* Unique per tool execution.

### Two-Tier Cancellation Mechanism

Module 3 and Module 1 enforce cancellation through two complementary layers:
1. **Active Cancellation (`asyncio.Task.cancel()`):** Module 3 issues cooperative physical cancellation to in-flight coroutines. Coroutines yield at `await` boundaries and terminate.
2. **Passive Generation Fencing (Staleness Invalidation):** Because network I/O or native threads may not cancel instantly, `Task.cancel()` cannot be relied upon as the sole correctness guarantee. Module 3 advances the generation epoch and configures the output gate to drop any result stamped with an older generation. Even if a cancelled task finishes executing, its output is intercepted and neutralized at the gate.

### Module 1's Role

Module 1 is strictly a reader of generation epochs:
- It queries `get_current_generation()` to stamp provisional hypotheses.
- It **never** calls `increment_generation()`.
- It **never** issues physical cancellation calls.

---

## 11. Backchannel Contract

A conversational backchannel occurs when a user utters a brief non-competitive confirmation ("uh-huh", "yeah", "okay") while the agent is speaking or thinking.

### `competitive=False` Contract

When an `INTERRUPTION` event arrives with `competitive=False`:
- **Module 3:** Skips task cancellation, skips generation increment, preserves active tasks.
- **Module 1:**
  - Preserves the active hypothesis in `HypothesisBuffer` without disposal.
  - Emits **zero** fillers or speech acts.
  - Makes **zero** floor transitions.
  - Records timing telemetry under `backchannel_ms`.
  - Exits immediately.

---

## 12. Testing

The Module 1 test suite comprises **119 dedicated tests** organized across four primary test files:

| Test File | Test Count | Focus Area |
|---|---|---|
| `module1/tests/test_router.py` | 15 tests | Integration matrix against live Module 3 runtime (matrix scenarios 1–15) |
| `module1/tests/test_hypothesis.py` | 45 tests | Entity extraction, inverted instead, multi-word cities, confidence scoring |
| `module1/tests/test_hardening.py` | 44 tests | Concurrency races, floor ordering independence, false positive protections |
| `module1/tests/test_high_fixes.py` | 15 tests | Regression tests for HIGH-1 (ack throttle separation) and HIGH-2 (`_correction_submitted` guard) |

### Test Philosophy

- **Real Module 3 Runtime:** Tests do not mock Module 3 internals. They instantiate real `Runtime`, `StateStore`, `TaskRegistry`, and `PriorityInputQueue` instances.
- **Virtual Time:** Tests use Module 3's `VirtualClock` for deterministic, millisecond-accurate scheduling without real-world sleeps.
- **No Overclaiming:** Tests verify protocol conformance, race safety, and regex extraction. They do not simulate speech acoustics or ASR phonetic distortions.

---

## 13. Telemetry

`FastPathTelemetry` (`module1/fast_path/telemetry.py`) provides zero-overhead, in-memory timing instrumentation.

### Tracked Metrics

- `fast_path_decision_ms`: Latency of standard `TEXT_CHUNK` evaluation (no correction).
- `correction_detected_ms`: Latency from chunk ingestion to synthetic `INTERRUPTION` submission.
- `end_of_turn_ms`: Latency of `END_OF_TURN` processing and floor transition.
- `interruption_ack_ms`: Latency of competitive interruption handling and ack emission.
- `backchannel_ms`: Latency of non-competitive backchannel filtering.

### Aggregation & Access

Latency percentiles are computed lazily on demand via `telemetry.summary()`:

```python
summary = router.telemetry.summary()
# Returns:
# {
#     "fast_path_decision_ms": {"count": 100, "p50": 0.04, "p95": 0.08, "max": 0.15, "min": 0.01},
#     "interruption_ack_ms":   {"count": 2,   "p50": 0.09, "p95": 0.12, "max": 0.12, "min": 0.07}
# }
```

### Telemetry Scope & Limitations

- Measurements reflect local CPU execution time using `clock.now()` deltas.
- Telemetry is in-memory and resets on session close. It does not export to external APMs (Prometheus/OpenTelemetry) in v1.

---

## 14. Research Mapping

The Module 1 design directly translates principles from recent conversational AI and interruption research into concrete code:

| Research Source | Transferable Principle | Current Module 1 Implementation | Status |
|---|---|---|---|
| **PASTE**<br>(arXiv:2603.18897) | Streaming chunk-level intent and slot revision tracking | `HypothesisState.update()` tracks streaming token prefixes and flags slot value modifications | **Implemented** |
| **Cost-Aware Speculative Execution**<br>(arXiv:2606.07846) | Speculative tool execution triggered on high-confidence provisional prefix | `SPECULATION_ENABLED = False`. Speculation disabled for baseline stability; extension point preserved | **Deferred / Future Work** |
| **Full-Duplex-Bench-v3**<br>(arXiv:2604.04847) | Structural discrimination between competitive barge-in and non-competitive backchannel | `competitive: bool` flag handling in `_on_interruption`; backchannels bypass cancellation and fillers | **Implemented** |
| **Chronological Thinking**<br>(arXiv:2510.05150) | Causal linear hypothesis tracking with chronological invalidation | `HypothesisBuffer` generation stamping, eager disposal on interruption, lazy invalidation on epoch advance | **Implemented** |
| **DDTSR**<br>(arXiv:2602.23266) | Dynamic dialogue-turn speech repair detection | Deterministic regex repair patterns (inverted instead, multi-word cities, slot change detection) | **Partially Implemented** (Rule-based) |
| **Dual-Axis Reward Model**<br>(arXiv:2604.14920) | Balancing turn-taking responsiveness with low semantic hallucination risk | Non-committal filler templates in `templates.py` containing no factual assertions | **Implemented** |

---

## 15. Current Limitations

1. **Acoustic Hesitation Classifier Not Implemented:** Module 1 classifies text tokens. Acoustic features (pitch drops, elongation, energy dip) indicating hesitation vs. completion require an upstream acoustic VAD/DSP model.
2. **Speculative Tool Calls Disabled:** `SPECULATION_ENABLED` is hardcoded to `False`. Speculative tool execution will be evaluated after slow-path reasoning is fully integrated.
3. **Domain-Specific Entity Extraction:** The current fast extractor is optimized for travel/reservation scenarios (`origin`, `destination`, `date`). Broader open-domain slot extraction requires extending the pattern definitions or integrating a distilled local SLU model.
4. **Context-Free Repair Detection:** Repair detection operates on lexical slot deltas within an utterance prefix. Complex pragmatic corrections that do not mention explicit entity names are left to Module 2.
5. **In-Memory Telemetry:** Telemetry metrics are stored in heap memory without background thread flushing or remote metrics export.

---

## 16. Integration Guide

### Canonical Setup for Developers

```python
import asyncio
from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.clock import VirtualClock
from module3.runtime.events import make_text_chunk, make_end_of_turn
from module1 import FastPathRouter

async def main():
    # 1. Initialize clock and Module 3 runtime
    clock = VirtualClock()
    runtime = Runtime(config=TEST_CONFIG, clock=clock)
    
    # 2. Instantiate Module 1 FastPathRouter (registers handlers automatically)
    router = FastPathRouter(runtime, clock)
    
    # 3. Start the runtime event loop
    await runtime.start()
    
    # 4. Create an isolated session
    session_id = runtime.create_session("session-001")
    
    # 5. Ingest streaming speech events
    await runtime.submit_event(make_text_chunk(
        session_id=session_id,
        timestamp_ms=clock.now(),
        text="Book a flight from Seoul to Tokyo",
    ))
    
    # 6. Read telemetry or inspect hypothesis state
    hypothesis = router.hypothesis_buffer.peek(session_id)
    print("Provisional destination:", hypothesis.entities.get("destination"))
    
    # 7. Complete the turn
    await runtime.submit_event(make_end_of_turn(
        session_id=session_id,
        timestamp_ms=clock.now(),
        final_text="Book a flight from Seoul to Tokyo",
    ))
    
    # Clean shutdown
    await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Architectural "Never Do" Rules

- 🚫 **NEVER call `request_cancellation()` from external client code:** Always submit an `INTERRUPTION` event to let the kernel manage state and tracing.
- 🚫 **NEVER pass an explicit `generation` integer when creating fillers:** Always leave `generation=None` so acknowledgement speech is not blocked by output gate epoch shifts.
- 🚫 **NEVER import internal Module 1 extractors into Module 2:** Module 2 should parse complete, finalized transcripts independently.
- 🚫 **NEVER place blocking network or LLM calls inside fast-path handlers:** Keep all handlers under 1ms execution time.
