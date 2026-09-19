# Module 3 — Developer API & Integration Contract

## 1. Public API Surface & Allowed Imports

External modules (**Module 1**, **Module 2**, **Module 4**, and integration harnesses) must interact with Module 3 **strictly via its public exports**. 

### Allowed Imports

```python
# 1. Interfaces & Runtime Client Protocol
from module3.runtime.interfaces import IRuntimeClient
from module3.runtime import Runtime, TEST_CONFIG, PRODUCTION_CONFIG

# 2. Base Events & Categorization Enums
from module3.runtime.events import (
    BaseEvent,
    EventCategory,
    InputEventType,
    OutputEventType,
    InternalEventType,
)

# 3. Input Event Factories
from module3.runtime.events import (
    make_text_chunk,
    make_end_of_turn,
    make_audio_wav,
    make_video_frame,
    make_interruption,
    make_tool_result,
    make_tool_manifest,
)

# 4. Output Event Factories
from module3.runtime.events import (
    make_filler,
    make_tool_call,
    make_cancellation,
    make_clarification,
    make_final_response,
    StateSnapshot,
)

# 5. Task & Lifecycle Types
from module3.runtime.tasks import TaskRecord, TaskStatus
```

> ⚠️ **Strict Boundary Rule:** Do NOT import from `module3.runtime.queues.*`, `module3.runtime.dispatcher.*`, `module3.runtime.sessions.*`, `module3.runtime.tasks.*`, or `module3.runtime.tracing.*` directly. All functionality is exposed via `IRuntimeClient`.

---

## 2. Public Methods (`IRuntimeClient`)

All interactions with Module 3 flow through the `IRuntimeClient` protocol:

### 2.1 Session Lifecycle

#### `create_session(session_id: str | None = None, metadata: dict[str, Any] | None = None) -> str`
Allocates a new isolated session partition.
- **Parameters:**
  - `session_id`: Optional caller-supplied unique session ID. If omitted, a UUID4 string is generated.
  - `metadata`: Optional arbitrary dictionary of contextual session data.
- **Returns:** The authoritative `session_id` (`str`).
- **Raises:** `ValueError` if a session with that ID already exists.

#### `close_session(session_id: str) -> None` *(async)*
Gracefully terminates the session and cooperatively cancels all in-flight background tasks.
- **Parameters:** `session_id`: ID of the session to terminate.
- **Side effects:** Records `SESSION_CLOSED` in trace; closes session queues.

#### `get_session_metadata(session_id: str) -> dict[str, Any]`
Retrieves a copy of the session metadata dictionary.
- **Returns:** Metadata dictionary (empty dict if session does not exist).

#### `get_current_generation(session_id: str) -> int`
Returns the active generation epoch counter for the session ($1, 2, 3\dots$).
- **Returns:** Current generation (`int`, or `0` if unknown).

---

### 2.2 Event Transport

#### `submit_event(event: BaseEvent) -> None` *(async)*
Ingests an external input event into the runtime's priority input queue. Returns immediately upon enqueuing (non-blocking).
- **Parameters:** `event`: An instance of `BaseEvent` (created via input event factories).
- **Behavior:**
  - If `event.session_id` is unknown or closed, the event is safely dropped with a warning.
  - Automatically routes `INTERRUPTION` to the unbounded control lane and other events to the data lane.
  - Records the event in the trace causal chain before dispatch.

#### `emit_output(event: BaseEvent, *, priority: int = 0) -> None` *(async)*
Submits an output event through the Central Authorization Gate to the egress queue.
- **Parameters:**
  - `event`: An instance of `BaseEvent` (created via output event factories).
  - `priority`: Optional priority weight (higher number = earlier egress).
- **Behavior:**
  - Passes event through `_authorize_output()`.
  - Rejects outputs from stale generations, unvalidated tool calls, or duplicate write idempotency keys.
  - Silently drops soft-rejected duplicates while auditing them to trace.

#### `next_output(event_type: Any | None = None, session_id: str | None = None, timeout_ms: float | None = None) -> BaseEvent` *(async)*
Awaits and pops the next authorized output event from the egress queue.
- **Parameters:**
  - `event_type`: Optional filter by `OutputEventType`.
  - `session_id`: Optional filter by session.
  - `timeout_ms`: Optional timeout in milliseconds.
- **Returns:** The next matching `BaseEvent`.
- **Raises:** `asyncio.TimeoutError` if timeout expires before an event arrives.

#### `drain_outputs() -> list[BaseEvent]` *(async)*
Non-blockingly drains and returns all currently pending output events from the egress queue.
- **Returns:** List of queued `BaseEvent` objects.

---

### 2.3 Event Handler Registration

#### `register_handler(event_type: AnyEventType, handler: Callable[[BaseEvent], Awaitable[None]]) -> None`
Registers an asynchronous coroutine handler for an incoming event type.
- **Parameters:**
  - `event_type`: Enum member (e.g. `InputEventType.TEXT_CHUNK`, `InputEventType.END_OF_TURN`).
  - `handler`: Coroutine function taking a single `event: BaseEvent` parameter.
- **Behavior:** Handlers run concurrently on the main event loop. Handler exceptions are isolated and logged without crashing the event loop.

---

### 2.4 Task & Cancellation Management

#### `create_background_task(session_id: str, coro: Coroutine, task_type: str, call_id: str | None = None, parent_task_id: str | None = None, metadata: dict | None = None) -> str` *(async)*
Schedules a coroutine as a tracked background task within Module 3's 7-state task machine.
- **Parameters:**
  - `session_id`: Target session.
  - `coro`: Unawaited coroutine to execute.
  - `task_type`: Descriptive label (e.g. `"hotel_reasoning"`, `"vision_ocr"`).
  - `call_id`: Optional associated tool call ID for correlation.
  - `parent_task_id`: Optional parent task ID for sub-task tracing.
  - `metadata`: Optional contextual metadata.
- **Returns:** Unique `task_id` (`str`).
- **Raises:** `KeyError` if session does not exist.

#### `request_cancellation(session_id: str, reason: str = "interruption") -> tuple[int, int]` *(async)*
Primary barge-in handler. Advances generation epoch and cooperatively cancels active tasks.
- **Parameters:**
  - `session_id`: Target session.
  - `reason`: Cancellation rationale (default: `"interruption"`).
- **Returns:** `(old_generation, new_generation)` tuple.
- **Causal Guarantee:** Records `TASK_CANCEL_REQUESTED` on active tasks, increments generation, records `GENERATION_INVALIDATED`, transitions floor state to `INTERRUPTED`, and invalidates pending calls.

#### `cancel_task(session_id: str, task_id: str, reason: str = "interrupted") -> bool` *(async)*
Cooperatively cancels a specific task.
- **Returns:** `True` if cancellation was requested; `False` if task was already terminal.

#### `get_task(session_id: str, task_id: str) -> TaskRecord | None`
Inspects the current state of a task.

#### `get_task_by_call_id(session_id: str, call_id: str) -> TaskRecord | None`
Finds the originating task record associated with a tool call ID.

---

### 2.5 Staleness & Idempotency Auditing

#### `is_stale_result(session_id: str, task_id: str) -> bool`
Checks if a task's generation is older than the current session generation.
- **Returns:** `True` if stale (or unknown); `False` if valid to apply.

#### `is_stale_call(session_id: str, call_id: str) -> bool`
Checks if a tool call belongs to a superseded generation.

#### `handle_stale_result(session_id: str, task_id: str, call_id: str | None = None) -> None` *(async)*
Explicitly records a late tool result as `STALE_RESULT_DISCARDED` in the trace. Safe and idempotent for repeat invocations.

---

### 2.6 Clock & Time

#### `runtime.clock.now() -> float`
Returns current timestamp in milliseconds (supports both virtual discrete time and real wall-clock).

#### `runtime.clock.sleep(duration_ms: float) -> None` *(async)*
Asynchronously sleeps for specified milliseconds, respecting virtual or wall-clock configuration.

---

## 3. Event Catalog & Payload Schemas

All events inherit from `BaseEvent`:
```python
class BaseEvent(BaseModel):
    event_id: str              # Unique UUID4 string
    session_id: str            # Target session ID
    timestamp_ms: float        # Millisecond timestamp (>= 0.0)
    event_type: AnyEventType   # Enum identifier
    category: EventCategory    # INPUT, OUTPUT, or INTERNAL
    payload: dict[str, Any]    # Strictly validated payload
    task_id: str | None        # Optional task correlation ID
    call_id: str | None        # Optional tool call correlation ID
    generation: int | None     # Active epoch generation
    correlation_id: str | None # Optional request correlation ID
    metadata: dict[str, Any]   # Context dictionary
```

---

### 3.1 Input Events (`EventCategory.INPUT`)

Import factories from `module3.runtime.events`:

| Event Type | Factory Function | Payload Schema | Purpose |
|---|---|---|---|
| `TEXT_CHUNK` | `make_text_chunk(...)` | `text: str`<br>`is_partial: bool = False` | Streaming user speech transcribed by ASR. |
| `END_OF_TURN` | `make_end_of_turn(...)` | `utterance_id: str`<br>`final_text: str \| None` | Marks end of user speaking turn; triggers reasoning. |
| `INTERRUPTION` | `make_interruption(...)` | `reason: str = "user_interruption"`<br>`text: str \| None`<br>`supersedes_generation: int \| None`<br>`competitive: bool = True` | Priority VAD/conversation signal. `competitive=True` cancels superseded work; `False` is a backchannel/acknowledgment and does not cancel or advance generation. |
| `AUDIO_WAV` | `make_audio_wav(...)` | `audio_b64: str`<br>`duration_ms: float`<br>`sample_rate: int = 16000` | Raw audio waveform chunk for acoustic analysis. |
| `VIDEO_FRAME` | `make_video_frame(...)` | `frame_b64: str`<br>`width: int`<br>`height: int`<br>`frame_index: int = 0` | Camera frame chunk for computer vision. |
| `TOOL_RESULT` | `make_tool_result(...)` | `call_id: str`<br>`task_id: str`<br>`result: Any = None`<br>`success: bool = True`<br>`error: str \| None`<br>`latency_ms: float \| None` | Asynchronous response from external tool execution. |
| `TOOL_MANIFEST` | `make_tool_manifest(...)` | `tools: list[dict[str, Any]]`<br>`scenario_id: str \| None` | Registers scenario tool definitions & schemas. |

---

### 3.2 Output Events (`EventCategory.OUTPUT`)

Import factories from `module3.runtime.events`:

| Event Type | Factory Function | Payload Schema | Purpose |
|---|---|---|---|
| `FILLER` | `make_filler(...)` | `text: str`<br>`filler_type: str = "acknowledgment"` | Sub-100ms rapid acknowledgment tokens. |
| `TOOL_CALL` | `make_tool_call(...)` | `call_id: str`<br>`tool_name: str`<br>`arguments: dict[str, Any]`<br>`timeout_ms: float \| None`<br>`idempotency_key: str \| None`<br>`side_effect: str \| None` | Tool execution request emitted by planner. |
| `CANCELLATION` | `make_cancellation(...)` | `cancelled_task_id: str`<br>`cancelled_call_id: str \| None`<br>`reason: str = "interrupted"`<br>`generation_at_cancel: int \| None` | Explicit notice to abort external tool operation. |
| `CLARIFICATION` | `make_clarification(...)` | `text: str`<br>`missing_slots: list[str]`<br>`context: dict[str, Any]` | Prompt emitted when input or visual scene is ambiguous. |
| `FINAL_RESPONSE` | `make_final_response(...)` | `text: str`<br>`state_snapshot: StateSnapshot`<br>`is_complete: bool = True` | Authoritative conversational answer carrying structured state. |

---

## 4. End-to-End Integration Examples

### Example 1: Module 1 Adapter (Fast Path & Barge-in)

```python
from module3.runtime import IRuntimeClient
from module3.runtime.events import BaseEvent, InputEventType, make_filler

class Module1FastPathAdapter:
    def __init__(self, runtime: IRuntimeClient):
        self.runtime = runtime

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.TEXT_CHUNK, self.on_text_chunk)
        self.runtime.register_handler(InputEventType.INTERRUPTION, self.on_interruption)

    async def on_text_chunk(self, event: BaseEvent) -> None:
        # Emit sub-100ms spoken acknowledgment
        filler = make_filler(
            session_id=event.session_id,
            timestamp_ms=self.runtime.clock.now(),
            text="Got it, checking flights now...",
        )
        await self.runtime.emit_output(filler, priority=10)

    async def on_interruption(self, event: BaseEvent) -> None:
        # Invalidate generation and cancel active tasks immediately
        old_gen, new_gen = await self.runtime.request_cancellation(
            event.session_id, reason=event.payload.get("reason", "barge_in")
        )
        print(f"Interruption applied: Epoch {old_gen} -> {new_gen}")
```

---

### Example 2: Module 2 Adapter (Reasoning, Tools & Stale Handling)

```python
from module3.runtime import IRuntimeClient
from module3.runtime.events import (
    BaseEvent,
    InputEventType,
    make_tool_call,
    make_final_response,
    StateSnapshot,
)

class Module2ReasoningAdapter:
    def __init__(self, runtime: IRuntimeClient):
        self.runtime = runtime

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.END_OF_TURN, self.on_end_of_turn)
        self.runtime.register_handler(InputEventType.TOOL_RESULT, self.on_tool_result)

    async def on_end_of_turn(self, event: BaseEvent) -> None:
        sid = event.session_id

        # 1. Propose tool call with required idempotency key for writes
        evt, cid = make_tool_call(
            session_id=sid,
            timestamp_ms=self.runtime.clock.now(),
            tool_name="book_flight",
            arguments={"dest": "Seoul", "seats": 1},
            idempotency_key=f"{sid}-book-seoul-v1",
            side_effect="STATE_MODIFYING",
        )

        # 2. Register background reasoning coroutine
        async def execute_tool():
            await self.runtime.clock.sleep(50)  # Simulated async I/O

        tid = await self.runtime.create_background_task(
            session_id=sid,
            coro=execute_tool(),
            task_type="flight_booking",
            call_id=cid,
        )

        # 3. Emit tool call through output gate
        await self.runtime.emit_output(evt)

    async def on_tool_result(self, event: BaseEvent) -> None:
        sid = event.session_id
        tid = event.task_id
        cid = event.call_id

        # 4. Check if generation was superseded while tool was running
        if self.runtime.is_stale_result(sid, tid):
            await self.runtime.handle_stale_result(sid, tid, call_id=cid)
            return  # Discard stale result without mutating state

        # 5. Emit final response carrying committed snapshot
        final = make_final_response(
            session_id=sid,
            timestamp_ms=self.runtime.clock.now(),
            text="Flight to Seoul successfully booked!",
            state_snapshot=StateSnapshot(
                intent="flight_booking",
                slots={"dest": "Seoul", "seats": 1},
                completed_actions=["book_flight"],
            ),
        )
        await self.runtime.emit_output(final)
```

---

### Example 3: Module 4 Adapter (Multimodal & Visual Ambiguity)

```python
from module3.runtime import IRuntimeClient
from module3.runtime.events import BaseEvent, InputEventType, make_clarification

class Module4MultimodalAdapter:
    def __init__(self, runtime: IRuntimeClient):
        self.runtime = runtime

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.VIDEO_FRAME, self.on_video_frame)

    async def on_video_frame(self, event: BaseEvent) -> None:
        frame_idx = event.payload.get("frame_index", 0)

        # Detect visual gesture ambiguity
        if frame_idx == 42:
            clarification = make_clarification(
                session_id=event.session_id,
                timestamp_ms=self.runtime.clock.now(),
                text="Are you pointing to the departure date or return date?",
                missing_slots=["selected_field"],
            )
            await self.runtime.emit_output(clarification, priority=5)
```

---

### Example 4: Complete Integration Script

```python
import asyncio
from module3.runtime import Runtime, TEST_CONFIG
from module3.runtime.events import make_text_chunk, make_interruption

async def main():
    # 1. Initialize runtime kernel in deterministic test mode
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()

    # 2. Wire up adapters
    m1 = Module1FastPathAdapter(rt)
    m2 = Module2ReasoningAdapter(rt)
    m4 = Module4MultimodalAdapter(rt)
    m1.install()
    m2.install()
    m4.install()

    # 3. Create session
    sid = rt.create_session()

    # 4. Ingest user speech
    await rt.submit_event(make_text_chunk(sid, rt.clock.now(), "Book flight to Tokyo"))

    # 5. User interrupts mid-stream
    await rt.submit_event(make_interruption(sid, rt.clock.now(), reason="user_correction"))

    # 6. Drain outputs and verify gate decisions
    outputs = await rt.drain_outputs()
    for out in outputs:
        print(f"Output Egress: [{out.event_type.value}] -> {out.payload}")

    # 7. Clean teardown
    await rt.close_session(sid)
    await rt.stop()

if __name__ == "__main__":
    asyncio.run(main())
```
