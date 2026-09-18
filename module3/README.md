# Module 3 — Shared Runtime Architecture & Evaluation Infrastructure

**PRISM GenAI Hackathon 3.0 — Theme 05: Interruptible Real-Time Agents**

---

## Overview

Module 3 is the **execution kernel** for the PRISM agent system. It provides a shared, model-agnostic, event-driven runtime and a local deterministic evaluation harness.

Think of it as _the operating system for the other modules_.

---

## Architecture

```
                    INPUT EVENTS
                         |
                         v
                +------------------+
                |   INPUT QUEUE    |  ← async FIFO, timestamp-aware
                +------------------+
                         |
                         v
                +------------------+
                |    DISPATCHER    |  ← model-agnostic handler registration
                +------------------+
                   /      |       \
                  /       |        \
                 v        v         v
            FAST PATH  SLOW PATH  MULTIMODAL
                 \       |        /
                  \      |       /
                   v     v      v
                +------------------+
                | TASK / SESSION   |
                |    MANAGER       |  ← generation-based stale protection
                +------------------+
                         |
                         v
                +------------------+
                |  OUTPUT QUEUE    |  ← externally observable events
                +------------------+
                         |
                         v
                +------------------+
                | TRACE RECORDER   |  ← append-only causal trace
                +------------------+
                         |
                         v
                +------------------+
                | EVALUATION       |
                | HARNESS          |  ← metrics, assertions, reports
                +------------------+
```

---

## Key Design Decisions

| Decision | Problem Solved | Research Inspiration |
|----------|---------------|---------------------|
| Dual async queues | Non-blocking event ingestion | DuplexOmni interaction/thinking separation |
| Generation-based stale protection | Late results from cancelled work | DuplexOmni async collaboration |
| Explicit task state machine | Prevent impossible state transitions | Structured Graph Harness |
| VirtualClock | Deterministic scenario replay | Gaia2 simulated time |
| Session isolation | No cross-session leakage | Sema Code session isolation |
| Handler registration | Model-agnostic dispatch | Sema Code programmable core |
| Phase-aware interruption | Context-sensitive cancellation | Bohus & Horvitz event/action separation |

---

## Project Structure

```
module3/
├── runtime/
│   ├── runtime.py          ← Main Runtime class (the kernel)
│   ├── config.py           ← RuntimeConfig (TEST/PRODUCTION presets)
│   ├── interfaces.py       ← Integration contracts for Modules 1/2/4
│   │
│   ├── events/
│   │   ├── base.py         ← BaseEvent (universal envelope)
│   │   ├── input_events.py ← TEXT_CHUNK, INTERRUPTION, TOOL_RESULT, ...
│   │   ├── output_events.py← FILLER, TOOL_CALL, FINAL_RESPONSE, ...
│   │   └── internal_events.py ← TASK_CREATED, STALE_RESULT_DISCARDED, ...
│   │
│   ├── queues/
│   │   ├── input_queue.py  ← Async FIFO input queue
│   │   └── output_queue.py ← Async output queue with drain helpers
│   │
│   ├── dispatcher/
│   │   └── dispatcher.py   ← Fan-out handler router
│   │
│   ├── sessions/
│   │   └── manager.py      ← SessionManager + SessionContext
│   │
│   ├── tasks/
│   │   ├── lifecycle.py    ← TaskStatus state machine
│   │   ├── registry.py     ← TaskRegistry per session
│   │   └── cancellation.py ← CancellationCoordinator (generation-based)
│   │
│   ├── clock/
│   │   ├── base.py         ← Clock protocol
│   │   ├── real_clock.py   ← Production wall-clock
│   │   └── virtual_clock.py← Deterministic test clock
│   │
│   └── tracing/
│       └── recorder.py     ← TraceRecorder (append-only, JSONL export)
│
├── evaluation/
│   ├── scenario.py         ← Scenario data model (YAML/JSON)
│   ├── runner.py           ← ScenarioRunner
│   ├── assertions.py       ← Trace + output assertion engine
│   ├── metrics.py          ← All scoring metrics
│   └── report.py           ← Human + machine readable reports
│
├── mocks/
│   └── tools.py            ← Deterministic mock async tools
│
├── scenarios/
│   ├── basic.yaml
│   ├── interruption.yaml
│   ├── stale_result.yaml
│   └── concurrency.yaml
│
├── tests/
│   ├── test_events.py
│   ├── test_queue.py
│   ├── test_dispatcher.py
│   ├── test_tasks.py
│   ├── test_cancellation.py
│   ├── test_sessions.py
│   ├── test_virtual_clock.py
│   ├── test_scenarios.py   ← All 16 required scenarios
│   └── test_integration.py ← Phase 8 end-to-end trace
│
├── examples/
│   └── demo_runtime.py     ← Full interruption demo
│
├── README.md
├── Dockerfile
└── (pyproject.toml in project root)
```

---

## Quick Start

### Install

```bash
pip install -e ".[test]"
```

### Run Tests

```bash
python -m pytest module3/tests/ -v --asyncio-mode=auto
```

### Run the Demo

```bash
python -m module3.examples.demo_runtime
```

### Run a YAML Evaluation Scenario

```bash
python -m module3.evaluation.runner --scenario module3/scenarios/basic.yaml --output report.json
```

The runner is protocol-only: scenarios needing agent behavior must supply
handlers through the Python API, so a CLI report can correctly fail assertions
that depend on Module 1, 2, or 4 behavior.

### Docker

```bash
docker build -f module3/Dockerfile -t prism-module3 .
docker run prism-module3
```

---

## Integration Contract for Other Modules

Other modules import **only** from:
- `module3.runtime.interfaces` — integration protocol
- `module3.runtime.events` — event factories and types
- `module3.runtime.config` — runtime configuration
- `module3.runtime.runtime.Runtime` — the runtime class itself

**Never import from internal submodules directly.**

### Module 1 (Fast Path)

```python
from module3.runtime.runtime import Runtime
from module3.runtime.events import make_filler, make_clarification

# Register fast-path handler
runtime.register_handler(InputEventType.TEXT_CHUNK, my_fast_handler)

# On interruption, request cancellation and then invalidate generation.
# The observable trace records TASK_CANCEL_REQUESTED first.
old_gen, new_gen = await runtime.request_cancellation(session_id)

# Emit filler acknowledgment
await runtime.emit_output(make_filler(session_id, clock.now(), "One moment..."))
```

### Module 2 (Slow Path)

```python
# Create a background reasoning task
task_id = await runtime.create_background_task(
    session_id, my_reasoning_coro(), task_type="slow_reasoning"
)

# Before publishing result, check for staleness
if runtime.is_stale_result(session_id, task_id):
    await runtime.handle_stale_result(session_id, task_id, call_id)
    return  # Do NOT publish

# Emit final response
await runtime.emit_output(make_final_response(session_id, ts, "Your answer..."))
```

### Module 4 (Multimodal)

```python
# Register multimodal handler
runtime.register_handler(InputEventType.VIDEO_FRAME, my_vision_handler)
runtime.register_handler(InputEventType.AUDIO_WAV, my_audio_handler)

# Submit multimodal results as background tasks
task_id = await runtime.create_background_task(
    session_id, my_vision_analysis(), task_type="multimodal"
)
```

---

## Task Lifecycle

```
PENDING → RUNNING → COMPLETED
              ↓         ↑ (race)
          WAITING → RUNNING
              ↓
    CANCELLATION_REQUESTED → CANCELLED
              ↓
           STALE (late result discarded)
              ↓
           FAILED
```

---

## Stale Result Protection

The generation counter is the core safety mechanism:

```
T=0    Task A created, generation=1
T=450  INTERRUPTION → cancellation requested, then generation increments to 2
T=700  Task A's tool result arrives
       runtime.is_stale_result(sid, task_A_id) → True (gen 1 < current gen 2)
T=701  STALE_RESULT_DISCARDED recorded in trace
T=720  Task B created, generation=2
T=900  Task B's FINAL_RESPONSE emitted
```

---

## Scoring Coverage

| Category | Mechanism |
|----------|-----------|
| Task Completion | TaskRegistry + TASK_COMPLETED trace |
| Interruption Recovery | CancellationCoordinator + GENERATION_INVALIDATED |
| Response Latency | TraceRecorder timestamps → metrics.py |
| Safety & Protocol | Pydantic validation + protocol_validity metrics |

---

## Dependencies

- Python ≥ 3.11
- `pydantic >= 2.5.0`
- `pyyaml >= 6.0.1`
- `pytest`, `pytest-asyncio` (test only)
