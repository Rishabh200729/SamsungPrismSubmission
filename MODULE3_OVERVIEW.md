# Module 3 — Orchestration Kernel: High-Level Overview

## 1. Executive Summary

**Module 3 (Shared Runtime & Evaluation Infrastructure)** is the central asynchronous execution kernel of the multi-agent system (PRISM GenAI Hackathon 3.0 — Theme 05: *Interruptible Real-Time Agents*).

In conventional agent frameworks, external components (audio listeners, LLMs, external tools) directly emit outputs and alter state. When real-time interruptions occur, this caller-discipline model breaks down: out-of-order tool results corrupt conversation state, cancelled tasks produce duplicate bookings, and high-frequency video feeds block urgent voice interruptions.

Module 3 replaces caller discipline with **authoritative kernel enforcement**. It acts as the operating system for the agent: it manages sessions, routes events across prioritized lanes, tracks task generations, fences stale results, deduplicates state-modifying actions, and records an append-only causal audit trace.

---

## 2. What Module 3 Does

Module 3 owns all operational, concurrency, and safety guarantees across the agent lifecycle:

1. **Two-Lane Priority Event Transport:**
   - **Control Lane:** Transports urgent lifecycle and interruption events (`INTERRUPTION`). Never dropped, unbounded capacity.
   - **Data Lane:** Transports high-volume sensory and conversational streams (`TEXT_CHUNK`, `VIDEO_FRAME`, `AUDIO_WAV`, `TOOL_RESULT`). Bounded with drop-oldest overflow policy under load.
   - **Priority Drain:** Consumer loops always drain the control lane completely before yielding data events, guaranteeing sub-millisecond barge-in dispatch.

2. **Multi-Tenant Session Isolation:**
   - Provides completely isolated session state stores.
   - Zero state, event, task, or memory leakage between concurrent user sessions.

3. **Authoritative Floor State Machine:**
   - Owns the official 6-state conversational floor:
     $$\text{LISTENING} \longleftrightarrow \text{THINKING} \longleftrightarrow \text{SPEAKING} \longleftrightarrow \text{INTERRUPTED} \longleftrightarrow \text{WAITING\_FOR\_USER / TOOL}$$
   - Enforces legal transitions and forces `SPEAKING -> INTERRUPTED` immediately upon user barge-in.
   - Enforces anti-spam filler throttling to eliminate repetitive verbal stuttering.

4. **Task Lifecycle & Generation Fencing:**
   - Runs tracked coroutines through a formal 7-state task machine (`PENDING`, `RUNNING`, `CANCELLATION_REQUESTED`, `CANCELLED`, `COMPLETED`, `FAILED`, `STALE`).
   - Tracks monotonic generation epochs ($1, 2, 3\dots$). Every user interruption invalidates the active generation.
   - Dual cancellation: issues physical `asyncio.Task.cancel()` and applies logical generation fencing so any delayed tool results from older generations are intercepted and neutralized.

5. **Tool Manifest & Mutating Action Ledger:**
   - Validates tool invocations against scenario-declared JSON schemas before execution.
   - Requires idempotency keys for `STATE_MODIFYING` tools (or derives a deterministic key from session + epoch + tool + arguments).
   - Enforces exactly-once write execution: duplicate state-modifying proposals are blocked before network dispatch.

6. **Mandatory Central Output Gate:**
   - A single gatekeeper (`_authorize_output`) through which all egress events must pass.
   - Silently drops stale outputs, prevents cross-session leakage, blocks unvalidated tool calls, and intercepts duplicate writes.

7. **Deterministic Simulation & Causal Tracing:**
   - Integrates a pluggable discrete-event virtual clock (`VirtualClock`) enabling instant, repeatable test execution without wall-clock sleeps.
   - Records an append-only, causal trace graph of every event, task state change, and gate decision.

---

## 3. What Module 3 Does NOT Do

Module 3 strictly decouples infrastructure from domain logic. It deliberately does **NOT**:

- ❌ **Perform Voice Activity Detection (VAD) or Audio DSP:** Module 3 does not analyze raw microphone energy, pitch, or acoustic features. (Owned by **Module 1**).
- ❌ **Run Speech-to-Text (ASR) or Text-to-Speech (TTS):** Module 3 does not transcribe waveforms or synthesize audio bytes. It transports structured text and token events. (Owned by **Module 1** / downstream services).
- ❌ **Make Reasoning or Tool Selection Policies:** Module 3 does not decide *which* tool to call, does not formulate search queries, and does not run LLM prompts. (Owned by **Module 2**).
- ❌ **Process Raw Computer Vision Pixels:** Module 3 does not run CNNs, YOLO detectors, or Vision-Language Models on video frames. It routes frame events and buffers them safely. (Owned by **Module 4**).
- ❌ **Render User Interfaces (UI):** Module 3 emits structured protocol events (`FINAL_RESPONSE`, `CLARIFICATION`, `FILLER`); it does not render web pages, mobile screens, or graphical widgets.

---

## 4. How Modules Communicate

The modules communicate through a **hub-and-spoke zero-trust architecture** mediated entirely by Module 3:

```text
       ┌────────────────────────┐
       │        MODULE 1        │
       │  (Fast Path / Floor)   │
       └───────────┬────────────┘
                   │ Ingests: TEXT_CHUNK, INTERRUPTION
                   │ Emits:   FILLER, request_cancellation()
                   ▼
       ┌─────────────────────────────────────────────────────────┐
       │                        MODULE 3                         │
       │                  (Orchestration Kernel)                 │
       │                                                         │
       │  PriorityInputQueue ──► Dispatcher ──► TaskRegistry     │
       │         ▲                     ▲               │         │
       │         │                     │               ▼         │
       │  StateStore (CAS) ────► Output Gate ◄── Scheduler       │
       └─────────────────────────────────────────────────────────┘
                   ▲                               ▲
                   │ Ingests: END_OF_TURN,         │ Ingests: VIDEO_FRAME,
                   │          TOOL_RESULT          │          AUDIO_WAV
                   │ Emits:   TOOL_CALL,           │ Emits:   CLARIFICATION
                   │          FINAL_RESPONSE       │
       ┌───────────┴────────────┐      ┌───────────┴─────────────┐
       │        MODULE 2        │      │        MODULE 4         │
       │  (Reasoning / Tools)   │      │ (Multimodal Perception) │
       └────────────────────────┘      └─────────────────────────┘
```

### Communication Principles:
1. **Typed Asynchronous Events:** All communications are strongly typed Pydantic models inheriting from `BaseEvent`. Every event carries `session_id`, `timestamp_ms`, `event_type`, `category`, and payload.
2. **Untrusted Proposals:** Modules 1, 2, and 4 do not directly modify session state or execute unmonitored external side-effects. They submit event proposals. Module 3 authorizes them against the session epoch and active schema.
3. **Decoupled Asynchronous Dispatch:** Modules register coroutine handlers (`register_handler`) for specific event types. The dispatcher invokes handlers concurrently with per-handler exception isolation.
4. **Cooperative & Logical Neutralization:** If Module 1 triggers an interruption, Module 3 cancels Module 2's and Module 4's in-flight tasks and bumps the generation. Even if a background coroutine finishes before cancellation completes, Module 3 rejects the late result at the gate.
