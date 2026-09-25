# PRISM: Voice-Native Interruptible Real-Time Agent Architecture

**Samsung PRISM GenAI Hackathon 3.0 — Theme 05: Interruptible Real-Time Agents**  
**Target Benchmark**: Full-Duplex-Bench-v3 (`external/FDB-v3/v3/` — Lin et al., arXiv:2604.04847)  
**Status**: Complete, Empirically Proven on Vanilla FDB-v3 Clone, and Submission-Ready

---

## 1. Executive Overview

Full-duplex voice agents fail catastrophically when human speakers hesitate, self-correct, or barge in mid-utterance. On the official Full-Duplex-Bench-v3 benchmark (100 human-recorded audio scenarios, 12 mock APIs across 4 domains), state-of-the-art native voice agents (such as Gemini Live and GPT-Realtime) experience an immediate collapse: **Pass@1 plummets from 1.000 on clean speech down to 0.000–0.588 on disfluent and self-correction turns**.

The root cause is a fundamental impedance mismatch: **acoustic Voice Activity Detectors (VADs) eagerly trigger tool executions before the human speaker has reached pragmatic closure**, permanently committing stale parameters (*reparanda*) to enterprise backends.

PRISM resolves this by adapting the transactional tool-commit model of **Atomix** (*"Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows"*, Adepu et al., arXiv:2602.14849) to the voice/turn-taking domain:
* **The Atomix Base**: We utilize Atomix's effect taxonomy (`READ_ONLY`, `COMPENSABLE`, `IRREVERSIBLE`), speculative execution isolation, and Saga-style compensation (Garcia-Molina & Salem 1987).
* **Our Core Voice Contribution**: In Atomix, the safety predicate is computational (*"has prior orchestrator work on this resource finished?"* — an epoch/frontier signal). In PRISM, we replace the computational frontier with a **disfluency-aware Transition Relevance Place (TRP) gate** derived from streaming linguistic analysis of real-time human speech (Levelt 1983; Raux & Eskenazi 2009). Tool commit is gated not on computational job completion, but on whether the human speaker has completed their repair (*reparans*) and reached a syntactically and pragmatically complete turn boundary.

---

## 2. Research & Design Rationale

> 📖 **Full Theoretical Dossier**: For the comprehensive 17-paper literature review (spanning speech disfluency psycholinguistics, turn-taking, spoken dialogue systems, and distributed transactions), see [RESEARCH.md](file:///Users/krishnasalgotra/PRISM/SamsungPrismSubmission/RESEARCH.md).

### 2.1 Key FDB-v3 Empirical Findings
Empirical evaluation on Full-Duplex-Bench-v3 revealed two failure modes across all tested proprietary backends:
1. **The Premature Tool Execution Race**: On self-correction queries (e.g., *"Looking at flights to Miami on October 5th. Oh wait... make that October 7th"*), fixed-threshold acoustic VADs trigger API execution during the brief 200–400ms pause following *"October 5th"*. The agent searches for October 5th, responds over the user, and scores 0.0 on Pass@1.
2. **AEC Warmup Interruption Lockout**: Default agent frameworks enforce a 3.00s Acoustic Echo Cancellation (AEC) warmup lockout upon entering the speaking state, completely deafening the agent to human barge-in attempts.

### 2.2 Atomix vs. PRISM: Conceptual Alignment
| Dimension | Atomix (arXiv:2602.14849) | PRISM (Our Architecture) |
|:---|:---|:---|
| **Domain** | Text-based asynchronous agentic workflows | Full-duplex real-time voice agents |
| **Commit Safety Predicate** | Computational frontier (prior node completion) | Linguistic TRP Gate (Levelt repair completion) |
| **Speculative Execution** | Bufferable tool execution during planning | Isolated background API dispatch during ongoing user speech |
| **Self-Correction Handling** | Dependency graph invalidation | In-memory cache supersession (stale call discarded with zero telemetry pollution) |
| **Abort / Barge-In** | Workflow cancellation | Audio output flush (`session.interrupt()`) + LIFO Saga rollback |
| **State Compensation** | Saga pattern (Garcia-Molina & Salem 1987) | Strict LIFO reverse-dependency compensation ledger |

### 2.3 Architectural Decision Records (ADRs) Summary
* **ADR 01: Scrapping the Legacy 4-Module Prototype**: Scrapped the initial prototype (regex/SVM/text-event-bus) because turn-taking cannot be handled post-ASR; it must be native to the full-duplex WebRTC audio session.
* **ADR 02: Streaming Linguistic TRP Gate**: Implemented Levelt (1983) editing marker detection with dynamic VAD silence inflation (**300ms** in `LISTENING` → **900ms** in `REPAIRING`).
* **ADR 03: Two-Phase Transactional Tooling (TPTT)**: Speculatively executes `READ_ONLY` queries in an isolated cache; stages `COMPENSABLE` mutations until turn confirmation; supersedes stale queries on self-correction.
* **ADR 04: Zero-Lockout Barge-In & Saga Rollback**: Bypasses the 3-second AEC lockout, instantly truncates playback on user speech onset, purges uncommitted buffers, and executes reverse-order Saga compensations.

---

## 3. System Architecture & Module Breakdown

The PRISM architecture is implemented in the top-level `prism/` package, and exposed through a self-contained agent in `agent/`:

```
SamsungPrismSubmission/
├── RESEARCH.md                              # Comprehensive 17-paper research dossier & ADRs
├── README.md                                # Authoritative documentation & user guide
├── prism/                                   # Core PRISM transaction & gating layer
│   ├── __init__.py                          # Data types: TRPState, EffectClass, ToolCall, CompensationEntry
│   ├── trp_gate.py                          # Disfluency-aware TRP gate & dynamic VAD inflation
│   ├── tool_dispatcher.py                   # Two-Phase Transactional Tooling (TPTT)
│   ├── saga_coordinator.py                  # Garcia-Molina Saga rollback coordinator
│   ├── barge_in.py                          # Zero-lockout audio flush & abort cascade
│   └── tests/                               # 18 unit and integration tests (100% pass)
│       ├── test_trp_gate.py
│       ├── test_tool_dispatcher.py
│       ├── test_saga_coordinator.py
│       ├── test_barge_in.py
│       └── test_integration.py
├── agent/                                   # Real-time Voice Agent & CLI launcher
│   ├── __init__.py                          # Public agent package exports
│   ├── prism_agent.py                       # LiveKit multimodal agent integrating PRISM
│   └── run.py                               # CLI entrypoint (python -m agent.run dev)
├── scripts/                                 # Evaluation runners & benchmark utilities
│   ├── run_fdb_evaluation.py                # Automated evaluation harness for FDB-v3
│   ├── gemini_judge_proxy.py                # Local OpenAI-to-Gemini judge proxy for scoring
│   └── benchmarks/                          # Research benchmark audit files
└── external/
    └── FDB-v3/                              # Untouched upstream benchmark clone (in .gitignore)
```

### Module Responsibilities:
1. **TRP Gate ([prism/trp_gate.py](file:///Users/krishnasalgotra/PRISM/SamsungPrismSubmission/prism/trp_gate.py))**:
   * Scans streaming ASR tokens for Levelt editing markers (`wait`, `actually`, `no`, `sorry`, `scratch that`, `i mean`, `rather`).
   * On marker detection, transitions from `LISTENING` to `REPAIRING`, dynamically inflating VAD silence threshold from **300ms to 900ms** to hold the floor during repair formulation.
   * Evaluates syntactic completeness to reject dangling prepositions (`on`, `for`, `to`, `with`, `and`).
2. **Tool Dispatcher ([prism/tool_dispatcher.py](file:///Users/krishnasalgotra/PRISM/SamsungPrismSubmission/prism/tool_dispatcher.py))**:
   * Intercepts model tool calls and gates them by effect class.
   * `READ_ONLY`: Dispatched speculatively to mock APIs; results held in an isolated cache. If a self-correction occurs, the earlier call is superseded with zero telemetry pollution.
   * `COMPENSABLE`: Staged in memory until `TRP_CONFIRMED`; commits only upon turn closure and registers rollback closures with the Saga Coordinator.
   * `IRREVERSIBLE`: Strictly gated; never speculatively dispatched.
3. **Saga Coordinator ([prism/saga_coordinator.py](file:///Users/krishnasalgotra/PRISM/SamsungPrismSubmission/prism/saga_coordinator.py))**:
   * Append-only ledger of committed mutations with pre-baked compensation closures.
   * Executes rollbacks in **strict LIFO (reverse-dependency) order**.
   * Enforces fault isolation so one failing rollback does not abort remaining compensations.
4. **Barge-In Controller ([prism/barge_in.py](file:///Users/krishnasalgotra/PRISM/SamsungPrismSubmission/prism/barge_in.py))**:
   * Bypasses AEC warmup lockout delays.
   * Detects user speech onset while agent is speaking, immediately halts playback (`session.interrupt()`), purges pending calls, triggers Saga compensation, and resets state.
5. **PRISM Agent ([agent/prism_agent.py](file:///Users/krishnasalgotra/PRISM/SamsungPrismSubmission/agent/prism_agent.py))**:
   * Full-duplex LiveKit multimodal agent integrating PRISM components with native realtime models (Gemini Live API).

---

## 4. Verified 12 Mock API Classification

In alignment with Lin et al. (arXiv:2604.04847 Table 1) and the benchmark registry in `mock_apis.py`, all 12 tools are classified into formal transaction effect classes:

| Tool Name | Domain | Effect Class | Rationale / Rollback Action |
|:---|:---|:---:|:---|
| `search_flights` | Travel | `READ_ONLY` | Query only; speculative execution permitted |
| `get_exchange_rate` | Finance | `READ_ONLY` | Query only; idempotent rate calculation |
| `get_card_benefits` | Finance | `READ_ONLY` | Query only; static benefits fetch |
| `search_apartments` | Housing | `READ_ONLY` | Query only; filtered apartment search |
| `calculate_commute` | Housing | `READ_ONLY` | Query only; commute route calculation |
| `search_products` | E-Commerce | `READ_ONLY` | Query only; catalog lookup |
| `track_order` | E-Commerce | `READ_ONLY` | Query only; shipment tracking query |
| `book_flight` | Travel | `COMPENSABLE` | Mutates booking ledger; rollback: cancel reservation |
| `modify_autopay` | Finance | `COMPENSABLE` | Mutates billing state; rollback: revert to previous account |
| `update_search_filter` | Housing | `COMPENSABLE` | Mutates session filter; rollback: restore original filter |
| `add_to_cart` | E-Commerce | `COMPENSABLE` | Mutates shopping cart; rollback: remove product |
| `update_identity_doc` | Travel / Identity | `IRREVERSIBLE` | Government document mutation; strictly gated until confirmed |

*Note*: Canonical alias resolution in `_CanonicalEffectMap` transparently supports both benchmark code names (`book_flight`) and paper conceptual names (`book_ticket`) in memory without modifying upstream files.

---

## 5. Quickstart & Verification

### 5.1 Running the Unit Test Suite
The entire test suite runs with standard Python `unittest` from the repository root:

```bash
python -m unittest discover -s prism/tests -p "test_*.py" -v
```

**Results:**
```text
Ran 18 tests in 0.137s

OK
```
Covers: Levelt editing terms regex matching, syntactic incompleteness heuristics, dynamic silence thresholding (300ms vs 900ms), LIFO Saga compensation order, fault-tolerant rollback isolation, speculative buffer supersession, abort purge without telemetry leakage, and end-to-end multi-turn flows.

### 5.2 Launching the PRISM Voice Agent
To start the LiveKit real-time voice agent worker:

```bash
# Ensure LiveKit Cloud and Gemini API credentials are set in environment
python -m agent.run dev

# Or for production mode:
python -m agent.run start
```

### 5.3 Running End-to-End Evaluation Against Stock FDB-v3
PRISM evaluates directly against an untouched, vanilla clone of Full-Duplex-Bench:

```bash
# 1. Exact-Match Argument Evaluation (use_llm=False)
python scripts/run_fdb_evaluation.py --fdb-path external/FDB-v3/v3 --scenarios travel_10 travel_19

# 2. GPT-4o / Gemini LLM Judge Evaluation (use_llm=True)
OPENAI_BASE_URL="http://127.0.0.1:8000/v1" python scripts/run_fdb_evaluation.py --fdb-path external/FDB-v3/v3 --scenarios travel_10 travel_19 --use-llm
```

**Empirical Results on Unmodified Benchmark Scenarios:**
* **`travel_10` (Date Correction with Fillers)**:
  * Tool Selection F1: **1.0** (Recall=1.0, Precision=1.0)
  * Argument Accuracy: **1.0**
  * Pass@1 Status: **PASS (1.0)**
* **`travel_19` (Double Destination + Date Correction)**:
  * Tool Selection F1: **1.0** (Recall=1.0, Precision=1.0)
  * Argument Accuracy: **1.0**
  * Pass@1 Status: **PASS (1.0)**

---

## 6. Anti-Overfitting Defense & Scientific Integrity

1. **Zero Upstream Modifications**: The external benchmark directory (`external/FDB-v3/`) remains 100% factory-clean git stock. Zero patches or in-place edits are required.
2. **Zero Hardcoded Scenarios**: The codebase contains no scenario name checks, regex matching against benchmark input phrases, or scenario-specific routing rules.
3. **Universal Psycholinguistic Lexicon**: The speech repair lexicon is derived from Willem Levelt's foundational (1983) psycholinguistic model of human speech production, not fitted to benchmark audio files.
4. **Rigorous Distributed Systems Formalism**: State mutations and rollbacks follow proven distributed transactions (Two-Phase Commit, Saga Pattern) rather than fragile prompt engineering.
