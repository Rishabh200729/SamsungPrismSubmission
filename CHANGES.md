# CHANGES — Module/TRAX-core audit and fixes

Everything below was verified by actually running code against the real FDB-v3 evaluator
(github.com/DanielLin94144/Full-Duplex-Bench, `v3/`), not inferred from reading it. Where a
claim needed a real model or a real LiveKit session I could not run, that's stated
explicitly rather than assumed to work — see "Not verified" at the end.

## 1. `prism/tool_dispatcher.py` — four defects that were unpassable-by-construction

The official scorer (`evaluate_pass_rate.py`) does strict multiset matching on tool names
against `expected_tool_calls`. Ten of the 100 public scenarios expect the **same tool called
more than once** with different arguments (`travel_15/23/25`, `finance_13/14/22`,
`housing_13/25`, `ecommerce_22/24`) — confirmed by loading `benchmark_data_v2.json` and
counting repeats directly, not by guessing.

- **Same-tool calls always superseded each other**, with no way to tell "the user repaired
  their speech" from "the model legitimately called the tool twice" (e.g. `track_order` for
  two different orders). This made all ten repeat-tool scenarios structurally unpassable.
  **Fix:** `ToolCall.issue_epoch` + `TRPGate.correction_epoch` — a later same-tool call is
  only treated as stale if a real repair marker fired *between* the two issue times.
  Reproduced and fixed; see `prism/tests/test_dispatcher_regressions.py::TestLegitimateRepeatsSurvive`
  and `::TestRepairsStillSupersede`.

- **`_write_telemetry` erased every earlier same-name line** before appending the new one —
  this deleted legitimate repeats even when both had already executed. **Fix:** append-only
  log; retraction (when a real repair invalidates an earlier call) now removes only that
  specific line, via a `call_id`. See `TestEvaluatorLogContract`.

- **Compensation records were written into the tool-call log the evaluator parses.**
  Reproduced the organizers' exact extraction loop (`v3/run_tool_benchmark.py` "Step 6")
  against a log containing one: it raises `TypeError` and the pipeline silently keeps
  whatever it collected *before* that line — so a corrected call logged *after* a rollback
  was invisible to the scorer, and the scorer saw only the stale one. **Fix:** compensation
  and audit records go to a separate `<log>.audit` file; the tool-call log now contains only
  `{"room","call"}` records. See `TestEvaluatorLogContract::test_corrected_call_after_rollback_is_visible_to_official_parser`.

- **Unknown tools defaulted to `READ_ONLY`**, running them speculatively before the turn was
  confirmed — for a tool nobody had classified, that's the wrong default (a mutation could
  run while the user is still repairing the request). **Fix:** fail-closed default of
  `COMPENSABLE`; `register_effect()` added for extending the map safely.

- **`reset_for_new_turn()` cancelled pending speculative tasks but never resolved the
  `dispatch()` future** a tool call was awaiting — if the model's tool call was still
  blocked on `TRP_CONFIRMED` when the user resumed speaking, that coroutine hung forever.
  **Fix:** every pending future now resolves to a `"discarded"` sentinel on reset. Confirmed
  this actually hangs on the unfixed code (`asyncio.wait(..., timeout=1.0)` times out) before
  fixing it — see `TestTurnResetDoesNotHang`.

All five fixes were mutation-tested: each was individually re-broken and confirmed that the
new regression suite catches it (including one that reproduces as a literal test-run hang,
which counts as "caught").

## 2. `prism/trp_gate.py` — repair-marker lexicon was over-broad

Bare `"no"`, `"rather"`, `"like"` were matched anywhere in the utterance. These are ordinary
English words — `"there is no problem"`, `"I would rather have the window seat"`, `"I'd
like to book a flight"` — all incorrectly triggered `REPAIRING` (a 900ms hold) before this
fix, and would have incorrectly *discarded a legitimate call* once epoch-based supersession
(item 1) was added, since the same lexicon now also decides what counts as a "repair" for
that purpose. **Fix:** these terms only count at a clause boundary (after a pause/punctuation
or before another repair cue), verified in both directions —
`prism/tests/test_trp_gate_precision.py::test_ordinary_speech_is_not_a_repair` (6 phrases,
all previously false-positived) and `::test_genuine_repairs_still_fire` (11 phrases, all
still correctly caught).

Added `TRPGate.correction_epoch`: a monotonic counter of distinct repair markers heard,
never reset by `reset_for_new_turn()` (repairs must remain visible across a turn boundary),
and deduplicated against cumulative interim ASR transcripts (the same marker text shouldn't
count twice just because ASR resent a growing partial).

## 3. Prompt / tool-schema drift between the live agent and the text benchmark harness

`agent/trax_agent.py` and `scripts/benchmarks/test_gemini_tool_calling.py` each carried an
independent, hand-written copy of the system prompt and the 12 tool descriptions. They had
already drifted (different wording, `update_search_filter`'s `value` hardcoded to `"string"`
in the harness, which would misscore any scenario expecting a number or boolean there) — so
the harness was measuring a different system than the one that ships.

**Fix:** `agent/tool_specs.py` is now the single source; both the agent
(`AssistantFnc`'s decorators + dynamically-assigned docstrings) and the harness
(`FDB_TOOLS = openai_tools()`) build from it. `prism/tests/test_agent_specs_sync.py` imports
the real `agent/trax_agent.py` (against a small local stub of the `livekit` SDK — see that
test's docstring for exactly what the stub does and doesn't prove) and checks every tool's
live description matches the shared spec byte-for-byte; mutation-tested against three
realistic re-breaks (hardcoding a description back, swapping the prompt, dropping the wiring
below), all caught.

## 4. Benchmark-answer leakage in prompts (compliance risk, not a bug)

The rules state: don't hardcode or pattern-match on the public benchmark's answers — FDB-v3
is public, and submissions are checked for this. Scanning the *original* prompts against the
actual `benchmark_data_v2.json` (not guessing) found 30 verbatim hits: e.g.
`calculate_commute`'s description literally quoted `"the office"`, which is
`housing_17`'s expected commute destination; `add_to_cart`'s docstring quoted `'B7'`, which
is `ecommerce_11`'s expected `product_id`. These read as innocuous style examples but are
exact answer keys.

**Fix:** every illustrative example in `agent/tool_specs.py` is synthetic and was checked
against the real benchmark data, not assumed safe. `scripts/audit_benchmark_leakage.py`
automates this check (whitelisting the organizers' own reference agent's vocabulary so it
doesn't flag legitimate shared interface terms like `"driving"` or `"BOB12"`) — **run it
after every prompt edit, and before every submission.** Current result: 0 findings across
`agent/trax_agent.py`, `agent/tool_specs.py`, `scripts/benchmarks/test_gemini_tool_calling.py`.

## 5. Wiring gap that would have silently defeated fix #1

`agent/trax_agent.py`'s `entrypoint()` constructed `ToolDispatcher` without
`correction_epoch_fn`, and constructed it *before* `gate = TRPGate()` even existed in scope.
Without that wiring, the dispatcher falls back to its legacy "same tool name = a correction"
rule — meaning fix #1 above would exist in `prism/tool_dispatcher.py` but never take effect
in the actual running agent. **Fixed:** `gate` is now constructed first and
`correction_epoch_fn=lambda: gate.correction_epoch` is passed through.
`test_agent_specs_sync.py::TestCorrectionEpochWiring` checks this wiring is present by
inspecting the real `entrypoint` source, and it's mutation-tested (removing the wiring is
caught).

## Test count

`python -m unittest discover -s prism/tests -p "test_*.py"` — **50 passed** (18 pre-existing
+ 19 dispatcher regressions + 8 gate precision + 5 agent/spec sync), all mutation-tested
where the fix was novel logic (not just a data change).

## Not verified — be honest about these before the demo

- **The live agent was never run against a real LiveKit session or a real realtime model.**
  `test_agent_specs_sync.py` imports the real `agent/trax_agent.py` against a small local
  stub of the `livekit` SDK (enough to prove descriptions/prompt/wiring are correct); it does
  not prove the agent behaves correctly end-to-end with real audio. Run the actual agent with
  `livekit-agents` installed before the demo.
- I could not run `scripts/benchmarks/test_gemini_tool_calling.py` itself (needs a Gemini API
  key); I verified its tool/prompt definitions now match the agent's by construction (shared
  import), not by executing the script.
- `EFFECT_MAP`'s classification of the 12 named tools was not changed and was not
  re-audited tool-by-tool against the benchmark's actual side-effect semantics beyond what
  the existing `test_effect_map_exact_12_tools` already checked — only the *unknown-tool
  default* changed.
