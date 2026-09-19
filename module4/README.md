# Module 4 — Multimodal Grounding & Protocol Compliance

Implements the two objectives Module 4 owns per the team's build plan: deciding
"ambiguous vs. answerable" across text + video frames (§3.2 item 5), and providing a
shared schema-enforcement utility for turning raw LLM output into valid event arguments
(§3.2 item 6 / Safety & Protocol's schema-adherence line).

## Files

| File | What it does |
|---|---|
| `clarifier.py` | Three-stage ambiguity pipeline: `TextClarifier` (manifest-driven missing slots), `VisionClarifier` (frame confidence/quality), `CrossModalClarifier` (deictic reference grounding with multi-candidate margin scoring). |
| `visual_buffer.py` | Session-scoped sliding-window frame buffer (last N frames), keyed to `VideoFramePayload`'s actual fields (`frame_b64`, `frame_index`). |
| `constrained_output.py` | Standalone JSON-schema enforcement utility (validate-and-repair for hosted-API LLMs, or a true XGrammar wrapper if you have local logits access). Not coupled to Module 3. |
| `adapter.py` | `Module4Adapter` — wires the above to the real `IRuntimeClient`. **Read the module docstring before wiring `install()`** — it documents a real race condition, not a hypothetical one, that's now reproduced in `tests/test_adapter_integration.py::test_defensive_handler_alone_can_race_with_module2`. |
| `tests/test_clarifier.py` | Pure unit tests, no runtime dependency. 8/8 passing. |
| `tests/test_adapter_integration.py` | Integration tests against the **real** `module3.runtime.Runtime`, not a fake. 3/3 passing. |

Run everything: `python -m pytest module4/ -v` (or just `pytest` from repo root — `module4/tests` is now in `pyproject.toml`'s `testpaths`).

## The one integration decision that actually matters

`module3/runtime/dispatcher/dispatcher.py` fans an event out to every registered handler
via `asyncio.gather` — concurrently, no ordering guarantee. If Module 4 registers its own
`END_OF_TURN` handler (to emit `CLARIFICATION`) while Module 2 independently registers its
own `END_OF_TURN` handler (to emit `TOOL_CALL`), **both fire for the same ambiguous turn**.
`test_defensive_handler_alone_can_race_with_module2` reproduces this concretely — it's not
a hypothetical, I ran it.

**Fix**: whoever owns Module 2's `END_OF_TURN` handler should call
`Module4Adapter.evaluate_ambiguity(session_id, text)` synchronously, in-line, before
deciding to emit a tool call — see `test_direct_call_path_is_race_free` for the exact
pattern. This needs a five-minute conversation with Module 2's owner before day 8's
integration pass, not a discovery during it.

## Status as of the `main` branch snapshot (resolved gaps)

Both issues below were flagged against an earlier, incomplete snapshot and are now
**confirmed fixed** — verified by actually running the full suite against your real
`main` branch, not by re-reading the code:

1. ~~`priority_input_queue.py` missing~~ — your teammate's real implementation is now in
   `module3/runtime/queues/priority_input_queue.py`. My earlier stopgap version has been
   deleted; it's no longer needed.
2. ~~`Runtime` not wired to `PriorityInputQueue`~~ — `runtime.py` now constructs
   `PriorityInputQueue` directly, and `module3/tests/test_runtime_priority_wiring.py`
   (your team's own test) passes: an `INTERRUPTION` submitted after a backlog of data
   events is dispatched first, every time, across a 20-run stress test.

Full suite result at the time of this integration: **276 passed, 0 failed**
(`python -m pytest` from repo root). Module 4 required zero code changes to work against
this snapshot — the earlier work against the documented interface contract paid off.

## Local LLM, not a hosted API — what this changes

Confirmed: the team is running a self-hosted model with direct logits access. This means
`constrained_output.py`'s `XGrammarBackend` is now the **primary, recommended** path for
Module 2's final-response/tool-argument generation — not a maybe-someday backend. See
`constrained_output.py`'s module docstring for full usage (`XGrammarBackend.generate(...)`
wraps compile → constrained `model.generate()` → decode → parse in one call).

**I could not test `XGrammarBackend` end-to-end** — couldn't install `xgrammar` or load a
real model in the sandbox I built this in (disk space). It's written directly from
XGrammar's current documented API and `ValidateRepairBackend` (which needed no model, just
`jsonschema`) is fully unit-tested and passing (5/5 in `test_constrained_output.py`), but
**someone on the team needs to run `XGrammarBackend` against your actual model before the
demo** — specifically check the two gotchas called out in that file's docstring
(tokenizer vocab_size mismatch, and `compile_json_schema`'s exact input format on your
xgrammar version). Don't take my word for that one path; verify it.

## Known gaps / honest limitations

- **No real vision model.** `VisionClarifier` and `CrossModalClarifier` consume whatever
  `caption_fn_async` returns; the default wraps Module 3's own `mock_frame_lookup` mock
  (deterministic 0.95 confidence, canned object lists). This is fine for integration
  testing and probably fine for the public 9-scenario suite. For the hidden 60-scenario
  set — which is exactly where the 1.5x multimodal multiplier lives — you likely want a
  real captioner/VLM behind this interface if there's time after core integration is
  stable. The interface (`(frame_b64, frame_index) -> (caption, objects, confidence)`)
  doesn't change either way.
- **XGrammar vs. hosted API.** `constrained_output.py`'s `XGrammarBackend` needs local
  logits access (self-hosted HF/vLLM model). If Module 2's reasoning calls a hosted LLM
  API instead, use `ValidateRepairBackend` (the default) — confirm which one applies
  *today*, not at integration time.
- **`CrossModalClarifier`'s candidate scoring is a placeholder** (recency-weighted presence,
  not real cross-modal similarity). The seam to upgrade is `_score_candidates()` in
  `clarifier.py` — swap in a CLIP-style similarity score between the referring expression
  and each detected object if you have time.
- **Session cleanup isn't wired to session close yet.** `Module4Adapter.drop_session()`
  exists and does the right thing (evicts the frame buffer + manifest cache, per the
  session-scoped-memory constraint in the guide's §6) but nothing calls it — there's no
  session-lifecycle hook exposed from `Runtime` for other modules to attach to yet. Raise
  this with whoever owns `runtime.py`/`sessions/manager.py`.

## Research references

- Yang, S. et al. "Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for
  Egocentric Intent Disambiguation." AAAI 2026. arXiv:2511.08971. — Source of the
  three-way text/vision/cross-modal clarifier decomposition used throughout `clarifier.py`.
  We use the decomposition, not their 3D-pointing-gesture geometry (out of scope — no
  gesture/pose stream in this task's I/O contract).
- "BLaVe-CoT: Consistency-Aware VQA for Blind and Low Vision Users." arXiv:2509.06010. —
  Source of the multiple-valid-groundings / margin-based ambiguity framing used in
  `CrossModalClarifier`.
- Dong, Y. et al. "XGrammar: Flexible and Efficient Structured Generation Engine for Large
  Language Models." arXiv:2411.15100. — `XGrammarBackend` in `constrained_output.py`. API
  usage confirmed against xgrammar.mlc.ai's current docs.
- Meta patent, "Multimodal dialog state tracking and action prediction for assistant
  systems" — prior-art reference only (not peer-reviewed) for the sliding-window visual
  buffer pattern in `visual_buffer.py`.
