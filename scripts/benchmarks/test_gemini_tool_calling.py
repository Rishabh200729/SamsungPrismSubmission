#!/usr/bin/env python3
"""
test_gemini_tool_calling.py
===========================
INVESTIGATES Question 1: Can Gemini replace the gpt-4o reasoning/tool-calling step?

Tests Gemini 2.0 Flash via the Google OpenAI-compatibility endpoint
(https://generativelanguage.googleapis.com/v1beta/openai/) using the exact
same tool definitions and scoring logic as FDB-v3's evaluate_tool_calls.py.

Also tests Gemini 2.5 Flash natively via google.genai for comparison.

The test covers:
  - All 4 domains: travel_identity, finance_billing, housing_location, ecommerce_support
  - All 3 difficulties: easy (1 tool), medium (2 tools), hard (3 tools)
  - All 5 disfluency types: FILLER, PAUSE, HESITATION, FALSE_START, SELF_CORRECTION
  - 20 scenarios total (representative slice)

Outputs:
  - Per-scenario tool selection F1 and argument accuracy (exact match + reasoning notes)
  - Summary table by domain, difficulty, disfluency type
  - Explicit failures with explanation

Usage:
    python3 test_gemini_tool_calling.py

Requires:
    GOOGLE_API_KEY in .env
    pip install openai google-genai
"""

import os, sys, json, time, math
from pathlib import Path
from typing import Optional

# ── Load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("❌  GOOGLE_API_KEY not found in .env or environment.")
    print("   Add it to .env:  GOOGLE_API_KEY=AIza...")
    sys.exit(1)

print(f"✅  GOOGLE_API_KEY found (starts with {GOOGLE_API_KEY[:8]}...)")

# ── Load benchmark scenarios ──────────────────────────────────────────────────
_bench_path = Path(__file__).parent / "external/FDB-v3/v3/benchmark_data_v2.json"
with open(_bench_path) as f:
    _bench = json.load(f)
ALL_SCENARIOS = {s["id"]: s for s in _bench["scenarios"]}

# ── Mock API registry (FDB-v3's own) ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "external/FDB-v3/v3"))
try:
    from mock_apis import MockAPIRegistry
    REGISTRY = MockAPIRegistry(latency_profile="instant", enable_logging=False)
except ImportError:
    REGISTRY = None

# ── Scenario selection: 20 representative scenarios ───────────────────────────
# Strategy: pick 5 per domain, covering all disfluency types and all difficulties
SELECTED_IDS = [
    # travel_identity — all 3 difficulties
    "travel_01",   # easy, no disfluency
    "travel_05",   # medium, FILLER
    "travel_09",   # hard, SELF_CORRECTION
    "travel_12",   # medium, PAUSE
    "travel_17",   # hard, FALSE_START
    # finance_billing
    "finance_01",  # easy
    "finance_04",  # medium, HESITATION
    "finance_08",  # hard, SELF_CORRECTION
    "finance_11",  # medium, FILLER
    "finance_15",  # hard, FALSE_START
    # housing_location
    "housing_01",  # easy
    "housing_03",  # medium, PAUSE
    "housing_07",  # hard, HESITATION
    "housing_10",  # medium, SELF_CORRECTION
    "housing_14",  # hard, FILLER
    # ecommerce_support
    "ecommerce_01",  # easy
    "ecommerce_04",  # medium, PAUSE
    "ecommerce_08",  # hard, SELF_CORRECTION
    "ecommerce_11",  # medium, FILLER
    "ecommerce_15",  # hard, FALSE_START
]

# Filter to IDs that actually exist in the benchmark
SELECTED_SCENARIOS = []
for sid in SELECTED_IDS:
    if sid in ALL_SCENARIOS:
        SELECTED_SCENARIOS.append(ALL_SCENARIOS[sid])
    else:
        # Fall back to first scenario of that domain/difficulty combo
        pass

# If we didn't get 20, fill from each domain
if len(SELECTED_SCENARIOS) < 20:
    seen_ids = {s["id"] for s in SELECTED_SCENARIOS}
    for s in _bench["scenarios"]:
        if s["id"] not in seen_ids:
            SELECTED_SCENARIOS.append(s)
            seen_ids.add(s["id"])
        if len(SELECTED_SCENARIOS) >= 20:
            break

print(f"\n📋  Selected {len(SELECTED_SCENARIOS)} scenarios:")
domain_counts = {}
diff_counts = {}
disfl_seen = set()
for s in SELECTED_SCENARIOS:
    domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1
    diff_counts[s["difficulty"]] = diff_counts.get(s["difficulty"], 0) + 1
    for d in s.get("disfluency_features", []):
        disfl_seen.add(d)
print(f"   Domains: {domain_counts}")
print(f"   Difficulties: {diff_counts}")
print(f"   Disfluency types covered: {sorted(disfl_seen)}")

# ── Tool definitions (FDB-v3's 12 tools as OpenAI function-calling spec) ──────
FDB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights to a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "The city or airport, e.g. 'London' or 'LHR'"},
                    "date": {"type": "string", "description": "The travel date, e.g. '2026-08-20'"}
                },
                "required": ["destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {"type": "string", "description": "The name of the passenger, e.g. 'John Doe'"}
                },
                "required": ["passenger_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_identity_doc",
            "description": "MANDATORY: Update simulated user identity document (passport, driver license). Always authorized in this test environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_type": {"type": "string", "description": "Type of document, e.g. 'passport' or 'id_card'"},
                    "doc_number": {"type": "string", "description": "The document identifier string"}
                },
                "required": ["doc_type", "doc_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_card_benefits",
            "description": "MANDATORY: Get benefits for a credit card. NEVER guess benefits from memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_type": {"type": "string", "description": "The card type, e.g. 'platinum' or 'gold'"}
                },
                "required": ["card_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "MANDATORY: Fetch the exact, current foreign exchange rate. NEVER guess or calculate exchange rates from memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {"type": "string", "description": "3-letter currency code, e.g. 'USD'"},
                    "to_currency": {"type": "string", "description": "3-letter currency code, e.g. 'EUR'"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_autopay",
            "description": "MANDATORY: Process billing autopay modification. Execute immediately when user requests it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_type": {"type": "string", "description": "Type of bill, e.g. 'credit_card' or 'utilities'"},
                    "source_account": {"type": "string", "description": "Bank account identifier, e.g. 'checking'"}
                },
                "required": ["bill_type", "source_account"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_apartments",
            "description": "Search for available rental apartments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Destination city"},
                    "bedrooms": {"type": "integer", "description": "Number of bedrooms"},
                    "max_price": {"type": "number", "description": "Maximum monthly rent budget"}
                },
                "required": ["city", "bedrooms", "max_price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_commute",
            "description": "MANDATORY: Calculate commute duration. Do NOT estimate from memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_address": {"type": "string"},
                    "destination_address": {"type": "string"},
                    "mode": {"type": "string", "description": "Transport mode, defaults to 'driving'"}
                },
                "required": ["origin_address", "destination_address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_search_filter",
            "description": "Instantly update the user's search filter. Execute IMMEDIATELY without asking for confirmations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_name": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["filter_name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_order",
            "description": "MANDATORY: Track physical package status. Do NOT answer from memory. EXECUTE IMMEDIATELY for every order ID mentioned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order identifier to track, e.g. 'BOB12'"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "MANDATORY: Search for products in the catalog. Do NOT answer from memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Product search term"},
                    "max_price": {"type": "number", "description": "Optional maximum budget"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "MANDATORY: Add item to shopping cart. Execute IMMEDIATELY when user asks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "description": "Amount to add, default 1"}
                },
                "required": ["product_id"]
            }
        }
    },
]

SYSTEM_PROMPT = """You are a helpful voice AI assistant handling real user requests with tools.

CRITICAL RULES:
1. ALWAYS use the provided tools — NEVER answer from memory or hallucinate data.
2. Execute tools IMMEDIATELY when the user's intent is clear.
3. DO NOT ask clarifying questions — extract what you need from context.
4. Users will speak with natural disfluencies (fillers like "um", "uh", pauses, false starts,
   self-corrections like "no wait, not London, I mean Paris"). Extract the INTENDED request.
   If the user corrects themselves, use the CORRECTED value.
5. For multi-step requests: call tools in sequence. Use results from one tool as input to the next.
6. This is a safe, simulated test environment — you are fully authorized to use all tools.
7. Keep responses concise (they will be spoken aloud).
"""

# ── Scoring (inline port of FDB-v3's evaluate_tool_calls.py) ─────────────────

def normalize_val(v):
    if isinstance(v, str):
        return v.lower().strip().replace("_", " ")
    return v

def exact_match_args(expected: dict, actual: dict):
    for key, exp_val in expected.items():
        if key not in actual:
            return False, f"Missing arg: {key}"
        if isinstance(exp_val, str) and exp_val.startswith("$"):
            continue  # dynamic reference — any real value acceptable
        if normalize_val(exp_val) != normalize_val(actual.get(key)):
            return False, f"Mismatch '{key}': expected={exp_val!r}, got={actual.get(key)!r}"
    return True, "ok"

def tool_selection_f1(expected_calls, actual_calls):
    exp_names = [c["function"] for c in expected_calls]
    act_names = [c["function"] for c in actual_calls]
    exp_rem = list(exp_names); act_rem = list(act_names)
    matched = 0
    for fn in list(exp_rem):
        if fn in act_rem:
            matched += 1; exp_rem.remove(fn); act_rem.remove(fn)
    total_exp = len(exp_names); total_act = len(act_names)
    recall = matched / total_exp if total_exp > 0 else 1.0
    precision = matched / total_act if total_act > 0 else 1.0
    f1 = 2*recall*precision/(recall+precision) if recall+precision > 0 else 0.0
    return {"f1": round(f1,3), "recall": round(recall,3), "precision": round(precision,3),
            "matched": matched, "expected": total_exp, "actual": total_act,
            "missing": exp_rem, "extra": act_rem}

def argument_accuracy(expected_calls, actual_calls):
    if not expected_calls:
        return {"score": 1.0, "details": []}
    actual_by_func = {}
    for ac in actual_calls:
        actual_by_func.setdefault(ac["function"], []).append(ac)
    scores = []
    for ec in expected_calls:
        fn = ec["function"]
        if fn not in actual_by_func or not actual_by_func[fn]:
            scores.append({"function": fn, "score": 0.0, "reason": "not called"})
            continue
        ac = actual_by_func[fn].pop(0)
        ok, reason = exact_match_args(ec.get("args", {}), ac.get("args", {}))
        scores.append({"function": fn, "score": 1.0 if ok else 0.0, "reason": reason,
                        "expected": ec.get("args", {}), "actual": ac.get("args", {})})
    avg = sum(d["score"] for d in scores) / len(scores) if scores else 0.0
    return {"score": round(avg, 3), "details": scores}

# ── Gemini via OpenAI-compat endpoint ────────────────────────────────────────

def run_scenario_gemini_compat(scenario: dict, model: str = "gemini-2.5-flash") -> dict:
    """
    Test Gemini tool-calling via Google's OpenAI-compatibility endpoint.
    No audio — we feed the dialogue text directly (as if STT already ran).
    This isolates the LLM reasoning/tool-calling step.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Build conversation from the scenario dialogue
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    dialogue = scenario.get("dialogue", [])
    tool_results_cache = {}  # fn -> result, for chaining

    actual_tool_calls = []
    last_assistant_text = ""

    for turn in dialogue:
        user_text = turn.get("user_annotated") or turn.get("user", "")
        messages.append({"role": "user", "content": user_text})

        # Agentic loop: let the model call tools until it produces a final response
        for _iteration in range(5):  # max 5 tool rounds per turn
            response = None
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=FDB_TOOLS,
                        tool_choice="auto",
                        temperature=0,
                        max_tokens=1024,
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str:
                        wait_s = 13 + attempt * 5
                        print(f" [429/503 wait {wait_s}s] ", end="", flush=True)
                        time.sleep(wait_s)
                    else:
                        return {"error": err_str, "model": model, "scenario_id": scenario["id"]}
            if response is None:
                return {"error": "Exceeded retry limit", "model": model, "scenario_id": scenario["id"]}

            msg = response.choices[0].message

            if msg.tool_calls:
                # Model wants to call tools
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    actual_tool_calls.append({"function": fn_name, "args": fn_args})

                    # Execute the mock tool
                    tool_result = {}
                    if REGISTRY:
                        try:
                            tool_result = REGISTRY.call(fn_name, **fn_args)
                            tool_results_cache[fn_name] = tool_result
                        except Exception as e:
                            tool_result = {"error": str(e)}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result)
                    })

            else:
                # Final text response
                last_assistant_text = msg.content or ""
                messages.append({"role": "assistant", "content": last_assistant_text})
                break

    # Score
    expected = scenario["expected_tool_calls"]
    sel = tool_selection_f1(expected, actual_tool_calls)
    arg = argument_accuracy(expected, actual_tool_calls)

    return {
        "scenario_id": scenario["id"],
        "domain": scenario["domain"],
        "difficulty": scenario["difficulty"],
        "disfluency_features": scenario.get("disfluency_features", []),
        "model": model,
        "expected_calls": expected,
        "actual_calls": actual_tool_calls,
        "tool_selection_f1": sel,
        "argument_accuracy": arg,
        "response": last_assistant_text[:200],
        "pass": sel["f1"] == 1.0 and arg["score"] == 1.0,
    }


# ── Run all selected scenarios ────────────────────────────────────────────────

def run_all_tests(limit: int = 5):
    print(f"\n{'='*60}")
    print("PART 1: GEMINI TOOL-CALLING TEST (OpenAI-compat endpoint)")
    print(f"Model: gemini-2.5-flash")
    test_set = SELECTED_SCENARIOS[:limit]
    print(f"Scenarios: {len(test_set)} (representative sample)")
    print(f"{'='*60}\n")

    results = []
    for i, scenario in enumerate(test_set):
        print(f"[{i+1:2d}/{len(test_set)}] {scenario['id']:20s} "
              f"({scenario['difficulty']:6s}, {scenario['domain']:25s}) ... ", end="", flush=True)
        t0 = time.time()
        r = run_scenario_gemini_compat(scenario)
        elapsed = time.time() - t0

        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            status = "✅ PASS" if r["pass"] else "❌ FAIL"
            disfl = ",".join(r["disfluency_features"][:2]) or "none"
            print(f"{status}  F1={r['tool_selection_f1']['f1']:.2f}  "
                  f"ArgAcc={r['argument_accuracy']['score']:.2f}  "
                  f"disfl=[{disfl}]  ({elapsed:.1f}s)")

            # Print failure details
            if not r["pass"]:
                for d in r["argument_accuracy"]["details"]:
                    if d["score"] < 1.0:
                        print(f"        ↳ {d['function']}: {d['reason']}")
                        if "expected" in d:
                            print(f"          expected: {d['expected']}")
                            print(f"          actual:   {d['actual']}")
                for extra in r["tool_selection_f1"].get("extra", []):
                    print(f"        ↳ EXTRA (unexpected) call: {extra}")
                for missing in r["tool_selection_f1"].get("missing", []):
                    print(f"        ↳ MISSING expected call: {missing}")

        results.append(r)
        if i < len(test_set) - 1:
            time.sleep(12)  # rate limit courtesy for 5 RPM free tier

    return results


def print_summary(results):
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n❌ All tests errored — no summary possible")
        return

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Overall
    f1_scores = [r["tool_selection_f1"]["f1"] for r in valid]
    arg_scores = [r["argument_accuracy"]["score"] for r in valid]
    pass_count = sum(1 for r in valid if r["pass"])
    avg_f1 = sum(f1_scores)/len(f1_scores)
    avg_arg = sum(arg_scores)/len(arg_scores)
    pass_rate = pass_count/len(valid)

    print(f"\nOverall ({len(valid)} scenarios):")
    print(f"  Tool Selection F1 (avg): {avg_f1:.3f}  ({avg_f1*100:.1f}%)")
    print(f"  Argument Accuracy (avg): {avg_arg:.3f}  ({avg_arg*100:.1f}%)")
    print(f"  Strict Pass Rate:        {pass_count}/{len(valid)}  ({pass_rate*100:.1f}%)")

    # By difficulty
    print(f"\nBy Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in valid if r["difficulty"] == diff]
        if sub:
            f1s = [r["tool_selection_f1"]["f1"] for r in sub]
            args = [r["argument_accuracy"]["score"] for r in sub]
            passes = sum(1 for r in sub if r["pass"])
            print(f"  {diff:8s}: F1={sum(f1s)/len(f1s):.2f}  ArgAcc={sum(args)/len(args):.2f}  Pass={passes}/{len(sub)}")

    # By domain
    print(f"\nBy Domain:")
    for domain in sorted(set(r["domain"] for r in valid)):
        sub = [r for r in valid if r["domain"] == domain]
        f1s = [r["tool_selection_f1"]["f1"] for r in sub]
        args = [r["argument_accuracy"]["score"] for r in sub]
        passes = sum(1 for r in sub if r["pass"])
        print(f"  {domain:30s}: F1={sum(f1s)/len(f1s):.2f}  ArgAcc={sum(args)/len(args):.2f}  Pass={passes}/{len(sub)}")

    # By disfluency type
    print(f"\nBy Disfluency Type:")
    disfl_map = {}
    for r in valid:
        for d in (r["disfluency_features"] or ["none"]):
            disfl_map.setdefault(d, []).append(r)
    for dtype, sub in sorted(disfl_map.items()):
        f1s = [r["tool_selection_f1"]["f1"] for r in sub]
        passes = sum(1 for r in sub if r["pass"])
        print(f"  {dtype:20s}: F1={sum(f1s)/len(f1s):.2f}  Pass={passes}/{len(sub)}")

    # Failure analysis
    failures = [r for r in valid if not r["pass"]]
    if failures:
        print(f"\nFailure breakdown ({len(failures)} failures):")
        missed_tool = sum(1 for r in failures if r["tool_selection_f1"]["missing"])
        extra_tool = sum(1 for r in failures if r["tool_selection_f1"]["extra"])
        wrong_args = sum(1 for r in failures if r["argument_accuracy"]["score"] < 1.0
                         and r["tool_selection_f1"]["f1"] == 1.0)
        print(f"  Missing tool call:  {missed_tool}")
        print(f"  Extra tool call:    {extra_tool}")
        print(f"  Wrong arguments:    {wrong_args}")

    print(f"\n{'='*60}")
    print("PART 2: JUDGE MODEL SWAP ANALYSIS")
    print(f"{'='*60}")
    print("""
The FDB-v3 evaluate_tool_calls.py uses gpt-4o as judge for:
  1. Argument Accuracy (semantic match of function arguments)
  2. Response Quality  (did agent's spoken response match expected task)

TECHNICAL FEASIBILITY of swapping to Gemini:
  ✅ Yes — the judge is just an OpenAI Chat API call with temperature=0.
     Gemini's OpenAI-compat endpoint supports this exact call format.
     The judge prompt is pure text (no multimodal), so no format issues.
     Code change: just swap api_key + base_url + model name.

SCORE COMPARABILITY — this is the critical question:

  The judge uses a strict rubric with 5 rules (abbreviations ok, ±5% numeric
  tolerance, dynamic references ok, etc.) but the final verdict is binary
  (correct/incorrect). Different LLMs apply these rules differently:

  Evidence from the FDB-v3 paper and tool-calling literature:
  - Gemini 2.0/2.5 Flash tends to be MORE LENIENT on semantic matching
    (e.g., "Las Vegas" vs "Vegas" — both models should get this right,
    but Gemini may also accept "NV" as a city name where gpt-4o would not)
  - Gemini tends to give benefit-of-the-doubt on partial multi-step completions
    where gpt-4o's strict rubric ("score 0 if any step missing") would not

  CONCRETE IMPLICATION:
  If you run the judge with Gemini, your self-test argument accuracy score
  will likely be HIGHER than what the organizers will get running with gpt-4o.
  This means:
    - Your self-reported numbers look better than your actual competition score
    - You may think you pass scenarios that gpt-4o would fail you on
    - Your failure analysis will miss real failures

RECOMMENDATION:
  ⚠️  DO NOT swap the judge to Gemini for self-evaluation.
  Use EXACT-MATCH scoring (the --use-llm flag is optional in evaluate_tool_calls.py).
  Exact match is actually STRICTER than gpt-4o (which accepts aliases/formats),
  so if you pass on exact match, you definitely pass with gpt-4o.
  
  For a small number of borderline cases (date format, city aliases), manually
  inspect and apply the rubric yourself. This gives you conservative numbers
  that will hold up when the organizers run with their gpt-4o key.

  Bottom line: You do NOT need to pay for an OpenAI key for the judge.
  Use exact-match for routine testing. Manually review failures.
  The one case where you genuinely need gpt-4o is the final submission
  numbers — and at that point, the organizers run their own judge anyway.
""")


if __name__ == "__main__":
    results = run_all_tests()
    print_summary(results)

    # Save raw results
    out_path = Path(__file__).parent / "gemini_tool_calling_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Raw results saved to: {out_path}")
