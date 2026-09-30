#!/usr/bin/env python3
"""
scripts/run_fdb_evaluation.py — TRAX Evaluation Runner
Runs evaluation scenarios against an untouched FDB-v3 clone directory.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("prism.eval")


def run_evaluation(fdb_path: Path, scenarios: list[str], use_llm: bool = False, output_file: str = None):
    fdb_path = fdb_path.resolve()
    if not (fdb_path / "evaluate_tool_calls.py").exists():
        raise FileNotFoundError(f"evaluate_tool_calls.py not found in {fdb_path}")

    # Add FDB path to sys.path so we can import stock evaluation functions
    sys.path.insert(0, str(fdb_path))
    from evaluate_tool_calls import evaluate_scenario
    from evaluate_pass_rate import evaluate_scenario_pass

    data_dir = fdb_path / "fdb_v3_data_released"
    if not data_dir.exists():
        raise FileNotFoundError(f"Benchmark data directory not found in {data_dir}")

    # Read telemetry tool calls
    calls_by_room = {}
    tool_log_path = Path("/tmp/agent_tool_calls.log")
    if tool_log_path.exists():
        with open(tool_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        calls_by_room.setdefault(d.get("room"), []).append(d.get("call"))
                    except Exception:
                        pass

    results = []

    print("=" * 80)
    print(f"TRAX EVALUATION RUNNER — Target: {fdb_path.name} | LLM Judge: {use_llm}")
    print("=" * 80)

    for scen_id in scenarios:
        # Find scenario directory
        matching = list(data_dir.glob(f"{scen_id}_*"))
        if not matching:
            print(f"⚠️ Scenario {scen_id} not found in {data_dir}")
            continue

        scen_folder = matching[0]
        meta_file = scen_folder / "metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            scen_meta = json.load(f)

        # Look for calls for this scenario room or fallback
        # Room names follow eval-room-{scen_id}*
        actual_calls = []
        for room_name, calls in calls_by_room.items():
            if scen_id.replace("_", "") in room_name.replace("-", "").replace("_", ""):
                actual_calls = calls

        # Run stock evaluation functions
        eval_tool = evaluate_scenario(scen_meta, actual_calls, use_llm=use_llm)
        eval_pass = evaluate_scenario_pass(scen_meta, actual_calls, use_llm=use_llm)

        metrics = eval_tool["metrics"]
        ts = metrics["tool_selection_acc"]
        aa = metrics["argument_acc"]
        passed = eval_pass["passed"]

        print(f"\nScenario: {scen_id} ({scen_meta.get('title', '')})")
        print(f"  Actual Calls:       {actual_calls}")
        print(f"  Expected Calls:     {scen_meta.get('expected_tool_calls', [])}")
        print(f"  Tool Selection F1:  {ts.get('score')} (Recall={ts.get('recall')}, Precision={ts.get('precision')})")
        print(f"  Argument Accuracy:  {aa.get('score')}")
        print(f"  Pass@1 Status:      {'PASS (1.0)' if passed else 'FAIL (0.0)'}")
        print(f"  Checks:             {eval_pass.get('checks', {})}")

        results.append({
            "scenario": scen_id,
            "title": scen_meta.get("title", ""),
            "tool_selection_f1": ts.get("score"),
            "argument_accuracy": aa.get("score"),
            "passed": passed,
        })

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TRAX FDB Benchmark Evaluator")
    parser.add_argument("--fdb-path", type=str, default="external/FDB-v3/v3",
                        help="Path to FDB-v3 repository root")
    parser.add_argument("--scenarios", nargs="+", default=["travel_10", "travel_19"],
                        help="Scenario IDs to evaluate")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM Judge")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results")
    args = parser.parse_args()

    run_evaluation(Path(args.fdb_path), args.scenarios, use_llm=args.use_llm, output_file=args.output)


if __name__ == "__main__":
    main()
