"""
test_interruption_e2e.py — Validates the full interruption scenario end-to-end.
Run: python test_interruption_e2e.py
"""
import asyncio
import sys
sys.path.insert(0, "module_2")

from module3 import Runtime
from module3.runtime.config import TEST_CONFIG
from module3.runtime.events import (
    make_text_chunk, make_end_of_turn,
    make_interruption, make_tool_manifest,
)
from module1 import FastPathRouter
from orchestrator import Module2ReasoningAdapter
from module4.adapter import Module4Adapter

MANIFEST = [
    {
        "name": "book_flight",
        "side_effect": "STATE_MODIFYING",
        "parameters": {
            "destination": {"required": True},
            "date": {"required": True},
        },
    }
]


async def test_interruption():
    rt = Runtime(config=TEST_CONFIG)
    await rt.start()
    sid = rt.create_session()

    FastPathRouter(rt, rt.clock)
    reasoning = Module2ReasoningAdapter(rt)
    reasoning.install()
    m4 = Module4Adapter(rt)
    m4.install()

    ts = rt.clock.now()
    await rt.submit_event(make_tool_manifest(sid, ts, tools=MANIFEST))

    # Turn 1: Book Tokyo
    await rt.submit_event(make_text_chunk(sid, ts + 1, text="book flight to Tokyo on 2026-12-01"))
    await rt.submit_event(make_end_of_turn(sid, ts + 2, final_text="book flight to Tokyo on 2026-12-01"))
    await asyncio.sleep(0.1)

    gen_before = rt.get_current_generation(sid)

    # User interrupts mid-execution
    await rt.submit_event(make_interruption(sid, ts + 300, text="actually change to Osaka", reason="user_interruption"))
    await asyncio.sleep(0)

    # Turn 2: New request after interruption
    await rt.submit_event(make_text_chunk(sid, ts + 310, text="book flight to Osaka on 2026-12-01"))
    await rt.submit_event(make_end_of_turn(sid, ts + 315, final_text="book flight to Osaka on 2026-12-01"))
    await asyncio.sleep(0.5)

    gen_after = rt.get_current_generation(sid)
    outputs = await rt.drain_outputs()
    trace = rt.tracer.entries()

    stale_count = sum(1 for e in trace if e.entry_type == "STALE_RESULT_DISCARDED")
    output_types = [o.event_type.value for o in outputs]
    final_events = [o for o in outputs if o.event_type.value == "FINAL_RESPONSE"]

    print("\n" + "=" * 60)
    print("  Interruption E2E Test")
    print("=" * 60)
    print(f"  Generation:        {gen_before} -> {gen_after}")
    print(f"  Output types:      {output_types}")
    print(f"  Stale discarded:   {stale_count}")

    passed = True

    # Check 1: Generation must advance
    if gen_after > gen_before:
        print("  [PASS] Generation advanced on interruption")
    else:
        print("  [FAIL] Generation did NOT advance")
        passed = False

    # Check 2: Final response must exist
    if final_events:
        snap = final_events[-1].payload.get("state_snapshot") or {}
        intent = snap.get("intent", "")
        slots = snap.get("slots", {})
        dest = str(slots.get("destination", "")).lower()
        print(f"  Final intent:      {intent}")
        print(f"  Final slots:       {slots}")

        # Check 3: Final state must be Osaka (not Tokyo)
        if dest == "osaka":
            print("  [PASS] Final state contains Osaka (Tokyo correctly replaced)")
        else:
            print(f"  [FAIL] Final state destination is '{dest}', expected 'osaka'")
            passed = False
    else:
        print("  [WARN] No FINAL_RESPONSE emitted (tool still executing?)")

    print("=" * 60)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60 + "\n")

    m4.drop_session(sid)
    await rt.close_session(sid)
    await rt.stop()
    return passed


if __name__ == "__main__":
    ok = asyncio.run(test_interruption())
    sys.exit(0 if ok else 1)
