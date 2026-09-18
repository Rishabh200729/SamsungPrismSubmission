"""
module3/examples/demo_runtime.py

End-to-end demonstration of the PRISM Module 3 runtime.

Runs the complete interruption scenario and prints:
1. The live execution trace
2. Evaluation metrics
3. Assertion results

Run with:
    python -m module3.examples.demo_runtime
"""

import asyncio
import sys
from pathlib import Path

# Make sure module3 is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from module3.runtime.runtime import Runtime
from module3.runtime.clock.virtual_clock import VirtualClock
from module3.runtime.config import TEST_CONFIG
from module3.runtime.events import (
    make_text_chunk, make_end_of_turn, make_interruption,
    make_filler, make_tool_call, make_final_response, StateSnapshot,
)
from module3.runtime.events.base import InputEventType
from module3.evaluation.scenario import Scenario, ScenarioEvent, TraceAssertion, OutputAssertion
from module3.evaluation.runner import ScenarioResult
from module3.evaluation.assertions import run_assertions
from module3.evaluation.metrics import compute_metrics
from module3.evaluation.report import build_report


async def main():
    print("\n" + "="*70)
    print("  PRISM Module 3 — Runtime Demo")
    print("  Theme 05: Interruptible Real-Time Agents")
    print("="*70)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    clock = VirtualClock(start_ms=0.0)
    rt = Runtime(config=TEST_CONFIG, clock=clock)

    # Track key IDs across handlers
    state = {
        "slow_task_id": None,
        "slow_call_id": None,
        "new_task_id": None,
    }

    # ------------------------------------------------------------------
    # Handler simulation (what Modules 1 and 2 would plug in)
    # ------------------------------------------------------------------
    async def on_end_of_turn(event):
        """Simulate Module 2: emit filler + tool call, start slow task."""
        sid = event.session_id
        gen = rt.get_current_generation(sid)
        print(f"\n[Module 2] END_OF_TURN received. Starting slow reasoning (gen={gen})")

        # Emit FILLER (fast path)
        await rt.emit_output(
            make_filler(sid, clock.now() + 5.0, "Got it! Looking into that for you...",
                        generation=gen)
        )

        # Start slow background task
        async def slow_reasoning():
            print(f"[Module 2] Slow reasoning started for gen={gen}...")
            await clock.sleep(700.0)  # Will be interrupted
            print(f"[Module 2] Slow reasoning completed (gen={gen})")

        task_id = await rt.create_background_task(sid, slow_reasoning(), "slow_reasoning")
        state["slow_task_id"] = task_id

        # Emit TOOL_CALL
        evt, cid = make_tool_call(
            sid, clock.now() + 200.0,
            "mock_booking",
            arguments={"origin": "Seoul", "destination": "Tokyo"},
            task_id=task_id,
            generation=gen,
        )
        state["slow_call_id"] = cid
        await rt.emit_output(evt)
        print(f"[Module 2] TOOL_CALL emitted call_id={cid[:8]}...")

    async def on_interruption(event):
        """Simulate Module 1: handle interruption, cancel current work."""
        sid = event.session_id
        old_gen, new_gen = await rt.request_cancellation(sid, reason="user_interruption")
        print(f"\n[Module 1] INTERRUPTION! Generation {old_gen} -> {new_gen}")

        # Emit FILLER acknowledging interruption
        await rt.emit_output(
            make_filler(sid, clock.now() + 2.0, "Sure, let me adjust that!",
                        filler_type="acknowledgment", generation=new_gen)
        )

    rt.register_handler(InputEventType.END_OF_TURN, on_end_of_turn)
    rt.register_handler(InputEventType.INTERRUPTION, on_interruption)

    await rt.start()
    sid = rt.create_session()

    print(f"\n[Runtime] Session created: {sid[:8]}...")

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------
    print("\n--- T=0: User says 'Book a flight from Seoul to Tokyo' ---")
    await rt.submit_event(make_text_chunk(sid, 0.0, "Book a flight from Seoul to Tokyo"))
    await rt.submit_event(make_end_of_turn(sid, 10.0, final_text="Book a flight from Seoul to Tokyo"))
    await rt.input_queue.join()
    await asyncio.sleep(0)

    print("\n--- Advancing to T=450ms (interruption) ---")
    await clock.advance_async(450.0)

    print("\n--- T=450: User interrupts ---")
    await rt.submit_event(
        make_interruption(sid, 450.0, text="Actually change destination to Osaka")
    )
    await rt.input_queue.join()
    await asyncio.sleep(0)

    # T=700: Stale TOOL_RESULT arrives for the old call
    # After interrupt, generation is now 2; old call from gen 1 is stale
    if state["slow_call_id"] and state["slow_task_id"]:
        # Ensure interrupt handler has fully processed
        await asyncio.sleep(0)
        current_gen = rt.get_current_generation(sid)
        print(f"\n[Runtime] T=701: Current generation={current_gen}. Old call was gen=1.")
        if current_gen > 1:
            print(f"[Runtime] Discarding stale result for call {state['slow_call_id'][:8]}...")
            await rt.handle_stale_result(sid, state["slow_task_id"], state["slow_call_id"])
        else:
            print(f"[Runtime] Interrupt not yet processed — skipping stale check")

    # New task for generation 2
    print("\n--- T=720: New task for Osaka booking (generation 2) ---")
    async def new_booking():
        await asyncio.sleep(0)

    new_tid = await rt.create_background_task(
        sid, new_booking(), "slow_reasoning_gen2",
        metadata={"intent": "book_flight", "destination": "Osaka"}
    )
    state["new_task_id"] = new_tid

    await asyncio.sleep(0.05)

    # Final response
    print("\n--- T=900: Emitting final response ---")
    snap = StateSnapshot(
        intent="book_flight",
        slots={"origin": "Seoul", "destination": "Osaka"},
        completed_actions=["search", "book"],
        result_data={"booking_id": "BK-99901", "status": "confirmed"},
    )
    await rt.emit_output(
        make_final_response(
            sid, 900.0, "Your flight from Seoul to Osaka has been booked! Booking ID: BK-99901",
            state_snapshot=snap, task_id=new_tid,
            generation=rt.get_current_generation(sid),
        )
    )

    await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    outputs = await rt.drain_outputs()
    rt.tracer.print_summary()

    result = ScenarioResult(
        scenario_name="demo_interruption",
        session_ids={"default": sid},
        outputs=outputs,
        trace=rt.tracer.entries(),
        final_virtual_time_ms=900.0,
    )
    metrics = compute_metrics(result)

    # Build assertions
    scenario = Scenario(
        name="demo_interruption",
        trace_assertions=[
            TraceAssertion("TASK_CREATED", min_count=1, description="Task was created"),
            TraceAssertion("GENERATION_INVALIDATED", min_count=1, description="Generation was invalidated"),
            TraceAssertion("STALE_RESULT_DISCARDED", min_count=1, description="Stale result was discarded"),
        ],
        output_assertions=[
            OutputAssertion("FILLER", min_count=1, description="Filler was emitted"),
            OutputAssertion("TOOL_CALL", min_count=1, description="Tool call was emitted"),
            OutputAssertion("FINAL_RESPONSE", min_count=1, description="Final response was emitted"),
        ],
    )
    assertions = run_assertions(scenario, result)
    report = build_report("demo_interruption", result, metrics, assertions)
    report.print_human_readable()

    await rt.stop()
    print("\n[Runtime] Demo complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
