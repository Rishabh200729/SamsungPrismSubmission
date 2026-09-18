"""Proof that external modules can use Module 3 without internal imports."""

import asyncio

# These are deliberately the only Module 3 imports in this file: public API.
from module3.runtime import IRuntimeClient, Runtime, TEST_CONFIG
from module3.runtime.events import (
    InputEventType,
    OutputEventType,
    make_clarification,
    make_end_of_turn,
    make_filler,
    make_final_response,
    make_interruption,
    make_text_chunk,
    make_video_frame,
)


class FakeModule1:
    """Fast path / interruption manager using only IRuntimeClient methods."""

    def __init__(self, runtime: IRuntimeClient) -> None:
        self.runtime = runtime

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.TEXT_CHUNK, self.on_text)
        self.runtime.register_handler(InputEventType.INTERRUPTION, self.on_interrupt)

    async def on_text(self, event) -> None:
        await self.runtime.emit_output(
            make_filler(event.session_id, event.timestamp_ms, "Module 1 acknowledged"),
            priority=10,
        )

    async def on_interrupt(self, event) -> None:
        await self.runtime.request_cancellation(event.session_id, reason="fake_module_1_interrupt")


class FakeModule2:
    """Slow path using tracked background work and only public task APIs."""

    def __init__(self, runtime: IRuntimeClient) -> None:
        self.runtime = runtime
        self.task_id: str | None = None

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.END_OF_TURN, self.on_end_of_turn)

    async def on_end_of_turn(self, event) -> None:
        async def reason() -> None:
            await asyncio.sleep(0)

        self.task_id = await self.runtime.create_background_task(
            event.session_id, reason(), "fake_module_2_reasoning"
        )
        await self.runtime.emit_output(
            make_final_response(event.session_id, event.timestamp_ms + 1, "Module 2 answer"),
        )


class FakeModule4:
    """Multimodal handler using only event registration and output emission."""

    def __init__(self, runtime: IRuntimeClient) -> None:
        self.runtime = runtime

    def install(self) -> None:
        self.runtime.register_handler(InputEventType.VIDEO_FRAME, self.on_frame)

    async def on_frame(self, event) -> None:
        await self.runtime.emit_output(
            make_clarification(event.session_id, event.timestamp_ms, "Module 4 needs a clearer frame"),
        )


async def test_fake_modules_integrate_through_public_contract_only():
    runtime = Runtime(config=TEST_CONFIG)
    assert isinstance(runtime, IRuntimeClient)

    module1 = FakeModule1(runtime)
    module2 = FakeModule2(runtime)
    module4 = FakeModule4(runtime)
    module1.install()
    module2.install()
    module4.install()

    await runtime.start()
    session_id = runtime.create_session(metadata={"integration": "public-contract"})
    try:
        await runtime.submit_event(make_text_chunk(session_id, 0, "hello"))
        await runtime.submit_event(make_video_frame(session_id, 1, "", 1, 1, 0))
        await runtime.submit_event(make_end_of_turn(session_id, 2, "hello"))
        outputs = [await runtime.next_output(timeout_ms=50) for _ in range(3)]
        assert module2.task_id is not None
        assert runtime.get_task(session_id, module2.task_id) is not None
        assert [output.event_type for output in outputs] == [
            OutputEventType.FILLER,
            OutputEventType.CLARIFICATION,
            OutputEventType.FINAL_RESPONSE,
        ]

        old_generation, new_generation = await runtime.request_cancellation(session_id)
        assert (old_generation, new_generation) == (1, 2)
        await runtime.submit_event(make_interruption(session_id, 3, "stop"))
        for _ in range(10):
            if runtime.get_current_generation(session_id) == 3:
                break
            await asyncio.sleep(0)
        assert runtime.get_current_generation(session_id) == 3
    finally:
        await runtime.stop()
