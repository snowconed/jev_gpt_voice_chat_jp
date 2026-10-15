"""Offline runtime -> real ChatCore cancellation, persistence and recovery tests."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from chat_config import ChatSettings
from chat_core import ChatCore
from chat_model import ChatGenerationCancelled
from speech_runtime import SpeechRuntime
from test_speech_runtime import Live, Speech
from voice_runtime import VoiceRuntime


ORIGINAL = "明日は図書館に行きます。"
CORRECTION = "訂正します。明後日に行きます。"
ANSWER = "明後日に図書館へ行く予定ですね。"


class CancellableResponder:
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.first_finished = threading.Event()
        self.histories = []
        self.first_cancel_event = None

    def respond(self, history, settings):
        raise AssertionError("Runtime must use the optional cancellable contract")

    def respond_cancellable(self, history, settings, cancel_event):
        self.histories.append(deepcopy(history))
        if len(self.histories) == 1:
            self.first_cancel_event = cancel_event
            self.started.set()
            try:
                if not cancel_event.wait(4):
                    raise TimeoutError("The runtime did not cancel the superseded request")
                self.cancelled.set()
                raise ChatGenerationCancelled("superseded")
            finally:
                self.first_finished.set()
        return ANSWER

    def close(self):
        if self.first_cancel_event is not None:
            self.first_cancel_event.set()


class GenerationCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def wait_for(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.005)

    async def make_runtime(self, runtime_type):
        parent = Path(__file__).resolve().parents[1] / ".runtime" / "generation_cancellation_tests"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        responder = CancellableResponder()
        directory = Path(temporary.name)
        core = ChatCore(directory, settings=ChatSettings(), responder=responder)
        sid = core.create()["id"]
        events = []

        async def send_json(event):
            events.append(event)

        async def send_bytes(_pcm):
            self.fail("Generation tests must not start speech playback")

        kwargs = {"client": Live()}
        if runtime_type is SpeechRuntime:
            kwargs["speech"] = Speech()
        runtime = runtime_type(core, sid, send_json, send_bytes, directory, **kwargs)
        runtime.ready = True
        runtime.tasks = set()
        runtime.spawn(runtime.process_inputs(), runtime.tasks, "core")

        async def cleanup():
            await runtime.close()
            responder.close()
            if responder.started.is_set():
                await self.wait_for(responder.first_finished.is_set)
            # A worker may finish saving after its coroutine was cancelled.
            await asyncio.to_thread(core.view, sid)
            await asyncio.to_thread(core.close)
            temporary.cleanup()

        self.addAsyncCleanup(cleanup)
        return runtime, core, responder, events, directory

    @staticmethod
    def stored(core, sid):
        return json.loads((core.data_dir / "sessions" / sid / "session.json").read_text(encoding="utf-8"))

    async def test_typed_correction_cancels_generation_and_preserves_both_inputs(self):
        for runtime_type in (VoiceRuntime, SpeechRuntime):
            with self.subTest(runtime=runtime_type.__name__):
                runtime, core, responder, events, _ = await self.make_runtime(runtime_type)
                self.assertTrue(await runtime.enqueue(ORIGINAL))
                await self.wait_for(responder.started.is_set)
                self.assertTrue(await runtime.enqueue(CORRECTION))
                await asyncio.wait_for(runtime.jobs.join(), 2)

                self.assertTrue(responder.cancelled.is_set())
                self.assertFalse(runtime.finished.is_set())
                self.assertIsNone(runtime.failure_message)
                self.assertFalse(any(event["type"] == "error" for event in events))
                state = await asyncio.to_thread(core.view, runtime.sid)
                self.assertIsNone(state["error"])
                self.assertEqual([row["content"] for row in state["messages"] if row["role"] == "user"],
                                 [ORIGINAL, CORRECTION])
                pending = await asyncio.to_thread(core.pending, runtime.sid)
                self.assertEqual([row["text"] for row in pending], [ANSWER])
                self.assertEqual(responder.histories, [
                    [{"role": "user", "content": ORIGINAL}],
                    [{"role": "user", "content": ORIGINAL}, {"role": "user", "content": CORRECTION}],
                ])
                requests = self.stored(core, runtime.sid)["requests"]
                self.assertEqual([row["status"] for row in requests.values()], ["interrupted", "completed"])
                logs = [json.loads(line) for line in (runtime.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(sum(row["type"] == "core_response_cancelled" for row in logs), 1)
                await runtime.close()

    async def test_runtime_close_cancels_request_before_its_normal_timeout(self):
        for runtime_type in (VoiceRuntime, SpeechRuntime):
            with self.subTest(runtime=runtime_type.__name__):
                runtime, core, responder, _events, _ = await self.make_runtime(runtime_type)
                await runtime.enqueue(ORIGINAL)
                await self.wait_for(responder.started.is_set)
                await asyncio.wait_for(runtime.close(), 1)
                await self.wait_for(responder.first_finished.is_set)
                self.assertTrue(responder.cancelled.is_set())
                state = await asyncio.wait_for(asyncio.to_thread(core.view, runtime.sid), 1)
                self.assertIsNone(state["error"])
                self.assertFalse(state["busy"])
                self.assertEqual(state["messages"][-1]["role"], "user")
                self.assertEqual(await asyncio.to_thread(core.pending, runtime.sid), [])
                request = next(iter(self.stored(core, runtime.sid)["requests"].values()))
                self.assertEqual(request["status"], "interrupted")

    async def test_substantive_speech_cancels_before_waiting_for_turn_commit(self):
        runtime, core, responder, _events, _ = await self.make_runtime(SpeechRuntime)
        await runtime.enqueue(ORIGINAL)
        await self.wait_for(responder.started.is_set)
        await runtime._accept_substantive_input(CORRECTION, time.monotonic(), 1000, caption=True)
        await asyncio.wait_for(runtime.jobs.join(), 2)
        self.assertTrue(responder.cancelled.is_set())
        self.assertEqual(runtime.input_text, CORRECTION)
        self.assertFalse(runtime.finished.is_set())
        self.assertEqual(await asyncio.to_thread(core.pending, runtime.sid), [])
        # The real turn detector will commit later; at this point the new
        # transcript remains intact and no unnecessary generation runs.
        self.assertEqual(len(responder.histories), 1)


if __name__ == "__main__":
    unittest.main()
