"""Real core/runtime regressions for generation failures and startup races."""
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest

from chat_config import ChatSettings
from chat_core import ChatCore
from jev_runtime import JevSpeechRuntime
from speech_runtime import SpeechRuntime
from test_jev_runtime import Judge
from test_speech_runtime import Live, Speech
from voice_runtime import VoiceRuntime


class Responder:
    def __init__(self):
        self.histories = []
        self.error = None
        self.started = threading.Event()
        self.release = None

    def respond(self, history, settings):
        self.histories.append(deepcopy(history))
        self.started.set()
        if self.release is not None and not self.release.wait(3):
            raise TimeoutError("test release missing")
        if self.error:
            raise self.error
        return "次のお話を受け取りました。"


class RuntimeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        parent = Path(__file__).resolve().parents[1] / ".runtime" / "runtime_recovery_tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.runtimes, self.workers, self.cores, self.releases = [], [], [], []

    async def asyncTearDown(self):
        for release in self.releases:
            release.set()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        for runtime in self.runtimes:
            await runtime.close()
        for core in self.cores:
            await asyncio.to_thread(core.close)
        self.temp.cleanup()

    def make_runtime(self, kind, send=None):
        directory = Path(self.temp.name) / str(len(self.cores))
        responder = Responder()
        core = ChatCore(directory, settings=ChatSettings(), responder=responder)
        state = core.create()
        core.delivery(state["id"], state["messages"][0]["id"], "completed")
        events = []

        async def send_json(event):
            events.append(event)
            if send:
                await send(event)

        async def send_bytes(pcm):
            events.append({"type": "test_audio", "bytes": len(pcm)})

        options = {"client": Live()}
        if kind is not VoiceRuntime:
            options["speech"] = Speech()
            # Generation tests may now dispatch immediately after success.
            # Hold actual synthesis until an explicit playback test releases it.
            options["speech"].before_first = asyncio.Event()
        else:
            async def approve(text):
                events.append({"type": "test_approval", "text": text})
            options["client"].approve = approve
        if kind is JevSpeechRuntime:
            options["jev"] = Judge()
        runtime = kind(core, state["id"], send_json, send_bytes, directory, **options)
        runtime.ready = True
        self.cores.append(core)
        self.runtimes.append(runtime)
        return runtime, core, responder, events

    def worker(self, runtime):
        task = asyncio.create_task(runtime.process_inputs())
        self.workers.append(task)
        return task

    async def test_failed_generation_recovers_on_next_input_without_losing_history(self):
        for kind in (VoiceRuntime, SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                runtime, core, responder, events = self.make_runtime(kind)
                responder.error = RuntimeError("private provider details")
                await runtime.enqueue("今日はローソンへ行きました。")
                worker = self.worker(runtime)
                await asyncio.wait_for(runtime.jobs.join(), 2)
                self.assertFalse(worker.done())
                self.assertFalse(runtime.finished.is_set())
                self.assertIsNone(runtime.failure_message)
                self.assertEqual(core.pending(runtime.sid), [])
                self.assertFalse(any(e["type"] in {"test_audio", "test_approval"} for e in events))
                self.assertTrue(any("このまま続けて" in e.get("text", "") for e in events))
                self.assertNotIn("private", str(events))

                responder.error = None
                await runtime.enqueue("おにぎりを買ったんです。")
                await asyncio.wait_for(runtime.jobs.join(), 2)
                state = core.view(runtime.sid)
                self.assertIsNone(state["error"])
                self.assertEqual([m["content"] for m in state["messages"] if m["role"] == "user"],
                                 ["今日はローソンへ行きました。", "おにぎりを買ったんです。"])
                self.assertEqual(responder.histories[-1], [
                    {"role": "assistant", "content": core.settings.greeting},
                    {"role": "user", "content": "今日はローソンへ行きました。"},
                    {"role": "user", "content": "おにぎりを買ったんです。"},
                ])
                self.assertEqual(len(core.pending(runtime.sid)), 1)
                self.assertFalse(any(e["type"] == "error" for e in events))

    async def test_input_before_request_starts_skips_obsolete_api_call_but_preserves_text(self):
        for kind in (VoiceRuntime, SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                entered, release = asyncio.Event(), asyncio.Event()

                async def send(event):
                    if event["type"] == "status" and not entered.is_set():
                        entered.set()
                        await release.wait()

                runtime, core, responder, _ = self.make_runtime(kind, send=send)
                await runtime.enqueue("明日は図書館へ行きます。")
                self.worker(runtime)
                await asyncio.wait_for(entered.wait(), 1)
                await runtime.enqueue("訂正します。明後日の予定です。")
                release.set()
                await asyncio.wait_for(runtime.jobs.join(), 2)
                self.assertEqual(len(responder.histories), 1)
                self.assertEqual([m["content"] for m in responder.histories[0] if m["role"] == "user"],
                                 ["明日は図書館へ行きます。", "訂正します。明後日の予定です。"])
                self.assertEqual(len(core.pending(runtime.sid)), 1)

    async def test_periodic_refresh_and_offer_never_wait_for_model_lock(self):
        for kind in (VoiceRuntime, SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                runtime, core, responder, events = self.make_runtime(kind)
                responder.release = threading.Event()
                self.releases.append(responder.release)
                generation = asyncio.create_task(asyncio.to_thread(core.submit,
                    runtime.sid, "処理中の入力です。", "request_lock_test"))
                try:
                    self.assertTrue(await asyncio.to_thread(responder.started.wait, 1))
                    # Simulate a request taking the lock after the tick's busy
                    # guard. Polling must defer, rather than block turn detection.
                    self.assertFalse(runtime.busy)
                    await asyncio.wait_for(runtime._refresh_state(), .25)
                    await asyncio.wait_for(runtime._offer_pending(0, 0), .25)
                    self.assertEqual(events, [])
                    self.assertIsNone(runtime.approved)
                    if kind is SpeechRuntime:
                        runtime.live_output_text = "はい"
                        await asyncio.wait_for(runtime.check_output(time.monotonic()), .25)
                        self.assertIsNone(runtime.approved)
                finally:
                    responder.release.set()
                    await asyncio.wait_for(generation, 2)


if __name__ == "__main__":
    unittest.main()
