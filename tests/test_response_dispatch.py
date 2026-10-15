"""Immediate post-generation dispatch uses the same guards as periodic polls."""
import asyncio
import time
import unittest

from jev_runtime import JevSpeechRuntime
from speech_runtime import SpeechRuntime
from voice_runtime import VoiceRuntime
import test_runtime_recovery as recovery


class ResponseDispatchTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = recovery.RuntimeRecoveryTests.asyncSetUp
    asyncTearDown = recovery.RuntimeRecoveryTests.asyncTearDown
    make_runtime = recovery.RuntimeRecoveryTests.make_runtime
    worker = recovery.RuntimeRecoveryTests.worker

    async def wait_for(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(.002)

    @staticmethod
    def session_barrier():
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(event):
            if event["type"] == "session" and not entered.is_set():
                entered.set()
                await release.wait()

        return entered, release, send

    async def feed(self, runtime, text):
        await runtime._accept_transcript({"type": "session.input_transcript.completed",
            "item_id": "addition", "sequence": 0, "segment_text": text})

    async def test_success_starts_exact_tts_without_starting_periodic_tick(self):
        for kind in (SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                runtime, core, _responder, events = self.make_runtime(kind)
                await runtime.enqueue("今日は散歩へ行きました。")
                self.worker(runtime)
                await asyncio.wait_for(runtime.speech.started.wait(), 1)
                self.assertFalse(runtime.busy)
                self.assertEqual(runtime.speech.calls,
                                 [core.view(runtime.sid)["messages"][-1]["content"]])
                self.assertEqual(len([e for e in events if e["type"] == "session"]), 1)
                self.assertFalse(runtime.finished.is_set())

    async def test_periodic_poll_and_completion_dispatch_reserve_speech_only_once(self):
        for kind in (SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                entered, release, send = self.session_barrier()
                runtime, _core, _responder, _events = self.make_runtime(kind, send)
                await runtime.enqueue("今日は散歩へ行きました。")
                self.worker(runtime)
                await asyncio.wait_for(entered.wait(), 1)
                await runtime._refresh_state()  # A periodic tick wins the race.
                await asyncio.wait_for(runtime.speech.started.wait(), 1)
                release.set()
                await asyncio.sleep(.01)
                self.assertEqual(len(runtime.speech.calls), 1)
                self.assertEqual(len(runtime.speech_tasks), 1)

    async def test_new_input_during_immediate_notification_prevents_stale_synthesis(self):
        for kind in (SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                entered, release, send = self.session_barrier()
                runtime, core, responder, _events = self.make_runtime(kind, send)
                await runtime.enqueue("明日は図書館へ行きます。")
                self.worker(runtime)
                await asyncio.wait_for(entered.wait(), 1)
                old_id = core.view(runtime.sid)["messages"][-1]["id"]
                await runtime.enqueue("訂正します。明後日です。")
                release.set()
                await asyncio.wait_for(runtime.speech.started.wait(), 1)
                self.assertEqual(len(runtime.speech.calls), 1)
                self.assertEqual(len(responder.histories), 2)
                self.assertNotEqual(runtime.approved["id"], old_id)
                old = next(m for m in core.view(runtime.sid)["messages"] if m["id"] == old_id)
                self.assertEqual(old["delivery"]["status"], "cancelled")

    async def test_session_finish_during_immediate_notification_does_not_start_tts(self):
        for kind in (VoiceRuntime, SpeechRuntime, JevSpeechRuntime):
            with self.subTest(runtime=kind.__name__):
                entered, release, send = self.session_barrier()
                runtime, _core, _responder, events = self.make_runtime(kind, send)
                await runtime.enqueue("今日は散歩へ行きました。")
                worker = self.worker(runtime)
                await asyncio.wait_for(entered.wait(), 1)
                runtime.finished.set()
                release.set()
                await asyncio.sleep(.01)
                if kind is not VoiceRuntime:
                    self.assertEqual(runtime.speech.calls, [])
                self.assertFalse(any(e["type"] == "test_approval" for e in events))
                self.assertIsNone(runtime.approved)
                self.assertFalse(worker.done())

    async def test_base_voice_finish_during_approval_revokes_it(self):
        runtime, core, _responder, events = self.make_runtime(VoiceRuntime)
        entered, release = asyncio.Event(), asyncio.Event()

        async def approve(_text):
            entered.set()
            await release.wait()

        runtime.client.approve = approve
        await runtime.enqueue("今日は散歩へ行きました。")
        self.worker(runtime)
        await asyncio.wait_for(entered.wait(), 1)
        runtime.finished.set()
        release.set()
        await self.wait_for(lambda: runtime.approved is None)
        await asyncio.gather(*list(runtime.delivery_tasks), *list(runtime.control_tasks))
        self.assertTrue(any(e["type"] == "interrupt" for e in events))
        self.assertEqual(core.pending(runtime.sid), [])

    async def test_wait_and_denial_during_immediate_notification_keep_the_floor(self):
        for command in ("待って。", "違います。"):
            with self.subTest(command=command):
                entered, release, send = self.session_barrier()
                runtime, _core, _responder, _events = self.make_runtime(JevSpeechRuntime, send)
                await runtime.enqueue("今日は散歩へ行きました。")
                self.worker(runtime)
                await asyncio.wait_for(entered.wait(), 1)
                await self.feed(runtime, command)
                release.set()
                await asyncio.sleep(.01)
                self.assertEqual(runtime.speech.calls, [])
                self.assertEqual(runtime.input_text, command)
                self.assertIsNotNone(runtime.waiting_for_continuation)
                self.assertTrue(runtime.jobs.empty())

    async def test_preparation_ack_holds_playback_until_existing_decision_retains_it(self):
        entered, release, send = self.session_barrier()
        runtime, _core, _responder, events = self.make_runtime(JevSpeechRuntime, send)
        await runtime.enqueue("今日は散歩へ行きました。")
        self.worker(runtime)
        await asyncio.wait_for(entered.wait(), 1)
        await self.feed(runtime, "そうですね。")
        release.set()
        await asyncio.wait_for(runtime.speech.started.wait(), 1)
        runtime.speech.before_first.set()
        await self.wait_for(lambda: runtime.noise_wait_started > 0)
        self.assertFalse(any(e["type"] == "playback" for e in events))
        self.assertEqual(runtime._text("preparation"), "そうですね。")
        runtime._retain_preparation()  # Exercise the existing retained-ack path.
        runtime.input_last_voice = time.monotonic() - 1
        await self.wait_for(lambda: any(e["type"] == "playback" for e in events))
        self.assertEqual(len(runtime.speech.calls), 1)

    async def test_preparation_correction_revokes_ready_audio_before_playback(self):
        entered, release, send = self.session_barrier()
        runtime, _core, _responder, events = self.make_runtime(JevSpeechRuntime, send)
        await runtime.enqueue("明日は図書館へ行きます。")
        self.worker(runtime)
        await asyncio.wait_for(entered.wait(), 1)
        await self.feed(runtime, "正しくは明後日の予定です。")
        release.set()
        await asyncio.wait_for(runtime.speech.started.wait(), 1)
        runtime.speech.before_first.set()
        await self.wait_for(lambda: runtime.noise_wait_started > 0)
        await runtime._promote_preparation("jev_new_content")
        self.assertFalse(any(e["type"] == "playback" for e in events))
        self.assertEqual(runtime.input_text, "正しくは明後日の予定です。")
        self.assertIsNone(runtime.approved)


if __name__ == "__main__":
    unittest.main()
