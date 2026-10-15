"""Offline Live interruption intent, retained continuations, and noise timing."""
import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from speech_runtime import SpeechRuntime
from test_speech_runtime import Core, Live, Speech


class LiveInterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = Path(__file__).resolve().parents[1] / ".runtime" / "live_interruption_tests"
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.now = 1000.0
        self.clock = SimpleNamespace(monotonic=lambda: self.now)
        self.patches = [patch("speech_runtime.time", self.clock), patch("voice_runtime.time", self.clock)]
        for item in self.patches:
            item.start()
        self.r = None
        self.counter = 0
        await self.new_runtime()

    async def asyncTearDown(self):
        try:
            if self.r:
                await self.r.close()
        finally:
            for item in reversed(self.patches):
                item.stop()
            self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root.resolve()))
            self.temp.cleanup()

    async def new_runtime(self):
        if self.r:
            await self.r.close()
        self.counter += 1
        self.core, self.live, self.speech = Core(), Live(), Speech()
        self.events, self.audio = [], []

        async def send_json(event):
            self.events.append(event)

        async def send_bytes(pcm):
            self.audio.append(pcm)

        self.r = SpeechRuntime(self.core, "a" * 32, send_json, send_bytes,
                               Path(self.temp.name) / str(self.counter), client=self.live, speech=self.speech)
        self.r.ready = True
        self.r.input_last_voice = self.now - 5
        self.offset = 0

    async def phase(self, phase):
        if phase == "generation":
            self.core.messages.clear()
            self.r.busy = True
            self.generation_cancel = threading.Event()
            self.r._active_cancel = self.generation_cancel
            return
        if phase == "preparation":
            self.speech.before_first = asyncio.Event()
        await self.r._offer_pending(self.r.input_revision, self.r.output_generation)
        speech_task = self.r.speech_task
        await asyncio.wait_for(self.speech.started.wait(), 1)
        if phase == "playback":
            await asyncio.wait_for(speech_task, 1)
            self.assertIsNotNone(self.r.playing)
        else:
            self.assertIsNotNone(self.r.approved)
            self.assertFalse(self.audio)

    async def feed(self, delta):
        self.offset += 200
        await self.r._accept_live_transcript({"type": "session.input_transcript.delta", "delta": delta,
                                             "start_ms": self.offset - 200, "end_ms": self.offset}, self.now)

    async def settle(self):
        await asyncio.gather(*list(self.r.speech_tasks), return_exceptions=True)
        await asyncio.gather(*list(self.r.delivery_tasks), return_exceptions=True)

    def native_boundary(self):
        self.r.delegation = {"offset_ms": self.offset + 1000}
        self.r.live_delegation_received_at = self.now
        self.r.native_response_boundary = self.now

    async def assert_bare_interruptions_wait(self, phase):
        for phrase, expected in (("待って。", "pause"), ("違う。", "correction"), ("いや。", "correction")):
            with self.subTest(phase=phase, phrase=phrase):
                await self.new_runtime()
                await self.phase(phase)
                await self.feed(phrase)
                await self.settle()
                self.assertEqual(self.r.input_text, phrase)
                self.assertEqual(self.r.waiting_for_continuation, expected)
                self.assertIsNone(self.r.playing)
                self.assertIsNone(self.r.approved)
                if phase == "generation":
                    self.assertTrue(self.generation_cancel.is_set())
                    self.r.busy = False
                else:
                    self.assertTrue(any(event["type"] == "interrupt" for event in self.events))
                    expected_delivery = "interrupted" if phase == "playback" else "cancelled"
                    self.assertEqual(self.core.deliveries, [("approved1", expected_delivery, "")])
                self.now += 30
                self.native_boundary()
                self.assertFalse(await self.r._commit_live_input(self.now))
                self.assertTrue(self.r.jobs.empty(), "a bare stop/denial cannot start a replacement answer")
                self.assertEqual(self.r.input_text, phrase, "the stop/denial remains available for its continuation")

    async def test_bare_wait_and_denials_interrupt_playback_without_replacement_reply(self):
        await self.assert_bare_interruptions_wait("playback")

    async def test_bare_wait_and_denials_cancel_preparation_without_replacement_reply(self):
        await self.assert_bare_interruptions_wait("preparation")

    async def test_bare_wait_and_denials_cancel_generation_without_replacement_reply(self):
        await self.assert_bare_interruptions_wait("generation")

    async def test_spoken_continuation_releases_wait_and_commits_every_input_character(self):
        await self.phase("playback")
        await self.feed("待って。")
        await self.settle()
        self.now += 10
        self.native_boundary()
        self.assertFalse(await self.r._commit_live_input(self.now))
        await self.feed("明日の話です。")
        self.assertFalse(self.r.waiting_for_continuation)
        self.assertEqual(self.r.input_text, "待って。明日の話です。")
        self.now += 1
        self.native_boundary()
        self.assertTrue(await self.r._commit_live_input(self.now))
        text, _request_id, _revision = self.r.jobs.get_nowait()
        self.assertEqual(text, "待って。明日の話です。")
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.r.input_text, "")
        self.r.jobs.task_done()

    async def test_repeated_stop_and_denial_phrases_do_not_release_wait(self):
        await self.phase("playback")
        await self.feed("いや、")
        await self.feed("違う。")
        await self.feed("待って。")
        await self.settle()
        self.now += 20
        self.native_boundary()
        self.assertEqual(self.r.waiting_for_continuation, "pause")
        self.assertFalse(await self.r._commit_live_input(self.now))
        self.assertEqual(self.r.input_text, "いや、違う。待って。")
        self.assertTrue(self.r.jobs.empty())

    async def test_no_is_a_normal_answer_when_assistant_is_not_speaking(self):
        self.core.messages.clear()
        await self.feed("いいえ。")
        self.assertFalse(self.r.waiting_for_continuation)
        self.now += 1
        self.native_boundary()
        self.assertTrue(await self.r._commit_live_input(self.now))
        text, _request_id, _revision = self.r.jobs.get_nowait()
        self.assertEqual(text, "いいえ。")
        self.r.jobs.task_done()
        self.assertFalse(any(event["type"] == "interrupt" for event in self.events))

    async def test_acknowledgments_during_playback_are_held_then_cleared_without_interrupting(self):
        for phrase in ("うん。", "なるほど。"):
            with self.subTest(phrase=phrase):
                await self.new_runtime()
                await self.phase("playback")
                current = self.r.playing
                await self.feed(phrase)
                self.assertIs(self.r.playing, current)
                self.assertEqual(self.r.reply_input_text, phrase)
                self.assertEqual(self.r.input_text, "")
                self.now += 1
                await self.r._resolve_preparation_input(self.now)
                self.assertEqual(self.r.reply_input_text, "")
                self.assertIs(self.r.playing, current)
                self.assertTrue(self.r.jobs.empty())
                self.assertEqual(self.core.deliveries, [])
                self.assertFalse(any(event["type"] == "interrupt" for event in self.events))

    async def test_ack_prefix_followed_by_a_correction_stops_and_preserves_full_input(self):
        await self.phase("playback")
        await self.feed("はい")
        self.assertIsNotNone(self.r.playing)
        self.assertEqual(self.r.reply_input_text, "はい")
        self.now += .1
        await self.feed("、でも違います。")
        await self.settle()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.reply_input_text, "")
        self.assertEqual(self.r.input_text, "はい、でも違います。")
        self.assertEqual(self.core.deliveries, [("approved1", "interrupted", "")])
        self.assertEqual("".join(event["text"] for event in self.events if event["type"] == "input_transcript"),
                         "はい、でも違います。")

    async def test_yes_overlapping_the_end_of_a_question_is_preserved_as_answer(self):
        await self.phase("playback")
        await self.feed("はい。")
        self.now += 1
        await self.r.played("approved1")
        await self.settle()
        await self.r._resolve_preparation_input(self.now)
        self.assertEqual(self.r.input_text, "はい。")
        self.assertFalse(self.r.waiting_for_continuation)
        self.now += 1
        self.native_boundary()
        self.assertTrue(await self.r._commit_live_input(self.now))
        text, _, _ = self.r.jobs.get_nowait()
        self.assertEqual(text, "はい。")
        self.r.jobs.task_done()

    async def test_continuation_commits_during_noise_after_native_boundary(self):
        await self.phase("playback")
        await self.feed("違う。")
        await self.settle()
        self.now += 2
        await self.feed("明日の話です。")
        voice_at = self.r.input_last_voice
        pcm = b"\x00\x10" * 4800
        for _index in range(15):
            self.now += .1
            await self.r.microphone(pcm)
        self.assertEqual(self.r.input_last_voice, voice_at, "noise must not overwrite recognized-speech timing")
        self.assertEqual(self.r.input_last_sound, self.now)
        self.native_boundary()
        self.assertTrue(await self.r._commit_live_input(self.now))
        text, _request_id, _revision = self.r.jobs.get_nowait()
        self.assertEqual(text, "違う。明日の話です。")
        self.r.jobs.task_done()

    async def test_typed_continuation_combines_waiting_voice_text_once(self):
        await self.phase("playback")
        await self.feed("待って。")
        await self.settle()
        accepted = await self.r.enqueue("明日の話です。", source="typed")
        self.assertIsNot(accepted, False)
        self.assertFalse(self.r.waiting_for_continuation)
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.r.jobs.qsize(), 1)
        text, _request_id, _revision = self.r.jobs.get_nowait()
        self.assertIn("待って。", text)
        self.assertIn("明日の話です。", text)
        self.assertEqual(text.count("待って。"), 1)
        self.assertEqual(text.count("明日の話です。"), 1)
        self.assertLess(text.index("待って。"), text.index("明日の話です。"))
        self.r.jobs.task_done()
        self.now += 20
        self.native_boundary()
        self.assertFalse(await self.r._commit_live_input(self.now))
        self.assertTrue(self.r.jobs.empty())


if __name__ == "__main__":
    unittest.main()
