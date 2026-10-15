"""No-API checks for Live annotation filtering and reply preparation input."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from input_transcript import InputTranscriptFilter
from speech_runtime import SpeechRuntime
from test_speech_runtime import Core, Live, Speech, TEXT


class TranscriptFilterTests(unittest.TestCase):
    def test_only_known_tags_are_removed_across_arbitrary_fragments(self):
        projection = InputTranscriptFilter()
        pieces = ["前[gas", "p]の[s", "igh]話[bre", "ath]です"]
        result = "".join(projection.feed(part, i * .1) for i, part in enumerate(pieces))
        self.assertEqual(result, "前の話です")
        self.assertFalse(projection.has_pending)

    def test_unknown_brackets_and_regular_body_are_preserved_exactly(self):
        projection = InputTranscriptFilter()
        parts = ["本文[sc", "ene 1]や[注", "記]は残す。", "[gasping about]も残す。"]
        self.assertEqual("".join(projection.feed(p, .1 * i) for i, p in enumerate(parts)), "".join(parts))

    def test_unclosed_exact_known_name_is_removed_but_ambiguous_prefix_returns(self):
        projection = InputTranscriptFilter()
        self.assertEqual(projection.feed("[sigh", 0), "")
        self.assertEqual(projection.flush(.99), "")
        self.assertTrue(projection.has_pending)
        self.assertEqual(projection.flush(1.01), "")
        self.assertFalse(projection.has_pending)
        self.assertEqual(projection.feed("]続き", 1.02), "続き")
        self.assertEqual(projection.feed("[s", 2), "")
        self.assertEqual(projection.flush(3.01), "[s")
        self.assertEqual(projection.feed("afety]", 3.02), "afety]")

    def test_lone_open_bracket_has_bounded_wait(self):
        projection = InputTranscriptFilter()
        self.assertEqual(projection.feed("[", 0), "")
        self.assertEqual(projection.flush(1.01), "[")
        self.assertFalse(projection.has_pending)


class ReplyPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        parent = Path(__file__).resolve().parents[1] / ".runtime" / "reply_input_tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.now = 1000.0
        self.clock = patch("speech_runtime.time", SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.core, self.live, self.speech = Core(), Live(), Speech()
        self.events, self.audio = [], []

        async def send_json(event):
            self.events.append(event)

        async def send_bytes(pcm):
            self.audio.append(pcm)

        self.r = SpeechRuntime(self.core, "a" * 32, send_json, send_bytes,
                               Path(self.temp.name), client=self.live, speech=self.speech)
        self.r.ready = True
        self.r.input_last_voice = self.now - 5

    async def asyncTearDown(self):
        if self.speech.before_first:
            self.speech.before_first.set()
        await self.r.close()
        self.clock.stop()
        self.temp.cleanup()

    async def prepare(self, kind="reply"):
        self.core.messages[0]["kind"] = kind
        self.speech.before_first = asyncio.Event()
        await self.r._offer_pending(self.r.input_revision, self.r.output_generation)
        await asyncio.wait_for(self.speech.started.wait(), 1)
        return self.r.speech_task

    async def feed(self, delta):
        await self.r._accept_live_transcript({"delta": delta, "start_ms": 100, "end_ms": 200}, self.now)

    async def wait_for(self, condition):
        async with asyncio.timeout(1):
            while not condition():
                await asyncio.sleep(.005)

    def captions(self):
        return "".join(e["text"] for e in self.events if e["type"] == "input_transcript")

    def logs(self):
        return [json.loads(line) for line in (self.r.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    async def test_fragmented_ack_keeps_generation_and_waits_before_playback(self):
        task = await self.prepare()
        revision, generation = self.r.input_revision, self.r.output_generation
        await self.feed("そう")
        self.speech.before_first.set()
        await self.wait_for(lambda: self.speech.streams_closed == 1)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertFalse(any(e["type"] == "playback" for e in self.events))
        self.now += .2
        await self.feed("ですね。 [bre")
        await self.feed("ath] [sigh")
        self.now += .8
        await self.r.check_output(self.now)
        # The exact unfinished tag still has its independent one-second bound.
        self.assertEqual(self.audio, [])
        self.now += .3
        await self.r.check_output(self.now)
        await asyncio.wait_for(task, 1)
        self.assertEqual((self.r.input_revision, self.r.output_generation), (revision, generation))
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.r.reply_input_text, "")
        self.assertFalse(self.r.input_filter.has_pending)
        self.assertEqual(self.speech.calls, [TEXT])
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertTrue(any(e["type"] == "playback_end" for e in self.events))
        raw = [e["delta"] for e in self.logs() if e["type"] == "session.input_transcript.delta"]
        self.assertEqual(raw, ["そう", "ですね。 [bre", "ath] [sigh"])

    async def test_ack_then_correction_interrupts_and_promotes_exactly_once(self):
        task = await self.prepare()
        await self.feed("はい")
        self.now += .1
        await self.feed("、でも前の話ではありません。")
        await asyncio.gather(task, return_exceptions=True)
        await self.r.check_output(self.now + 3)
        expected = "はい、でも前の話ではありません。"
        self.assertEqual(self.r.input_text, expected)
        self.assertEqual(self.captions(), expected)
        self.assertEqual(self.r.input_revision, 1)
        self.assertEqual(self.r.reply_input_text, "")
        self.assertIsNone(self.r.approved)
        self.assertEqual(self.audio, [])
        self.assertEqual(len([e for e in self.events if e["type"] == "interrupt"]), 1)

    async def test_bare_ambiguous_prefix_is_promoted_after_two_seconds(self):
        task = await self.prepare()
        await self.feed("そう")
        self.now += .8
        await self.r.check_output(self.now)
        self.assertEqual(self.r.reply_input_text, "そう")
        self.assertEqual(self.r.input_text, "")
        self.now += 1.21
        await self.r.check_output(self.now)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.r.input_text, "そう")
        self.assertEqual(self.r.input_revision, 1)
        self.assertEqual(self.captions(), "そう")

    async def test_stop_is_immediate_without_waiting_for_candidate_deadline(self):
        task = await self.prepare()
        await self.feed("止めてください")
        await asyncio.gather(task, return_exceptions=True)
        self.assertIsNone(self.r.approved)
        self.assertEqual(self.r.input_text, "止めてください")
        self.assertEqual(self.r.reply_input_text, "")
        self.assertIn({"type": "interrupt", "revision": self.r.input_revision}, self.events)

    async def test_yes_answer_to_normal_question_remains_real_input(self):
        task = await self.prepare(kind="listen")
        self.speech.before_first.set()
        await asyncio.wait_for(task, 2)
        await self.r.played("approved1")
        await asyncio.gather(*list(self.r.delivery_tasks))
        await self.feed("はい")
        self.assertEqual(self.r.input_text, "はい")
        self.assertEqual(self.r.input_revision, 1)
        self.assertEqual(self.r.reply_input_text, "")
        self.assertIsNone(self.r.approved)

    async def test_unknown_bracket_text_is_promoted_verbatim(self):
        task = await self.prepare()
        await self.feed("[注記]これは本文です。")
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.r.input_text, "[注記]これは本文です。")
        self.assertEqual(self.captions(), self.r.input_text)

    async def test_ambiguous_tag_prefix_restores_real_input_after_bound(self):
        task = await self.prepare()
        await self.feed("[s")
        self.assertEqual(self.r.input_text, "")
        self.now += 1.01
        await self.r.check_output(self.now)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.r.input_text, "[s")
        self.assertFalse(self.r.input_filter.has_pending)

    async def test_close_clears_candidate_filter_and_waiting_speech(self):
        task = await self.prepare()
        await self.feed("そう[bre")
        self.speech.before_first.set()
        await self.wait_for(lambda: self.speech.streams_closed == 1)
        await self.r.close()
        self.assertTrue(task.done())
        self.assertEqual(self.r.reply_input_text, "")
        self.assertFalse(self.r.input_filter.has_pending)
        self.assertEqual(self.audio, [])
        self.assertFalse(self.r._offer_current(self.r.input_revision, self.r.output_generation))

    async def actual_delayed_punctuation(self, *, playing):
        task = await self.prepare()
        revision, generation = self.r.input_revision, self.r.output_generation
        await self.feed("そう")
        self.now += .719
        await self.feed("ですね")
        self.now += .8
        await self.r.check_output(self.now)
        self.assertEqual(self.r.reply_input_text, "")
        if playing:
            self.speech.before_first.set()
            await asyncio.wait_for(task, 1)
            self.assertIsNotNone(self.r.playing)
        self.now += 3.184
        await self.feed("。 [g")
        self.now += .188
        await self.feed("asp] [")
        self.now += .140
        await self.feed("sigh")
        self.now += 1.01
        await self.r.check_output(self.now)
        self.assertEqual(self.r.input_text, "")
        self.assertEqual((self.r.input_revision, self.r.output_generation), (revision, generation))
        self.assertFalse(any(e["type"] == "interrupt" for e in self.events))
        self.speech.before_first.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertTrue(any(e["type"] == "playback_end" for e in self.events))

    async def test_actual_delayed_punctuation_does_not_cancel_preparation(self):
        await self.actual_delayed_punctuation(playing=False)

    async def test_actual_delayed_punctuation_does_not_cancel_playback(self):
        await self.actual_delayed_punctuation(playing=True)

    async def test_pending_unknown_tag_retains_ack_prefix_when_promoted(self):
        task = await self.prepare()
        await self.feed("はい[s")
        self.now += .8
        await self.r.check_output(self.now)
        self.assertEqual(self.r.reply_input_text, "はい")
        self.assertEqual(self.r.input_text, "")
        self.now += .21
        await self.r.check_output(self.now)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.r.input_text, "はい[s")
        self.assertEqual(self.captions(), "はい[s")
        self.assertEqual(self.r.input_revision, 1)

    async def test_pending_filter_blocks_normal_turn_until_raw_prefix_restored(self):
        await self.feed("本文[s")
        self.r.native_response_boundary = self.now
        self.now += .8
        self.assertFalse(await self.r._commit_live_input(self.now))
        self.assertTrue(self.r.jobs.empty())
        self.now += .21
        await self.r.check_output(self.now)
        self.assertEqual(self.r.input_text, "本文[s")
        self.now += 4
        self.assertTrue(await self.r._commit_live_input(self.now))
        self.assertEqual(self.r.jobs.get_nowait()[0], "本文[s")

    async def test_punctuation_after_substantive_input_remains_verbatim(self):
        await self.feed("本文です")
        await self.feed("。 ")
        self.assertEqual(self.r.input_text, "本文です。 ")
        self.assertEqual(self.captions(), "本文です。 ")

    async def test_lone_bracket_flushed_through_runtime_is_preserved(self):
        await self.feed("[")
        self.assertEqual(self.r.input_text, "")
        self.now += 1.01
        await self.r.check_output(self.now)
        self.assertEqual(self.r.input_text, "[")
        self.assertEqual(self.captions(), "[")

    async def test_old_delegation_cannot_resolve_candidate_without_quiet(self):
        task = await self.prepare()
        self.r.delegation = {"offset_ms": 10000}
        self.r.live_delegation_received_at = self.now - 1
        await self.feed("はい")
        self.now += .8
        self.r.input_last_voice = self.now
        await self.r.check_output(self.now)
        self.assertEqual(self.r.reply_input_text, "はい")
        self.r.live_delegation_received_at = self.now
        await self.r.check_output(self.now)
        self.assertEqual(self.r.reply_input_text, "")
        self.speech.before_first.set()
        await asyncio.wait_for(task, 1)


if __name__ == "__main__":
    unittest.main()
