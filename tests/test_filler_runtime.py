"""Offline cached-filler races and Jev interruption semantics. No API calls."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev_runtime import JevSpeechRuntime
from chat_config import ChatSettings
from chat_core import ChatCore
from test_generation_cancellation import CancellableResponder, ANSWER
from test_jev_runtime import Judge, answer
from test_speech_runtime import Core, Live, Speech, TEXT


CLIPS = [{"id": "one", "text": "ええと、", "audio_bytes": 28800, "audio_seconds": .6},
         {"id": "two", "text": "えーっと、", "audio_bytes": 28800, "audio_seconds": .6}]
ORIGINAL = "明日は図書館に行きます。"


class PreparationJudge(Judge):
    def __init__(self):
        super().__init__()
        self.preparation_choice = "retain"

    async def classify_preparation(self, text, **metadata):
        result = await self.classify(text, **metadata)
        choice = self.preparation_choice
        result["preparation"] = {"choice": choice, "confidence": 1.,
            "probabilities": {name: float(name == choice) for name in ("retain", "revise", "uncertain")}}
        return result


class FillerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = Path(__file__).resolve().parents[1] / ".runtime" / "filler_tests"
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.now = 1000.
        self.clock = patch("jev_runtime.time", SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.core, self.client, self.speech, self.judge = Core(), Live(), Speech(), PreparationJudge()
        self.client.pending_audio = False
        self.events, self.audio = [], []

        async def send_json(event):
            self.events.append(event)

        async def send_bytes(pcm):
            self.audio.append(pcm)

        self.r = JevSpeechRuntime(self.core, "b" * 32, send_json, send_bytes,
            Path(self.temp.name), client=self.client, speech=self.speech, jev=self.judge,
            filler_clips=CLIPS)
        self.r.ready = True

    async def asyncTearDown(self):
        await self.r.close()
        self.clock.stop()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root.resolve()))
        self.temp.cleanup()

    async def wait_for(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.002)

    def logs(self):
        return [json.loads(line) for line in (self.r.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    async def feed(self, text, *, item="one", final=True, sequence=0):
        await self.r._accept_transcript({"type": "session.input_transcript.completed" if final else "session.input_transcript.delta",
            "item_id": item, "sequence": sequence, "segment_text": text})

    async def step(self, delta=.81):
        self.now += delta
        await self.r._advance(self.now)
        if self.r.judge_task:
            await asyncio.wait_for(asyncio.shield(self.r.judge_task), 2)
            await asyncio.sleep(0)

    async def begin(self, text=ORIGINAL):
        self.assertTrue(await self.r.enqueue(text))
        self.r.jobs.get_nowait()
        self.r.jobs.task_done()
        await self.wait_for(lambda: bool(self.r.filler.active and self.r.filler.active["sent_at"]))
        return self.r.filler.active

    async def ack_filler(self):
        state = self.r.filler.active
        state["sent_at"] = time.monotonic() - state["audio_seconds"]
        await self.r.played(state["id"])

    async def prepare_reply(self):
        await self.r._offer_pending(self.r.input_revision, self.r.output_generation)
        task = self.r.speech_task
        await self.wait_for(lambda: any(row["type"] == "speech_ready" for row in self.logs()))
        return task

    async def test_only_final_committed_turn_triggers_once(self):
        await self.feed(ORIGINAL, final=False)
        await self.step()
        await self.step(.3)
        self.assertFalse(any(event["type"] == "filler" for event in self.events))
        await self.feed(ORIGINAL)
        await self.step(.3)
        await self.wait_for(lambda: any(event["type"] == "filler" for event in self.events))
        self.assertEqual(self.r.jobs.get_nowait()[0], ORIGINAL)
        self.assertEqual(self.r.input_text, "")
        await self.feed(ORIGINAL)
        await self.step(1)
        self.assertEqual(sum(event["type"] == "filler" for event in self.events), 1)

    async def test_disabled_has_no_events_and_does_not_delay_reply(self):
        self.r.filler.clips = ()
        await self.r.enqueue(ORIGINAL)
        self.r.jobs.get_nowait()
        task = await self.prepare_reply()
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)
        self.assertFalse(any(event["type"].startswith("filler") for event in self.events))

    async def test_typing_is_once_per_turn_and_avoids_immediate_repeat(self):
        first = await self.begin()
        self.r.filler.schedule()
        await self.ack_filler()
        second = await self.begin("次の質問です。")
        self.assertNotEqual(first["clip_id"], second["clip_id"])
        self.assertEqual(sum(event["type"] == "filler" for event in self.events), 2)

    async def test_tts_prepares_during_filler_and_exact_reply_waits_for_ack(self):
        await self.begin()
        self.r.last_heard_text = "以前の本回答"
        context = self.r.preparation_context
        task = await self.prepare_reply()
        self.assertEqual(self.speech.calls, [TEXT])
        self.assertEqual(self.audio, [])
        self.assertIs(self.r.preparation_context, context)
        self.assertEqual(self.r.last_heard_text, "以前の本回答")
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.core.deliveries, [])
        await self.ack_filler()
        await asyncio.wait_for(task, 2)
        self.assertEqual("".join(event["text"] for event in self.events if event["type"] == "output_transcript"), TEXT)
        self.assertIsNone(self.r.preparation_context)
        self.assertEqual(self.core.deliveries, [])
        self.r.playback_segments[0]["earliest_end"] = time.monotonic()
        await self.r.played("approved1")
        await asyncio.gather(*self.r.delivery_tasks)
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])
        self.assertEqual(self.r.last_heard_text, TEXT)

    async def test_reply_ready_before_dispatch_skips_filler(self):
        await self.r.enqueue(ORIGINAL)
        message = {"id": "ready", "text": TEXT, "kind": "reply"}
        self.r.approved = message
        self.assertTrue(await self.r._wait_auxiliary_playback(message, self.r.input_revision, self.r.output_generation))
        self.assertIsNone(self.r.filler.active)
        self.assertFalse(any(event["type"] == "filler" for event in self.events))

    async def assert_explicit_interruption(self, phrase, kind):
        await self.begin()
        self.r._active_cancel = threading.Event()
        task = await self.prepare_reply()
        await self.feed(phrase)
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(self.r._active_cancel.is_set())
        self.assertIsNone(self.r.filler.active)
        self.assertEqual(self.r.waiting_for_continuation, kind)
        self.assertEqual(self.r.input_text, phrase)
        self.assertIsNone(self.r.approved)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.audio, [])
        self.assertTrue(any(event["type"] == "interrupt" for event in self.events))
        await self.step(9)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.judge.requests, [])
        self.assertEqual(sum(event["type"] == "filler" for event in self.events), 1)

    async def test_wait_stops_filler_and_prepared_reply_then_holds(self):
        await self.assert_explicit_interruption("待って", "pause")

    async def test_chotto_wait_stops_and_holds(self):
        await self.assert_explicit_interruption("ちょっと待って", "pause")

    async def test_denial_stops_filler_and_prepared_reply_then_holds(self):
        await self.assert_explicit_interruption("違う", "correction")

    async def test_iya_stops_and_holds(self):
        await self.assert_explicit_interruption("いや", "correction")

    async def test_iie_stops_and_holds(self):
        await self.assert_explicit_interruption("いいえ", "correction")

    async def test_continuation_after_wait_commits_exact_words_and_new_filler(self):
        await self.assert_explicit_interruption("待って", "pause")
        text = "待って、明後日に行きます。"
        await self.feed(text)
        await self.step(.81)
        await self.step(.26)
        await self.wait_for(lambda: self.r.filler.active is not None)
        self.assertEqual(self.r.jobs.get_nowait()[0], text)
        self.assertIsNone(self.r.waiting_for_continuation)
        self.assertEqual(self.r.preparation_context["original_text"], text)

    async def test_backchannel_uses_original_preparation_context_and_retains_reply(self):
        state = await self.begin()
        self.r._active_cancel = threading.Event()
        await self.prepare_reply()
        await self.feed("うん")
        self.assertEqual(self.r.segments["one"]["lane"], "preparation")
        await self.step(.81)
        await self.step(.26)
        self.assertEqual(self.judge.request_metadata[-1]["original_text"], ORIGINAL)
        self.assertEqual(self.judge.request_metadata[-1]["last_heard_text"], "")
        self.assertEqual(self.r.segments["one"]["lane"], "ignored_preparation")
        self.assertFalse(self.r._active_cancel.is_set())
        self.assertIs(self.r.filler.active, state)
        self.assertIsNotNone(self.r.approved)
        self.assertFalse(any(event["type"] == "interrupt" for event in self.events))
        self.assertTrue(self.r.jobs.empty())
        await self.ack_filler()
        await asyncio.wait_for(self.r.speech_task, 2)
        self.assertTrue(self.audio)

    async def test_filler_completion_keeps_unanswered_turn_context_for_backchannel(self):
        await self.begin()
        await self.ack_filler()
        await self.feed("はい")
        self.assertEqual(self.r.segments["one"]["lane"], "preparation")
        await self.step(.81)
        await self.step(.26)
        self.assertEqual(self.r.segments["one"]["lane"], "ignored_preparation")
        self.assertEqual(self.r.preparation_context["original_text"], ORIGINAL)
        self.assertEqual(self.r.last_heard_text, "")

    async def test_substantive_correction_cancels_then_commits_without_losing_text(self):
        await self.begin()
        self.r._active_cancel = threading.Event()
        self.judge.preparation_choice = "revise"
        text = "訂正します。明後日に行きます。"
        await self.feed(text)
        await self.step(.81)
        self.assertTrue(self.r._active_cancel.is_set())
        self.assertIsNone(self.r.filler.active)
        self.assertEqual(self.r.input_text, text)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], text)
        await self.wait_for(lambda: self.r.filler.active is not None)
        self.assertEqual(self.r.preparation_context["original_text"], text)

    async def test_ack_extended_to_correction_is_reconsidered(self):
        await self.begin()
        self.r._active_cancel = threading.Event()
        await self.feed("うん")
        await self.step(.81)
        await self.step(.26)
        self.judge.preparation_choice = "revise"
        await self.feed("うん、でも明後日に変えます。")
        await self.step(.81)
        self.assertTrue(self.r._active_cancel.is_set())
        self.assertEqual(self.r.input_text, "うん、でも明後日に変えます。")
        self.assertIsNone(self.r.filler.active)

    async def test_noise_and_empty_asr_do_not_stop_filler_or_cancel_generation(self):
        state = await self.begin()
        self.r._active_cancel = threading.Event()
        await self.r.microphone(b"\x00\x10" * 480)
        await self.r._accept_transcript({"type": "session.input_transcript.pending", "item_id": "noise", "sequence": 0})
        await self.feed("", item="noise")
        self.assertIs(self.r.filler.active, state)
        self.assertFalse(self.r._active_cancel.is_set())
        self.assertFalse(any(event["type"] == "interrupt" for event in self.events))

    async def test_missing_ack_stops_only_filler_and_releases_prepared_reply(self):
        self.r.filler.clips = ({**CLIPS[0], "audio_seconds": .02},)
        with patch("filler_runtime.FILLER_ACK_GRACE_SECONDS", .04):
            await self.begin()
            task = await self.prepare_reply()
            await asyncio.wait_for(task, 2)
        kinds = [event["type"] for event in self.events]
        self.assertLess(kinds.index("filler_stop"), kinds.index("playback"))
        self.assertFalse(self.r.finished.is_set())
        self.assertTrue(self.audio)
        self.assertTrue(any(row["type"] == "filler_stopped" and row["reason"] == "ack_timeout" for row in self.logs()))

    async def test_early_or_wrong_ack_does_not_release_audio(self):
        state = await self.begin()
        await self.prepare_reply()
        await self.r.played("filler_unknown")
        await self.r.played(state["id"])
        self.assertIs(self.r.filler.active, state)
        self.assertEqual(self.audio, [])

    async def test_browser_skip_releases_reply_without_error(self):
        state = await self.begin()
        task = await self.prepare_reply()
        await self.r.filler_skipped(state["id"])
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)
        self.assertFalse(self.r.finished.is_set())

    async def test_old_filler_ack_cannot_finish_new_filler(self):
        first = await self.begin()
        await self.ack_filler()
        second = await self.begin("新しい質問です。")
        await self.r.played(first["id"])
        await self.r.filler_skipped(first["id"])
        self.assertIs(self.r.filler.active, second)

    async def test_late_filler_socket_send_is_stopped_after_interrupt(self):
        gate, sending = asyncio.Event(), asyncio.Event()

        async def send_json(event):
            if event["type"] == "filler":
                sending.set()
                await gate.wait()
            self.events.append(event)

        self.r.send_json = send_json
        await self.r.enqueue(ORIGINAL)
        self.r.jobs.get_nowait()
        await asyncio.wait_for(sending.wait(), 2)
        stop = asyncio.create_task(self.feed("待って"))
        await self.wait_for(lambda: any(event["type"] == "interrupt" for event in self.events))
        gate.set()
        await asyncio.wait_for(stop, 2)
        kinds = [event["type"] for event in self.events]
        self.assertLess(kinds.index("filler"), kinds.index("filler_stop"))
        stopped_revision = next(event["revision"] for event in self.events if event["type"] == "interrupt")
        filler_revision = next(event["revision"] for event in self.events if event["type"] == "filler")
        self.assertGreaterEqual(stopped_revision, filler_revision)
        self.assertIsNone(self.r.filler.active)
        self.assertEqual(self.r.waiting_for_continuation, "pause")

    async def test_failed_generation_stops_filler_but_keeps_session_open(self):
        class FailedCore(Core):
            def __init__(self):
                super().__init__()
                self.messages = []

            def submit(self, sid, text, request_id):
                return {"id": sid, "messages": [{"role": "user", "content": text}], "error": "temporary failure"}

        self.r.core = FailedCore()
        await self.r.enqueue(ORIGINAL)
        await self.wait_for(lambda: self.r.filler.active and self.r.filler.active["sent_at"])
        self.r.tasks = set()
        self.r.spawn(self.r.process_inputs(), self.r.tasks, "core")
        await asyncio.wait_for(self.r.jobs.join(), 2)
        await self.wait_for(lambda: self.r.filler.active is None)
        self.assertFalse(self.r.finished.is_set())
        self.assertTrue(any(event["type"] == "filler_stop" and event["reason"] == "no_reply" for event in self.events))

    async def test_old_generation_cleanup_does_not_stop_new_filler(self):
        old = await self.begin()
        await self.ack_filler()
        new = await self.begin("新しい質問です。")
        await self.r._input_processing_finished(old["revision"], False)
        self.assertIs(self.r.filler.active, new)

    async def test_partly_sent_filler_failure_is_stopped_before_reply(self):
        async def send_json(event):
            self.events.append(event)
            if event["type"] == "filler":
                raise OSError("simulated send failure after dispatch")

        self.r.send_json = send_json
        await self.r.enqueue(ORIGINAL)
        self.r.jobs.get_nowait()
        task = await self.prepare_reply()
        await asyncio.wait_for(task, 2)
        kinds = [event["type"] for event in self.events]
        self.assertLess(kinds.index("filler_stop"), kinds.index("playback"))
        self.assertFalse(self.r.finished.is_set())
        self.assertEqual("".join(event["text"] for event in self.events if event["type"] == "output_transcript"), TEXT)

    async def test_failed_best_effort_stop_does_not_fail_session(self):
        async def send_json(event):
            self.events.append(event)
            if event["type"] in ("filler", "filler_stop"):
                raise OSError("simulated optional send failure")

        self.r.send_json = send_json
        await self.r.enqueue(ORIGINAL)
        self.r.jobs.get_nowait()
        task = await self.prepare_reply()
        await asyncio.wait_for(task, 2)
        self.assertFalse(self.r.finished.is_set())
        self.assertTrue(self.audio)
        self.assertIsNone(self.r.filler.active)
        self.assertTrue(any(row["type"] == "filler_stop_failed" for row in self.logs()))

    async def test_automatic_commit_rechecks_voice_after_filler_interrupt_await(self):
        await self.begin()

        async def send_json(event):
            self.events.append(event)
            if event["type"] == "interrupt":
                self.r.input_last_voice += 1
                self.r.input_revision += 1
                self.r.input_text = "後から届いた続き"

        self.r.send_json = send_json
        self.assertFalse(await self.r.enqueue("古い入力", "jev_complete"))
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.r.input_text, "後から届いた続き")

    async def assert_real_core_cancellation_and_recovery(self, phrase, kind):
        responder = CancellableResponder()
        core = ChatCore(Path(self.temp.name), settings=ChatSettings(), responder=responder)
        sid = core.create()["id"]
        self.r.core, self.r.sid = core, sid
        self.r.tasks = set()
        self.r.spawn(self.r.process_inputs(), self.r.tasks, "core")
        try:
            await self.r.enqueue(ORIGINAL)
            await self.wait_for(lambda: responder.started.is_set() and self.r.filler.active)
            filler_id = self.r.filler.active["id"]
            await self.feed(phrase)
            await asyncio.wait_for(self.r.jobs.join(), 3)
            self.assertTrue(responder.cancelled.is_set())
            self.assertEqual(self.r.waiting_for_continuation, kind)
            self.assertIsNone(self.r.filler.active)
            self.assertEqual(await asyncio.to_thread(core.pending, sid), [])
            self.assertFalse(self.r.finished.is_set())
            self.assertEqual(self.audio, [])
            continuation = phrase + "、明後日に行きます。"
            await self.feed(continuation)
            await self.step(.81)
            await self.step(.26)
            await asyncio.wait_for(self.r.jobs.join(), 3)
            await self.wait_for(lambda: bool(self.speech.calls))
            self.assertEqual(self.speech.calls, [ANSWER])
            self.assertNotEqual(self.r.filler.active["id"], filler_id)
            task = self.r.speech_task
            await self.ack_filler()
            await asyncio.wait_for(task, 2)
            state = await asyncio.to_thread(core.view, sid)
            self.assertEqual([row["content"] for row in state["messages"] if row["role"] == "user"],
                             [ORIGINAL, continuation])
            self.assertFalse(any(row.get("kind") == "filler" for row in state["messages"]))
            self.assertEqual(responder.histories[-1], [
                {"role": "user", "content": ORIGINAL}, {"role": "user", "content": continuation}])
            self.assertEqual("".join(event["text"] for event in self.events if event["type"] == "output_transcript"), ANSWER)
        finally:
            await self.r.close()
            await asyncio.to_thread(core.close)

    async def test_wait_during_filler_cancels_real_core_and_recovers(self):
        await self.assert_real_core_cancellation_and_recovery("待って", "pause")

    async def test_denial_during_filler_cancels_real_core_and_recovers(self):
        await self.assert_real_core_cancellation_and_recovery("違う", "correction")

    async def test_close_stops_filler_and_cleans_owned_tasks(self):
        await self.begin()
        await self.r.close()
        self.assertIsNone(self.r.filler.active)
        self.assertFalse(self.r.filler.tasks)
        self.assertTrue(any(event["type"] == "filler_stop" for event in self.events))
        self.assertEqual(self.core.deliveries, [])


if __name__ == "__main__":
    unittest.main()
