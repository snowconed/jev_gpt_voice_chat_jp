"""Offline Jev turn/acknowledgement/ASR ordering regressions. No model APIs."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chat_config import ChatSettings
from chat_core import ChatCore
from jev_runtime import JevSpeechRuntime
from test_generation_cancellation import CancellableResponder, ORIGINAL, ANSWER
from test_speech_runtime import Core, Live, Speech, TEXT

HERE = Path(__file__).resolve().parents[1]


def answer(turn="complete", interruption="interrupt", confidence=1.0):
    return {"turn": {"choice": turn, "confidence": 1.0,
                     "probabilities": {k: float(k == turn) for k in ("complete", "short", "long")}},
            "interruption": {"choice": interruption, "confidence": confidence,
                     "probabilities": {k: float(k == interruption) for k in ("interrupt", "backchannel", "uncertain")}},
            "request_ms": 1.0}


class Judge:
    model = "offline-jev"

    def __init__(self):
        self.calls = []
        self.requests = []
        self.request_metadata = []
        self.result = answer()
        self.error = None
        self.gate = None
        self.started = asyncio.Event()
        self.closes = 0

    async def classify(self, text, assistant_text="", speaking=False, **metadata):
        self.requests.append((text, assistant_text, speaking))
        self.request_metadata.append(dict(metadata))
        self.started.set()
        if self.gate:
            await self.gate.wait()
        if self.error:
            raise self.error
        return deepcopy(self.result)

    async def classify_preparation(self, text, **metadata):
        result = await self.classify(text, **metadata)
        result["preparation"] = {"choice": "revise", "confidence": 1.0,
                                 "probabilities": {"retain": 0., "revise": 1., "uncertain": 0.}}
        return result

    async def close(self):
        self.closes += 1


class JevRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = (HERE / ".runtime" / "jev_tests").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.now = 1000.0
        self.clock = patch("jev_runtime.time", SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.core, self.client, self.speech, self.judge = Core(), Live(), Speech(), Judge()
        self.client.pending_audio = False
        self.events, self.audio = [], []

        async def send_json(event):
            self.events.append(event)

        async def send_bytes(pcm):
            self.audio.append(pcm)

        self.r = JevSpeechRuntime(self.core, "a" * 32, send_json, send_bytes, Path(self.temp.name),
                                  client=self.client, speech=self.speech, jev=self.judge)
        self.r.ready = True
        self.r.last_activity = self.now

    async def asyncTearDown(self):
        if self.judge.gate:
            self.judge.gate.set()
        await self.r.close()
        self.clock.stop()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root))
        self.temp.cleanup()

    async def feed(self, text, *, final=True, item="one", sequence=0):
        await self.r._accept_transcript({"type": "session.input_transcript.completed" if final else "session.input_transcript.delta",
                                         "item_id": item, "sequence": sequence,
                                         "segment_text": text, "text": text, "delta": text})

    def playing(self):
        self.r.playing = {"id": "approved1", "kind": "reply", "text": TEXT}
        self.r.playing_started = time.monotonic() - 20
        self.r.playing_seconds = 10

    async def step(self, seconds=0, *, settle=True):
        self.now += seconds
        await self.r._advance(self.now)
        if settle and self.r.judge_task:
            await asyncio.wait_for(asyncio.shield(self.r.judge_task), 2)
            await asyncio.sleep(0)

    async def deliveries(self):
        await asyncio.gather(*list(self.r.delivery_tasks))

    def logs(self):
        return [json.loads(line) for line in (self.r.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    async def test_normal_complete_has_grace_and_exact_single_commit(self):
        self.r.input_last_voice = self.now
        await self.feed("今日はうれしかったです。")
        await self.step(.19)
        self.assertEqual(self.judge.requests, [])
        await self.step(.02)
        self.assertTrue(self.r.jobs.empty())
        await self.step(.24)
        self.assertTrue(self.r.jobs.empty())
        await self.step(.02)
        self.assertTrue(self.r.jobs.empty())
        await self.step(.34)
        self.assertEqual(self.r.jobs.get_nowait()[0], "今日はうれしかったです。")
        await self.feed("今日はうれしかったです。")
        await self.step(1)
        self.assertTrue(self.r.jobs.empty())

    async def test_complete_waits_for_final_asr_replacing_partial(self):
        await self.feed("今日は", final=False)
        await self.step(1)
        await self.step(.3)
        self.assertTrue(self.r.jobs.empty())
        await self.feed("今日は良い日でした。", final=True)
        await self.step(.21)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "今日は良い日でした。")

    async def test_normal_judge_sends_current_partial_timing_without_snapshot_churn(self):
        self.r.input_last_voice = self.now - .9
        await self.feed("今日は", final=False)
        snapshot = self.r._snapshot("normal")
        await self.step(.25)
        self.assertEqual(self.judge.request_metadata, [{"transcript_final": False,
                         "silence_ms": 1150.0, "text_stable_ms": 250.0}])
        self.assertEqual(self.r._snapshot("normal"), snapshot)
        await self.step(.3)
        self.assertEqual(len(self.judge.requests), 1)
        self.assertTrue(self.r.jobs.empty())
        starts = [event for event in self.logs() if event["type"] == "jev_judge_started"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["input_chars"], 3)
        self.assertFalse(starts[0]["transcript_final"])
        self.assertFalse({"text", "segment_text", "assistant_text"} & starts[0].keys())
        result = next(event for event in self.logs() if event["type"] == "jev_classification")
        self.assertEqual(result["request_sequence"], starts[0]["request_sequence"])

    async def test_normal_final_judge_sends_elapsed_metadata_and_keeps_yes_answer(self):
        self.r.input_last_voice = self.now
        await self.feed("はい")
        await self.step(.81)
        self.assertEqual(self.judge.requests, [("はい", "", False)])
        self.assertEqual(self.judge.request_metadata, [{"transcript_final": True,
                         "silence_ms": 810.0, "text_stable_ms": 810.0}])
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "はい")

    async def test_request_elapsed_metadata_is_nonnegative(self):
        await self.feed("本文です")
        self.r.input_last_voice = self.now + 1
        self.r.input_last_change = self.now + 1
        await self.r._judge(self.r._snapshot("normal"))
        self.assertEqual(self.judge.request_metadata, [{"transcript_final": True,
                         "silence_ms": 0.0, "text_stable_ms": 0.0}])

    async def test_candidate_receives_finality_without_elapsed_time(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("うん", final=False)
        await self.step(.21)
        self.assertEqual(self.judge.requests, [("うん", TEXT, True)])
        self.assertEqual(self.judge.request_metadata, [{"transcript_final": False}])
        await self.feed("うん", final=True)
        await self.step(.1)
        self.assertIsNotNone(self.r.playing)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(len(self.judge.requests), 1)

    async def test_same_text_final_rechecks_partial_short_and_long_once(self):
        for choice in ("short", "long"):
            with self.subTest(choice=choice):
                # Each subcase starts a fresh listening turn, without the core
                # worker / actual robot playback used by integration tests.
                self.r.preparation_context = None
                before = len(self.judge.requests)
                self.judge.result = answer(choice)
                self.r.input_last_voice = self.now
                await self.feed("今日は出かけました。", final=False, item=choice)
                changed = self.r.input_last_change
                await self.step(.81)
                self.assertFalse(self.judge.request_metadata[-1]["transcript_final"])
                self.now += .05
                await self.feed("今日は出かけました。", final=True, item=choice)
                self.assertEqual(self.r.input_last_change, changed)
                self.judge.result = answer("complete")
                await self.step(.46)  # Keep the existing half-second request interval.
                self.assertEqual(len(self.judge.requests), before + 2)
                self.assertTrue(self.judge.request_metadata[-1]["transcript_final"])
                self.assertEqual(self.judge.request_metadata[-1]["text_stable_ms"], 1320.0)
                await self.step(.26)
                self.assertEqual(self.r.jobs.get_nowait()[0], "今日は出かけました。")
                await self.feed("今日は出かけました。", final=True, item=choice)
                await self.step(.6)
                self.assertEqual(len(self.judge.requests), before + 2)

    async def test_same_text_final_during_inflight_short_and_long_rechecks_once(self):
        for choice in ("short", "long"):
            with self.subTest(choice=choice):
                self.r.preparation_context = None
                before = len(self.judge.requests)
                self.judge.started.clear()
                self.judge.gate = asyncio.Event()
                self.judge.result = answer(choice)
                self.r.input_last_voice = self.now
                await self.feed("話し終わりました。", final=False, item=choice)
                changed = self.r.input_last_change
                await self.step(.81, settle=False)
                await asyncio.wait_for(self.judge.started.wait(), 2)
                task = self.r.judge_task
                self.now += .1
                await self.feed("話し終わりました。", final=True, item=choice)
                self.assertEqual(self.r.input_last_change, changed)
                self.judge.gate.set()
                await task
                await asyncio.sleep(0)
                self.assertEqual(self.r.verdict["choice"], choice)
                self.assertFalse(self.r.verdict["transcript_final"])
                self.judge.result = answer("complete")
                await self.step(.41)
                self.assertEqual(len(self.judge.requests), before + 2)
                self.assertTrue(self.judge.request_metadata[-1]["transcript_final"])
                await self.step(.26)
                self.assertEqual(self.r.jobs.get_nowait()[0], "話し終わりました。")

    async def test_same_text_final_reuses_partial_complete_without_duplicate_request(self):
        await self.feed("最後まで話しました。", final=False)
        changed = self.r.input_last_change
        await self.step(.81)
        await self.step(.3)
        self.assertTrue(self.r.jobs.empty())
        await self.feed("最後まで話しました。", final=True)
        self.assertEqual(self.r.input_last_change, changed)
        await self.step()
        self.assertEqual(self.r.jobs.get_nowait()[0], "最後まで話しました。")
        self.assertEqual(len(self.judge.requests), 1)

    async def test_same_text_final_during_inflight_complete_reuses_answer(self):
        self.judge.gate = asyncio.Event()
        await self.feed("最後まで話しました。", final=False)
        changed = self.r.input_last_change
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        task = self.r.judge_task
        self.now += .1
        await self.feed("最後まで話しました。", final=True)
        self.assertEqual(self.r.input_last_change, changed)
        self.judge.gate.set()
        await task
        await asyncio.sleep(0)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "最後まで話しました。")
        self.assertEqual(len(self.judge.requests), 1)

    async def test_out_of_order_final_segments_keep_sequence_order(self):
        await self.feed("後半です。", item="two", sequence=1)
        await self.feed("前半です。", item="one", sequence=0)
        self.assertEqual(self.r.input_text, "前半です。後半です。")
        self.assertEqual(self.events[-1]["text"], "前半です。後半です。")
        self.assertTrue(self.events[-1]["replace"])

    async def test_microphone_never_interrupts_playback_by_loudness(self):
        self.playing()
        revision = self.r.input_revision
        await self.r.microphone(b"\x00\x10" * 4800)
        self.assertIsNotNone(self.r.playing)
        self.assertEqual(self.r.input_revision, revision)
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.core.deliveries, [])

    async def test_candidate_can_interrupt_during_continued_sound_but_not_commit_turn(self):
        self.playing()
        self.judge.gate = asyncio.Event()
        await self.feed("その説明は違います", final=False)
        await self.step(.21, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        await self.r.microphone(b"\x00\x10" * 480)
        self.judge.gate.set()
        await self.r.judge_task
        await self.deliveries()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.input_text, "その説明は違います")
        self.assertIsNone(self.r.verdict)  # Sound invalidates only the EOT part.
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.core.deliveries, [("approved1", "interrupted", "")])

    async def test_interrupt_and_short_turn_are_independent(self):
        self.playing()
        self.judge.result = answer("short", "interrupt")
        await self.feed("いや、その話ではなく、昨日のことです")
        await self.step(.21)
        await self.deliveries()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.verdict["choice"], "short")
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.core.deliveries, [("approved1", "interrupted", "")])

    async def test_backchannel_needs_final_and_quiet_and_keeps_ack(self):
        self.playing()
        revision, generation = self.r.input_revision, self.r.output_generation
        self.judge.result = answer(interruption="backchannel")
        self.r.input_last_voice = self.now
        await self.feed("うん", final=False)
        await self.step(.21)
        await self.step(1)
        self.assertEqual(self.r._text("candidate"), "うん")
        await self.r.microphone(b"\x00\x10" * 480)
        await self.feed("うん", final=True)
        await self.step(.5)
        self.assertEqual(self.r._text("candidate"), "うん")
        await self.step(.31)
        self.assertEqual(self.r._text("candidate"), "")
        self.assertEqual(self.r.input_text, "")
        self.assertEqual((self.r.input_revision, self.r.output_generation), (revision, generation))
        await self.r.played("approved1")
        await self.deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])

    async def test_ignored_ack_reopens_when_same_segment_grows(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("うん")
        await self.step(.21)
        await self.step(.01)
        self.assertEqual(self.r.segments["one"]["lane"], "ignored")
        await self.feed("うん、でも違います", final=False)
        self.assertFalse(self.r.segments["one"]["final"])
        self.judge.result = answer("short", "interrupt")
        await self.step(.51)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.input_text, "うん、でも違います")
        self.assertEqual(self.judge.requests[-1][0], "うん、でも違います")

    async def test_next_segment_after_ignored_ack_is_not_discarded(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("うん")
        await self.step(.21)
        await self.step(.01)
        self.judge.result = answer("short", "interrupt")
        await self.feed("でも違います", item="two", sequence=1)
        await self.step(.51)
        self.assertEqual(self.r.input_text, "でも違います")

    async def test_stale_backchannel_answer_cannot_drop_added_correction(self):
        self.playing()
        self.judge.gate = asyncio.Event()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("うん", final=False)
        await self.step(.21, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        await self.feed("うん、でも違います", final=True)
        self.judge.gate.set()
        await self.r.judge_task
        self.assertEqual(self.r._text("candidate"), "うん、でも違います")
        self.assertFalse(any(e["type"] == "backchannel_ignored" for e in self.logs()))

    async def test_bad_backchannel_label_cannot_discard_obvious_continuation(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("うん、でも違います")
        await self.step(.21)
        self.assertEqual(self.r.verdict["choice"], "uncertain")
        self.assertEqual(self.r._text("candidate"), "うん、でも違います")

    async def test_recorded_weak_backchannel_never_becomes_timeout_interruption(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel", confidence=.49)
        self.judge.result["interruption"]["probabilities"] = {
            "backchannel": .73, "interrupt": .01, "uncertain": .26}
        await self.feed("お", final=False)
        await self.step(.21)
        self.assertEqual(self.r.verdict["choice"], "uncertain")
        self.assertTrue(self.r.verdict["backchannel_dominant"])
        await self.feed("お", final=True)
        await self.step(.51)
        self.assertEqual(len(self.judge.requests), 2)
        await self.step(8)
        self.assertIsNotNone(self.r.playing)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.r.segments["one"]["lane"], "ignored")
        self.assertTrue(any(e.get("reason") == "bounded_backchannel_dominant" for e in self.logs()))

    async def test_failed_final_refresh_preserves_weak_backchannel_evidence(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel", confidence=.49)
        self.judge.result["interruption"]["probabilities"] = {
            "backchannel": .73, "interrupt": .01, "uncertain": .26}
        await self.feed("お", final=False)
        await self.step(.21)
        self.judge.error = TimeoutError("offline")
        await self.feed("お", final=True)
        await self.step(.51)
        self.assertEqual(self.r.verdict["choice"], "error")
        await self.step(8)
        self.assertIsNotNone(self.r.playing)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.r.segments["one"]["lane"], "ignored")

    async def test_weak_backchannel_timeout_keeps_editable_words_for_late_correction(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel", confidence=.49)
        self.judge.result["interruption"]["probabilities"] = {
            "backchannel": .73, "interrupt": .01, "uncertain": .26}
        await self.feed("お")
        await self.step(.21)
        await self.step(8)
        self.judge.result = answer()
        await self.feed("お、でもさっきの説明は違います。")
        await self.step(.51)
        await self.step(.26)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.jobs.get_nowait()[0], "お、でもさっきの説明は違います。")

    async def test_weak_backchannel_score_cannot_discard_substantive_continuation(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel", confidence=.49)
        self.judge.result["interruption"]["probabilities"] = {
            "backchannel": .73, "interrupt": .01, "uncertain": .26}
        text = "うん、でも明日の予定も話したいです。"
        await self.feed(text)
        await self.step(.21)
        self.assertFalse(self.r.verdict["backchannel_dominant"])
        await self.step(8)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.jobs.get_nowait()[0], text)

    async def test_material_interrupt_probability_is_not_a_protected_backchannel(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel", confidence=.49)
        self.judge.result["interruption"]["probabilities"] = {
            "backchannel": .60, "interrupt": .20, "uncertain": .20}
        text = "確認したいことがあります。"
        await self.feed(text)
        await self.step(.21)
        self.assertFalse(self.r.verdict["backchannel_dominant"])
        await self.step(8)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.jobs.get_nowait()[0], text)

    async def test_final_refresh_inflight_cannot_be_overruled_by_timeout(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel", confidence=.49)
        self.judge.result["interruption"]["probabilities"] = {
            "backchannel": .73, "interrupt": .01, "uncertain": .26}
        await self.feed("お", final=False)
        await self.step(.21)
        self.judge.gate = asyncio.Event()
        self.judge.result = answer()
        await self.feed("お", final=True)
        await self.step(8, settle=False)
        await asyncio.sleep(0)
        await self.step(.5, settle=False)
        self.assertEqual(self.r._text("candidate"), "お")
        self.assertFalse(any(e["type"] == "backchannel_ignored" for e in self.logs()))
        self.judge.gate.set()
        await self.r.judge_task
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "お")

    async def test_played_while_judge_waits_keeps_original_context_for_rejudge(self):
        self.playing()
        self.judge.gate = asyncio.Event()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("なるほど")
        await self.step(.21, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        await self.r.played("approved1")
        self.judge.gate.set()
        await self.r.judge_task
        await asyncio.sleep(0)
        await self.deliveries()
        self.assertEqual(self.r._text("candidate"), "なるほど")
        self.assertFalse(self.r._offer_current(self.r.input_revision, self.r.output_generation))
        await self.step(.51)
        await self.step(.01)
        self.assertTrue(self.judge.requests[-1][2])
        self.assertEqual(self.judge.requests[-1][1], TEXT)
        self.assertEqual(self.r.input_text, "")
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])

    async def test_normal_judge_survives_sound_but_commit_waits_for_quiet(self):
        self.judge.gate = asyncio.Event()
        await self.feed("今日は")
        await self.step(1, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        await self.r.microphone(b"\x00\x10" * 480)
        self.judge.gate.set()
        await self.r.judge_task
        self.assertEqual(self.r.verdict["choice"], "complete")
        self.assertTrue(self.r.jobs.empty())
        await self.step(.26)
        self.assertTrue(self.r.jobs.empty())
        await self.step(.55)
        self.assertEqual(self.r.jobs.get_nowait()[0], "今日は")
        self.assertEqual(len(self.judge.requests), 1)

    async def test_final_turn_reuses_one_judgment_through_noise_and_preserves_late_words(self):
        await self.feed("今日は買い物に行きました。")
        self.client.pending_audio = True
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                         "item_id": "noise", "sequence": 1})
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(.21)
        self.assertEqual(len(self.judge.requests), 1)
        self.assertTrue(self.judge.request_metadata[-1]["transcript_final"])
        self.assertTrue(self.r.jobs.empty())
        started = self.r.noise_turn_started
        self.now = started+1.9
        await self.feed("", item="noise", sequence=1)
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                         "item_id": "late", "sequence": 2})
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(.11)
        self.assertEqual(self.r.jobs.get_nowait()[0], "今日は買い物に行きました。")
        self.assertEqual(len(self.judge.requests), 1)
        self.assertTrue(self.r.segments["late"]["noise_bypassed"])
        self.assertNotEqual(self.r.segments["late"]["lane"], "consumed")
        await self.feed("ただ、ひとつ訂正があります。", final=False, item="late", sequence=2)
        self.assertEqual(self.r._text("preparation"), "ただ、ひとつ訂正があります。")
        self.assertEqual(self.r.segments["late"]["text"], "ただ、ひとつ訂正があります。")

    async def test_old_empty_candidates_do_not_make_new_answer_a_backchannel(self):
        self.playing()
        for item, sequence in (("old-one", 0), ("old-two", 1)):
            await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                             "item_id": item, "sequence": sequence})
        await self.r.played("approved1")
        await self.deliveries()
        await self.feed("はい", item="answer", sequence=2)
        self.assertEqual(self.r.segments["answer"]["lane"], "normal")
        self.assertEqual(self.r.input_text, "はい")
        self.assertFalse(self.r._text("candidate"))
        self.assertEqual(self.r._context(), {})

    async def test_first_words_in_old_empty_candidate_after_playback_are_normal_answer(self):
        self.playing()
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                         "item_id": "delayed", "sequence": 0})
        await self.r.played("approved1")
        await self.deliveries()
        await self.feed("はい", item="delayed", sequence=0)
        self.assertEqual(self.r.segments["delayed"]["lane"], "normal")
        self.assertEqual(self.r.input_text, "はい")
        self.assertFalse(self.r._text("candidate"))
        self.assertEqual(self.r.segments["delayed"]["context"], {})

    async def test_meaningful_candidate_uses_current_reply_context_before_old_empty_context(self):
        self.playing()
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                         "item_id": "old-empty", "sequence": 0})
        await self.r.played("approved1")
        await self.deliveries()
        self.r.playing = {"id": "second-reply", "kind": "listen", "text": "新しい話への返答です。"}
        await self.feed("その点は違います", final=False, item="current", sequence=1)
        self.assertEqual(self.r._context(), {"id": "second-reply", "text": "新しい話への返答です。"})
        self.assertEqual(self.r._snapshot("candidate")["context"]["id"], "second-reply")
        self.assertEqual(self.r.segments["old-empty"]["context"]["id"], "approved1")

    async def test_error_fallback_ack_keeps_empty_row_for_late_normal_words(self):
        self.playing()
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                         "item_id": "delayed", "sequence": 0})
        self.judge.error = RuntimeError("offline")
        await self.feed("はい", item="ack", sequence=1)
        await self.step(.21)
        started = self.r.noise_turn_started
        self.now = started+2.1
        await self.feed("", item="noise-proof", sequence=2)
        await self.step()
        self.assertEqual(self.r.segments["ack"]["lane"], "ignored")
        self.assertNotIn(self.r.segments["delayed"]["lane"], ("ignored", "consumed"))
        self.assertTrue(self.r.segments["delayed"]["noise_bypassed"])
        await self.r.played("approved1")
        await self.deliveries()
        await self.feed("いいえ、先ほどの話を訂正します。", item="delayed", sequence=0)
        self.assertEqual(self.r.segments["delayed"]["lane"], "normal")
        self.assertEqual(self.r.input_text, "いいえ、先ほどの話を訂正します。")

    async def test_complete_grace_is_invalidated_by_text_change(self):
        await self.feed("今日は良かったです")
        await self.step(1)
        await self.feed("でも気になることがあります", item="two", sequence=1)
        await self.step(.1)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.r.input_text, "今日は良かったですでも気になることがあります")

    async def test_short_and_long_do_not_commit_on_first_pause_and_bound_wait(self):
        for choice, wait in (("short", 2), ("long", 3.5)):
            with self.subTest(choice=choice):
                self.r.preparation_context = None
                self.r.segments.clear()
                self.r.input_text = ""
                self.r.verdict = None
                self.r.next_judge_at = 0
                self.judge.result = answer(choice)
                await self.feed("ええと、続きは", item=choice)
                await self.step(.81)
                self.assertEqual(self.r.verdict["choice"], choice)
                self.assertGreaterEqual(self.r.verdict["due"], self.r.input_last_change+wait)
                await self.step(.3)
                self.assertTrue(self.r.jobs.empty())
                await self.step(8)
                self.assertEqual(self.r.jobs.get_nowait()[0], "ええと、続きは")
        self.assertTrue(any(e["type"] == "jev_bounded_fallback" for e in self.logs()))

    async def test_unchanged_final_short_and_long_are_judged_once_through_timeout(self):
        for choice in ("short", "long"):
            with self.subTest(choice=choice):
                self.r.preparation_context = None
                before = len(self.judge.requests)
                self.judge.result = answer(choice)
                self.r.input_last_voice = self.now
                await self.feed("ええと、続きは", item=choice)
                for _ in range(81):
                    await self.step(.1)
                self.assertEqual(len(self.judge.requests), before + 1)
                self.assertEqual(self.r.jobs.get_nowait()[0], "ええと、続きは")
                self.assertTrue(self.r.jobs.empty())

    async def test_cached_wait_is_replaced_when_new_words_arrive(self):
        self.judge.result = answer("short")
        self.r.input_last_voice = self.now
        await self.feed("それで", final=True)
        for _ in range(30):
            await self.step(.1)
        self.assertEqual(len(self.judge.requests), 1)
        self.judge.result = answer("complete")
        await self.feed("今日は楽しかったです。", item="new", sequence=1)
        self.r.input_last_voice = self.now
        await self.step(.81)
        await self.step(.26)
        self.assertEqual(len(self.judge.requests), 2)
        self.assertEqual(self.r.jobs.get_nowait()[0], "それで今日は楽しかったです。")

    async def test_uncertain_partial_candidate_rechecks_final_once_then_waits(self):
        self.playing()
        self.r.input_last_voice = self.now
        self.judge.result = answer(interruption="uncertain")
        await self.feed("はい", final=False)
        await self.step(.21)
        for _ in range(15):
            await self.step(.1)
        self.assertEqual(len(self.judge.requests), 1)
        await self.feed("はい", final=True)
        await self.step(.1)
        self.assertEqual(len(self.judge.requests), 2)
        self.assertTrue(self.judge.request_metadata[-1]["transcript_final"])
        for _ in range(65):
            await self.step(.1)
        self.assertEqual(len(self.judge.requests), 2)
        self.assertEqual(self.r.segments["one"]["lane"], "ignored")
        self.assertIsNotNone(self.r.playing)

    async def test_cached_candidate_rechecks_when_heard_context_changes(self):
        self.playing()
        self.r.input_last_voice = self.now
        self.judge.result = answer(interruption="uncertain")
        await self.feed("それは", final=True)
        await self.step(.81)
        self.assertEqual(len(self.judge.requests), 1)
        self.r.segments["one"]["context"]["text"] = "別の説明です。"
        await self.step(.51)
        self.assertEqual(len(self.judge.requests), 2)
        self.assertEqual(self.judge.requests[-1][1], "別の説明です。")

    async def test_jev_failure_uses_two_second_silence_without_ending_connection(self):
        self.judge.error = RuntimeError("private upstream detail")
        await self.feed("今日は楽しかったです")
        await self.step(.81)
        await self.step(1.0)
        self.assertTrue(self.r.jobs.empty())
        await self.step(.21)
        self.assertEqual(self.r.jobs.get_nowait()[0], "今日は楽しかったです")
        self.assertFalse(self.r.finished.is_set())
        self.assertNotIn("private", (self.r.path / "events.jsonl").read_text(encoding="utf-8"))

    async def test_jev_failure_protects_short_ack_during_playback(self):
        self.playing()
        self.judge.error = RuntimeError("error")
        await self.feed("はい")
        await self.step(.21)
        await self.step(2)
        self.assertIsNotNone(self.r.playing)
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.r.segments["one"]["lane"], "ignored")

    async def test_pending_audio_and_untranscribed_segment_block_old_text_commit(self):
        await self.feed("前半です")
        await self.step(.81)
        self.client.pending_audio = True
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(1)
        await self.step(.3)
        self.assertTrue(self.r.jobs.empty())
        await self.r._accept_transcript({"type": "session.input_transcript.pending", "item_id": "two", "sequence": 1})
        self.client.pending_audio = False
        await self.step(2)
        await self.step(.3)
        self.assertTrue(self.r.jobs.empty())
        await self.feed("後半です", item="two", sequence=1)
        await self.step(.51)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "前半です後半です")

    async def test_empty_pending_allows_tts_preparation_but_waits_for_final(self):
        await self.r._accept_transcript({"type": "session.input_transcript.pending", "item_id": "one", "sequence": 0})
        self.assertTrue(self.r._offer_current(0, 0))
        self.assertFalse(self.r._all_final("normal"))
        await self.feed("")
        self.assertTrue(self.r._offer_current(0, 0))

    async def test_empty_noise_final_does_not_restart_existing_turn_wait(self):
        self.judge.result = answer("short")
        self.r.input_last_voice = self.now
        await self.feed("今日は出かけました。")
        original_changed = self.r.input_last_change
        revision = self.r.input_revision
        await self.step(.81)
        original_verdict = self.r.verdict
        original_version = self.r.buffer_version
        self.now = original_changed + 5
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                         "item_id": "noise", "sequence": 1})
        version = self.r.buffer_version
        self.assertFalse(self.r._all_final("normal"))
        self.now = original_changed + 6
        await self.feed("", item="noise", sequence=1)
        self.assertTrue(self.r._all_final("normal"))
        self.assertEqual(self.r.buffer_version, version)
        self.assertEqual(self.r.buffer_version, original_version)
        self.assertIs(self.r.verdict, original_verdict)
        self.assertEqual(self.r.input_last_change, original_changed)
        self.assertEqual(self.r.input_revision, revision)
        self.assertEqual(self.r.input_text, "今日は出かけました。")
        await self.step()  # A due short-wait retry may run; empty audio did not reset it.
        await self.step(.49)
        self.assertTrue(self.r.jobs.empty())
        await self.step(1.56)
        self.assertEqual(self.r.jobs.get_nowait()[0], "今日は出かけました。")

    async def test_empty_final_removing_real_interim_restarts_content_wait(self):
        self.judge.result = answer("short")
        self.r.input_last_voice = self.now
        await self.feed("確定した前半。")
        started = self.now
        self.now += 1
        await self.feed("仮の後半", final=False, item="two", sequence=1)
        revision = self.r.input_revision
        self.now = started + 6
        await self.feed("", item="two", sequence=1)
        self.assertEqual(self.r.input_last_change, self.now)
        self.assertEqual(self.r.input_revision, revision + 1)
        self.assertEqual(self.r.input_text, "確定した前半。")
        self.assertTrue(self.r._all_final("normal"))
        await self.step(.21)
        await self.step(1.8)  # Eight seconds since original text is not enough.
        self.assertTrue(self.r.jobs.empty())
        await self.step(6.05)
        self.assertEqual(self.r.jobs.get_nowait()[0], "確定した前半。")

    async def test_empty_final_after_interim_releases_pending_reply_through_pcm_playback(self):
        await self.feed("仮の文字", final=False)
        revision = self.r.input_revision
        self.assertEqual(self.r.input_text, "仮の文字")
        await self.r._offer_pending(revision, self.r.output_generation)
        self.assertEqual(self.speech.calls, [])

        await self.feed("", final=True)
        self.assertEqual(self.r._text("normal"), "")
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.r.input_revision, revision + 1)
        self.assertEqual(self.events[-1], {"type": "input_transcript", "text": "", "replace": True})
        self.assertTrue(self.r.jobs.empty())
        # The old snapshot is invalid; a fresh state poll may offer the reply.
        await self.r._offer_pending(revision, self.r.output_generation)
        self.assertEqual(self.speech.calls, [])
        await self.r._refresh_state()
        task = self.r.speech_task
        self.assertIsNotNone(task)
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.speech.calls, [TEXT])
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertTrue(any(e.get("type") == "playback" and e.get("id") == "approved1" for e in self.events))
        self.assertTrue(any(e.get("type") == "playback_end" and e.get("id") == "approved1" for e in self.events))
        self.assertEqual(self.r.playing["kind"], "reply")
        self.assertEqual(self.core.deliveries, [])

    async def test_empty_final_preserves_other_normal_segments_and_correction_caption(self):
        await self.feed("前半。", item="one", sequence=0)
        await self.feed("誤認識", final=False, item="two", sequence=1)
        await self.feed("後半。", item="three", sequence=2)
        revision = self.r.input_revision
        await self.feed("", item="two", sequence=1)
        self.assertEqual(self.r.input_text, "前半。後半。")
        self.assertEqual(self.r._text("normal"), "前半。後半。")
        self.assertEqual(self.events[-1]["text"], "前半。後半。")
        self.assertEqual(self.r.input_revision, revision + 1)
        self.assertEqual(self.r.segments["one"]["lane"], "normal")
        self.assertEqual(self.r.segments["three"]["lane"], "normal")
        await self.feed("訂正した後半。", item="three", sequence=2)
        self.assertEqual(self.r.input_text, "前半。訂正した後半。")
        self.assertEqual(self.events[-1]["text"], self.r.input_text)
        self.assertEqual(self.r.input_revision, revision + 2)

    async def test_empty_final_invalidates_inflight_normal_judge(self):
        self.judge.gate = asyncio.Event()
        await self.feed("途中の仮認識", final=False)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        task = self.r.judge_task
        revision, version = self.r.input_revision, self.r.buffer_version
        await self.feed("", final=True)
        self.assertGreater(self.r.input_revision, revision)
        self.assertGreater(self.r.buffer_version, version)
        self.judge.gate.set()
        await task
        await self.step(10)
        self.assertIsNone(self.r.verdict)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.r.input_text, "")
        self.assertTrue(self.r._offer_current(self.r.input_revision, self.r.output_generation))
        self.assertTrue(any(e["type"] == "jev_result_discarded" for e in self.logs()))

    async def test_empty_candidate_final_clears_caption_without_revoking_real_playback(self):
        await self.r._offer_pending(0, 0)
        await self.r.speech_task
        message = self.r.playing
        revision, generation = self.r.input_revision, self.r.output_generation
        self.judge.gate = asyncio.Event()
        await self.feed("仮の相槌", final=False)
        await self.step(.21, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        task = self.r.judge_task
        version = self.r.buffer_version
        await self.feed("", final=True)
        self.assertEqual(self.r._text("candidate"), "")
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.events[-1]["text"], "")
        self.assertGreater(self.r.buffer_version, version)
        self.assertEqual((self.r.input_revision, self.r.output_generation), (revision, generation))
        self.judge.gate.set()
        await task
        self.assertIsNone(self.r.verdict)
        self.assertIs(self.r.playing, message)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.r.playing_started = time.monotonic() - 1
        await self.r.played(message["id"])
        await self.deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])

    async def test_late_empty_final_cannot_rewrite_an_already_committed_caption(self):
        await self.feed("確定済み。")
        await self.step(.81)
        await self.step(.26)
        self.assertEqual(self.r.segments["one"]["lane"], "consumed")
        await self.feed("", final=True)
        await self.feed("次の話。", item="two", sequence=1)
        self.assertEqual(self.r.segments["one"]["text"], "確定済み。")
        self.assertEqual(self.r._text("preparation"), "次の話。")
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.events[-1]["text"], "確定済み。次の話。")

    async def test_empty_candidate_final_does_not_block_next_reply_after_ack(self):
        self.playing()
        await self.r._accept_transcript({"type": "session.input_transcript.pending", "item_id": "one", "sequence": 0})
        self.assertEqual(self.r.segments["one"]["lane"], "candidate")
        await self.feed("")
        await self.r.played("approved1")
        await self.deliveries()
        await asyncio.sleep(0)
        self.assertEqual(self.r.segments["one"]["lane"], "consumed")
        self.assertTrue(self.r._offer_current(self.r.input_revision, self.r.output_generation))

    async def test_acknowledgement_during_pcm_send_preserves_every_byte_and_final_delivery(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def send_bytes(pcm):
            entered.set()
            await release.wait()
            self.audio.append(pcm)

        self.r.send_bytes = send_bytes
        await self.r._offer_pending(0, 0)
        speech_task = self.r.speech_task
        await asyncio.wait_for(entered.wait(), 2)
        self.judge.result = answer(interruption="backchannel")
        await self.feed("なるほど")
        await self.step(.21)
        await self.step(.01)
        self.assertEqual(self.r.input_revision, 0)
        self.assertEqual(self.r.output_generation, 0)
        self.assertIsNotNone(self.r.playing)
        release.set()
        await asyncio.wait_for(speech_task, 2)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertTrue(any(e["type"] == "playback_end" for e in self.events))
        await self.r.played("approved1")
        await self.deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])

    async def test_recent_voice_allows_tts_preparation_without_canceling_existing(self):
        await self.r.microphone(b"\x00\x10" * 480)
        self.assertTrue(self.r._offer_current(0, 0))
        self.now += .31
        self.assertTrue(self.r._offer_current(0, 0))

    async def test_new_content_cancels_tts_after_preparation_judgment(self):
        self.speech.before_first = asyncio.Event()
        self.r._new_preparation("今日は買い物に行きました。")
        await self.r._offer_pending(0, 0)
        speech_task = self.r.speech_task
        await asyncio.wait_for(self.speech.started.wait(), 2)
        await self.feed("話を変えます")
        self.assertIsNotNone(self.r.approved)
        await self.step(.81)
        await asyncio.wait_for(asyncio.gather(speech_task, return_exceptions=True), 2)
        await self.deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "cancelled", "")])
        self.assertEqual(self.audio, [])

    async def test_typed_input_invalidates_inflight_candidate_and_stops_playback(self):
        self.playing()
        self.judge.gate = asyncio.Event()
        await self.feed("うん", final=False)
        await self.step(.21, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        await self.r.enqueue("これは手入力です。")
        self.judge.gate.set()
        await self.r.judge_task
        await self.deliveries()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.jobs.get_nowait()[0], "これは手入力です。")
        self.assertIsNone(self.r.verdict)

    async def test_local_stop_is_logged_separately_and_does_not_need_judge(self):
        self.playing()
        await self.feed("待って", final=False)
        await self.deliveries()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.judge.requests, [])
        self.assertTrue(any(e["type"] == "local_explicit_stop" for e in self.logs()))

    async def assert_short_interruption_waits(self, text, kind):
        self.playing()
        await self.feed(text, final=False)
        await self.deliveries()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.input_text, text)
        self.assertEqual(self.r.waiting_for_continuation, kind)
        await self.step(1)
        await self.feed(text, final=True)
        # Even a classifier configured to say "complete" and the ordinary
        # eight-second fallback cannot turn a bare command into a reply.
        await self.step(9)
        await self.step(9)
        self.assertTrue(self.r.jobs.empty())
        self.assertEqual(self.judge.requests, [])
        self.assertEqual(self.r.segments["one"]["lane"], "normal")
        self.assertEqual(self.r.input_text, text)

    async def test_wait_command_interrupts_immediately_and_never_auto_submits(self):
        await self.assert_short_interruption_waits("待って", "pause")

    async def test_short_correction_interrupts_immediately_and_waits_for_detail(self):
        await self.assert_short_interruption_waits("違う", "correction")

    async def test_short_denial_interrupts_immediately_and_waits_for_detail(self):
        await self.assert_short_interruption_waits("いや", "correction")

    async def test_held_command_accepts_same_asr_segment_continuation_exactly(self):
        await self.assert_short_interruption_waits("待って", "pause")
        text = "待って、昨日ではなく今日の話です。"
        await self.feed(text)
        self.assertIsNone(self.r.waiting_for_continuation)
        await self.step(.21)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], text)
        self.assertFalse(self.r.input_interrupted_output)
        self.assertEqual(self.r.last_committed_text, text)

    async def test_held_denial_accepts_later_asr_segment_without_losing_prefix(self):
        await self.assert_short_interruption_waits("いや", "correction")
        detail = "昨日ではなく今日の話です。"
        await self.feed(detail, item="two", sequence=1)
        self.assertIsNone(self.r.waiting_for_continuation)
        await self.step(.21)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "いや" + detail)
        self.assertFalse(self.r.input_interrupted_output)

    async def test_denial_during_normal_listening_remains_a_valid_answer(self):
        await self.feed("いいえ")
        await self.step(.21)
        await self.step(.26)
        self.assertEqual(self.r.jobs.get_nowait()[0], "いいえ")
        self.assertIsNone(self.r.waiting_for_continuation)

    async def test_naruhodo_backchannel_does_not_interrupt_or_enter_command_wait(self):
        self.playing()
        self.judge.result = answer(interruption="backchannel")
        await self.feed("なるほど")
        await self.step(.21)
        await self.step(.01)
        self.assertIsNotNone(self.r.playing)
        self.assertEqual(self.r.segments["one"]["lane"], "ignored")
        self.assertIsNone(self.r.waiting_for_continuation)
        self.assertTrue(self.r.jobs.empty())

    async def test_typed_continuation_keeps_held_command_once(self):
        await self.assert_short_interruption_waits("待って", "pause")
        self.assertTrue(await self.r.enqueue("明日の話です。"))
        text = "待って\n明日の話です。"
        self.assertEqual(self.r.jobs.get_nowait()[0], text)
        self.assertEqual(self.r.last_committed_text, text)
        self.assertEqual(self.r.segments["one"]["lane"], "consumed")
        self.assertIsNone(self.r.waiting_for_continuation)

    async def test_rejected_long_typed_continuation_preserves_held_command(self):
        await self.assert_short_interruption_waits("待って", "pause")
        self.assertFalse(await self.r.enqueue("あ" * 1999))
        self.assertEqual(self.r.input_text, "待って")
        self.assertEqual(self.r.waiting_for_continuation, "pause")
        self.assertEqual(self.r.segments["one"]["lane"], "normal")
        self.assertTrue(self.r.jobs.empty())

    async def assert_short_interruption_cancels_generation(self, text, kind):
        responder = CancellableResponder()
        core = ChatCore(Path(self.temp.name), settings=ChatSettings(), responder=responder)
        sid = core.create()["id"]
        self.r.core, self.r.sid = core, sid
        self.r.tasks = set()
        self.r.spawn(self.r.process_inputs(), self.r.tasks, "core")
        try:
            await self.r.enqueue(ORIGINAL)
            async with asyncio.timeout(2):
                while not responder.started.is_set():
                    await asyncio.sleep(.005)
            await self.feed(text)
            self.assertTrue(responder.first_cancel_event.is_set())
            await asyncio.wait_for(self.r.jobs.join(), 2)
            await self.step(9)
            self.assertTrue(responder.cancelled.is_set())
            self.assertEqual(self.r.waiting_for_continuation, kind)
            self.assertEqual(self.r.input_text, text)
            self.assertFalse(self.r.finished.is_set())
            self.assertEqual(await asyncio.to_thread(core.pending, sid), [])
            state = await asyncio.to_thread(core.view, sid)
            self.assertIsNone(state["error"])
            self.assertEqual([row["content"] for row in state["messages"] if row["role"] == "user"], [ORIGINAL])
            stored = json.loads((core.data_dir / "sessions" / sid / "session.json").read_text(encoding="utf-8"))
            self.assertEqual([row["status"] for row in stored["requests"].values()], ["interrupted"])
            self.assertEqual(self.audio, [])
            self.assertEqual(self.judge.requests, [])

            continued = text + "、明後日に行きます。"
            await self.feed(continued)
            await self.step(.21)
            await self.step(.26)
            await asyncio.wait_for(self.r.jobs.join(), 2)
            self.assertEqual([row["text"] for row in await asyncio.to_thread(core.pending, sid)], [ANSWER])
            self.assertEqual(responder.histories[-1], [
                {"role": "user", "content": ORIGINAL},
                {"role": "user", "content": continued},
            ])
        finally:
            await self.r.close()
            await asyncio.to_thread(core.close)

    async def test_stop_cancels_inflight_backend_and_waits_without_emitting_old_reply(self):
        await self.assert_short_interruption_cancels_generation("待って", "pause")

    async def test_denial_cancels_inflight_backend_and_waits_without_emitting_old_reply(self):
        await self.assert_short_interruption_cancels_generation("違う", "correction")

    async def test_close_cancels_judge_and_saves_only_safe_metadata_once(self):
        self.judge.gate = asyncio.Event()
        self.judge.calls = [{"model": "offline", "request_ms": 5, "input_chars": 2,
                             "key": "private", "text": "private"}]
        await self.feed("話です")
        await self.step(1, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        task = self.r.judge_task
        await self.r.close()
        await self.r.close()
        self.assertTrue(task.cancelled())
        self.assertEqual(self.judge.closes, 1)
        self.assertEqual(self.client.disconnects, 1)
        self.assertEqual(self.speech.closes, 1)
        self.assertNotIn("private", (self.r.path / "jev_api_calls.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
