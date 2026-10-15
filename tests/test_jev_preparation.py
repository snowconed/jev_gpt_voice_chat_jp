"""Offline races for extra speech while Jev's first reply is being prepared.

Run the real input worker and buffered speech pipeline. The barriers represent
chat/API latency; no speech, Jev, or OpenAI API is contacted.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jev_runtime import JevSpeechRuntime
from test_jev_runtime import answer
from test_speech_runtime import Live, Speech
from test_voice_races import Core as RaceCore, output


HERE = Path(__file__).resolve().parents[1]
ORIGINAL = "今日はコンビニに行ってきました。"
REPETITION = "はい、コンビニに行ってきたんです。"
CORRECTION = "いや、コンビニではなくスーパーでした。"


def preparation(choice="retain", *, turn="complete", probability=1.0):
    result = answer(turn)
    result.pop("interruption")
    result["preparation"] = {
        "choice": choice, "confidence": probability,
        "probabilities": {key: probability if key == choice else (1-probability)/2
                          for key in ("retain", "revise", "uncertain")},
    }
    return result


class PreparationJudge:
    model = "offline-preparation-jev"

    def __init__(self):
        self.calls = []
        self.requests = []
        self.preparations = []
        self.result = preparation()
        self.error = None
        self.gate = None
        self.started = asyncio.Event()
        self.closed = False

    async def classify(self, text, assistant_text="", speaking=False, **metadata):
        self.requests.append((text, assistant_text, speaking, metadata))
        return answer()

    async def classify_preparation(self, text, **context):
        self.preparations.append({"text": text, **deepcopy(context)})
        # Freeze this request's answer before allowing a newer transcript.
        result, error = deepcopy(self.result), self.error
        self.started.set()
        if self.gate:
            await self.gate.wait()
        if error:
            raise error
        return result

    async def close(self):
        self.closed = True


class PreparationCore(RaceCore):
    def __init__(self):
        super().__init__()
        self.reply_kind = "reply"


    def submit(self, sid, text, rid):
        if self.before_submit:
            self.before_submit()
        self.events.append(("submit", text))
        self.inputs.append((text, rid))
        self.messages.append({"id": rid, "role": "user", "content": text, "kind": "user"})
        self.messages.append(output(f"reply-{len(self.inputs)}", "そこでどんなことがあったんですか？",
                                    kind=self.reply_kind))
        return self.view(sid)


class JevPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = (HERE / ".runtime" / "jev_preparation_tests").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.now = 1000.0
        self.clock = patch("jev_runtime.time", SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.core, self.client, self.speech, self.judge = PreparationCore(), Live(), Speech(), PreparationJudge()
        self.client.pending_audio = False
        self.events, self.audio, self.releases = [], [], []

        async def send_json(event):
            self.events.append(event)

        async def send_bytes(pcm):
            self.audio.append(pcm)

        self.r = JevSpeechRuntime(self.core, "a"*32, send_json, send_bytes,
                                  Path(self.temp.name), client=self.client,
                                  speech=self.speech, jev=self.judge)
        self.r.ready = True
        self.r.last_activity = self.now
        self.r.input_last_voice = 0
        self.r.tasks = set()
        self.worker = self.r.spawn(self.r.process_inputs(), self.r.tasks, "core")

    async def asyncTearDown(self):
        for release in self.releases:
            release.set()
        if self.judge.gate:
            self.judge.gate.set()
        if self.speech.before_first:
            self.speech.before_first.set()
        if self.speech.after_first:
            self.speech.after_first.set()
        await self.r.close()
        self.clock.stop()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root))
        self.temp.cleanup()

    async def feed(self, text, *, final=True, item="extra", sequence=1):
        await self.r._accept_transcript({
            "type": "session.input_transcript.completed" if final else "session.input_transcript.delta",
            "item_id": item, "sequence": sequence, "segment_text": text, "text": text,
        })

    async def step(self, seconds=0, *, settle=True):
        self.now += seconds
        await self.r._advance(self.now)
        task = self.r.judge_task
        if settle and task:
            await asyncio.wait_for(asyncio.shield(task), 2)
        await asyncio.sleep(0)

    async def wait_for(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.005)

    def block_first_submit(self):
        entered, release = asyncio.Event(), threading.Event()
        self.releases.append(release)
        loop = asyncio.get_running_loop()
        calls = 0

        def before_submit():
            nonlocal calls
            calls += 1
            if calls != 1:
                return
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(8):
                raise TimeoutError("The test did not release the first core request")

        self.core.before_submit = before_submit
        return entered, release

    async def begin_first(self, *, block=False, text=ORIGINAL):
        entered, release = self.block_first_submit() if block else (None, None)
        await self.feed(text, item="initial", sequence=0)
        await self.step(.81)
        await self.step(.26)
        if entered:
            await asyncio.wait_for(entered.wait(), 2)
        else:
            await asyncio.wait_for(self.r.jobs.join(), 2)
        return release

    async def offer(self):
        await self.r._offer_pending(self.r.input_revision, self.r.output_generation)
        return self.r.speech_task

    async def prepare_tts(self, *, original=ORIGINAL):
        await self.begin_first(text=original)
        self.speech.before_first = asyncio.Event()
        task = await self.offer()
        await asyncio.wait_for(self.speech.started.wait(), 2)
        return task

    def logs(self):
        return [json.loads(line) for line in (self.r.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    async def resolve(self, seconds=.81):
        await self.step(seconds)
        await self.step(.26)

    async def test_repetition_during_core_keeps_original_reply_and_one_submission(self):
        release = await self.begin_first(block=True)
        revision = self.r.input_revision
        await self.feed(REPETITION)
        self.assertEqual(self.r.segments["extra"]["lane"], "preparation")
        self.assertEqual(self.r.input_revision, revision)
        self.assertEqual(self.r.input_text, "")
        await self.resolve()
        self.assertEqual(self.judge.preparations[0]["original_text"], ORIGINAL)
        self.assertEqual(self.judge.preparations[0]["text"], REPETITION)
        self.assertTrue(self.judge.preparations[0]["transcript_final"])
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL])
        self.assertEqual(self.core.deliveries, [])
        self.assertEqual(len(self.speech.calls), 1)
        self.assertTrue(self.audio)
        self.assertFalse(any(row["type"] == "stale_reply_cancelled" for row in self.logs()))

    async def test_repetition_during_tts_keeps_same_pcm_and_delivery_waits_for_ack(self):
        task = await self.prepare_tts()
        revision = self.r.input_revision
        await self.feed(REPETITION)
        self.assertEqual(self.r.input_revision, revision)
        await self.resolve()
        self.assertNotIn("reply_text", self.judge.preparations[-1])
        self.speech.before_first.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(len(self.speech.calls), 1)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertEqual(self.core.deliveries, [])
        await self.r.played("reply-1")
        await asyncio.gather(*list(self.r.delivery_tasks))
        self.assertEqual(self.core.deliveries, [("reply-1", "completed", "")])
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL])

    async def test_unheard_reply_becoming_ready_keeps_inflight_preparation_verdict(self):
        release = await self.begin_first(block=True)
        self.judge.gate = asyncio.Event()
        await self.feed(REPETITION)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        judge_task = self.r.judge_task
        snapshot = self.r._snapshot("preparation")
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.speech.before_first = asyncio.Event()
        speech_task = await self.offer()
        await asyncio.wait_for(self.speech.started.wait(), 2)
        self.assertTrue(self.r.preparation_context["reply_text"])
        self.assertEqual(self.r._snapshot("preparation"), snapshot)
        self.judge.gate.set()
        await judge_task
        await asyncio.sleep(0)
        await self.step(.26)
        self.speech.before_first.set()
        await asyncio.wait_for(speech_task, 2)
        self.assertEqual(len(self.judge.preparations), 1)
        self.assertNotIn("reply_text", self.judge.preparations[0])
        self.assertEqual(self.r.segments["extra"]["lane"], "ignored_preparation")
        self.assertTrue(self.audio)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL])

    async def test_unchanged_uncertain_preparation_is_judged_once_before_fallback(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("uncertain")
        await self.feed(CORRECTION)
        for _ in range(41):
            await self.step(.1)
        self.assertEqual(len(self.judge.preparations), 1)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])

    async def test_changed_heard_context_invalidates_preparation_verdict(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("uncertain")
        await self.feed("はい")
        await self.step(.81)
        self.assertEqual(len(self.judge.preparations), 1)
        self.r.preparation_context["last_heard_text"] = "楽しかったですか？"
        self.judge.result = preparation("revise")
        await self.step(.51)
        self.assertEqual(len(self.judge.preparations), 2)
        self.assertEqual(self.judge.preparations[-1]["last_heard_text"], "楽しかったですか？")
        await self.step(.26)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, "はい"])

    async def test_ready_tts_waits_for_preparation_decision(self):
        task = await self.prepare_tts()
        self.judge.gate = asyncio.Event()
        await self.feed(REPETITION)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        self.speech.before_first.set()
        await self.wait_for(lambda: any(event["type"] == "speech_ready" for event in self.logs()))
        self.assertEqual(self.audio, [])
        self.assertFalse(task.done())
        self.assertIsNone(self.r.playing)
        self.judge.gate.set()
        if self.r.judge_task:
            await asyncio.wait_for(self.r.judge_task, 2)
        await self.step(.26)
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)
        self.assertEqual(len(self.speech.calls), 1)

    async def test_correction_during_core_submits_full_extra_once_and_cancels_first(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("revise")
        await self.feed(CORRECTION)
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])
        self.assertIn(("reply-1", "cancelled", ""), self.core.deliveries)
        self.assertEqual([event["id"] for event in self.events if event["type"] == "playback"], ["reply-2"])
        self.assertEqual(len(self.speech.calls), 1)

    async def test_correction_during_tts_sends_no_old_audio(self):
        first_tts = await self.prepare_tts()
        self.judge.result = preparation("revise")
        await self.feed(CORRECTION)
        await self.resolve()
        await asyncio.gather(first_tts, return_exceptions=True)
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual(self.audio, [])
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])
        self.assertIn(("reply-1", "cancelled", ""), self.core.deliveries)
        self.assertFalse(any(event["type"] == "playback" for event in self.events))

    async def test_stale_retain_cannot_discard_yes_but_correction(self):
        release = await self.begin_first(block=True)
        self.judge.gate = asyncio.Event()
        await self.feed("はい", final=True)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        initial_task = self.r.judge_task
        full_text = "はい、でもコンビニではなくスーパーでした。"
        await self.feed(full_text)
        self.judge.gate.set()
        await asyncio.wait_for(initial_task, 2)
        self.judge.result = preparation("revise")
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, full_text])
        self.assertEqual(self.audio, [])
        self.assertTrue(any(event["type"] == "jev_result_discarded" for event in self.logs()))

    async def test_unfinalized_retain_does_not_release_prepared_audio(self):
        task = await self.prepare_tts()
        await self.feed("はい", final=False)
        await self.resolve()
        self.assertEqual(self.judge.preparations, [])
        self.speech.before_first.set()
        await self.wait_for(lambda: any(event["type"] == "speech_ready" for event in self.logs()))
        self.assertEqual(self.audio, [])
        self.assertFalse(task.done())
        await self.feed("はい", final=True)
        await self.resolve()
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)
        self.assertEqual(len(self.core.inputs), 1)

    async def test_asr_pending_audio_blocks_retain_and_playback(self):
        task = await self.prepare_tts()
        await self.feed(REPETITION)
        self.client.pending_audio = True
        await self.resolve()
        self.speech.before_first.set()
        await self.wait_for(lambda: any(event["type"] == "speech_ready" for event in self.logs()))
        self.assertEqual(self.audio, [])
        self.assertFalse(task.done())
        self.client.pending_audio = False
        await self.resolve()
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)

    async def test_local_transport_backlog_arriving_after_verdict_blocks_playback(self):
        task = await self.prepare_tts()
        await self.feed(REPETITION)
        await self.step(.81)
        self.assertEqual(len(self.judge.preparations), 1)
        self.r.microphone_backlog_bytes = 48000
        await self.step(.26)
        self.speech.before_first.set()
        await self.wait_for(lambda: any(event["type"] == "speech_ready" for event in self.logs()))
        self.assertEqual(self.audio, [])
        self.assertEqual(self.r._text("preparation"), REPETITION)
        self.r.microphone_backlog_bytes = 0
        await self.step()
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)
        self.assertEqual(len(self.core.inputs), 1)

    async def test_typed_input_overrides_preparation_and_is_submitted_once(self):
        task = await self.prepare_tts()
        self.judge.gate = asyncio.Event()
        await self.feed(REPETITION)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        typed = "文字入力で訂正します。スーパーに行きました。"
        await self.r.enqueue(typed)
        self.judge.gate.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(self.r.jobs.join(), 2)
        await self.step(.81)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, typed])
        self.assertEqual(self.audio, [])
        self.assertFalse(self.r._text("preparation"))

    async def test_close_during_preparation_never_releases_old_audio(self):
        task = await self.prepare_tts()
        self.judge.gate = asyncio.Event()
        await self.feed(REPETITION)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        await asyncio.wait_for(self.r.close(), 2)
        self.assertTrue(task.done())
        self.assertEqual(self.audio, [])
        self.assertTrue(self.judge.closed)
        self.assertFalse(any(event["type"] == "playback" for event in self.events))

    async def test_explicit_stop_immediately_revokes_prepared_tts(self):
        task = await self.prepare_tts()
        revision = self.r.input_revision
        await self.feed("止めて", final=False)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*list(self.r.delivery_tasks))
        self.assertGreater(self.r.input_revision, revision)
        self.assertEqual(self.audio, [])
        self.assertIsNone(self.r.approved)
        self.assertIn(("reply-1", "cancelled", ""), self.core.deliveries)
        self.assertEqual(self.judge.preparations, [])
        await self.feed("止めて", final=True)
        await self.step(9)
        await self.step(9)
        self.assertEqual(self.r.waiting_for_continuation, "pause")
        self.assertEqual(self.r.input_text, "止めて")
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL])
        self.assertEqual([request[0] for request in self.judge.requests], [ORIGINAL])

    async def test_short_denial_revokes_preparation_then_keeps_following_detail(self):
        task = await self.prepare_tts()
        await self.feed("違う", final=False)
        await asyncio.gather(task, return_exceptions=True)
        await self.feed("違う", final=True)
        await self.step(9)
        self.assertEqual(self.r.waiting_for_continuation, "correction")
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL])
        self.assertEqual(self.judge.preparations, [])
        self.assertEqual(self.audio, [])
        detail = "コンビニではなくスーパーでした。"
        await self.feed(detail, item="detail", sequence=2)
        self.assertIsNone(self.r.waiting_for_continuation)
        await self.step(.21)
        await self.step(.26)
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, "違う" + detail])

    async def test_yes_without_preparing_reply_remains_normal_answer(self):
        # The previous robot question was fully played; this "yes" is an answer,
        # not a disposable acknowledgement of a reply still being prepared.
        self.core.messages.append(output("intro", "今日は出かけましたか？", kind="question"))
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        await self.r.played("intro")
        await asyncio.gather(*list(self.r.delivery_tasks))
        await self.feed("はい", item="answer", sequence=0)
        await self.resolve()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], ["はい"])
        self.assertEqual(self.judge.preparations, [])

    async def test_preparation_question_includes_last_actually_heard_robot_text(self):
        heard = "今日はどこかに出かけましたか？"
        self.core.messages.append(output("intro", heard, kind="question"))
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        await self.r.played("intro")
        await asyncio.gather(*list(self.r.delivery_tasks))
        release = await self.begin_first(block=True)
        await self.feed(REPETITION)
        await self.resolve()
        self.assertEqual(self.judge.preparations[-1]["last_heard_text"], heard)
        self.assertEqual(self.judge.preparations[-1]["original_text"], ORIGINAL)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual(len(self.core.inputs), 1)

    async def test_low_probability_retain_preserves_extra_by_bounded_fallback(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("retain", probability=.7)
        extra = "はい、それと明日の話もしたいです。"
        await self.feed(extra)
        await self.resolve()
        self.assertEqual(self.r._text("preparation"), extra)
        self.assertEqual(self.audio, [])
        await self.step(2.0)
        # The fallback promotes content, then ordinary EOT must still accept it.
        await self.resolve()
        await self.step(.26)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, extra])
        self.assertEqual(self.audio, [])

    async def test_recorded_sentence_tail_is_reviewed_before_cancelling_ready_audio(self):
        original = "うんえっとそうですね、まあいつもあんま変わらないってたまにはいいかなって"
        task = await self.prepare_tts(original=original)
        self.judge.result = preparation("revise", probability=.56)
        self.judge.result["preparation"]["confidence"] = .34
        self.judge.result["preparation"]["probabilities"] = {
            "retain": .29, "revise": .56, "uncertain": .15}
        await self.feed("感じですね。")
        await self.step(.81)
        self.speech.before_first.set()
        await self.wait_for(lambda: any(event["type"] == "speech_ready" for event in self.logs()))
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertEqual(self.core.deliveries, [])
        self.assertEqual(self.r._text("preparation"), "感じですね。")
        self.judge.result = preparation("retain", probability=.94)
        await self.step(.51)
        self.assertTrue(self.judge.preparations[-1]["review_ambiguous"])
        await self.step(.26)
        await asyncio.wait_for(task, 2)
        self.assertEqual([text for text, _ in self.core.inputs], [original])
        self.assertEqual(len(self.speech.calls), 1)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertEqual(self.core.deliveries, [])
        self.assertEqual(len(self.judge.preparations), 2)
        self.assertFalse(any(row["type"] == "preparation_promoted" for row in self.logs()))

    async def test_reviewed_correction_is_preserved_and_submitted_once(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("revise", probability=.56)
        await self.feed(CORRECTION)
        await self.step(.81)
        self.assertEqual(self.r._text("preparation"), CORRECTION)
        self.judge.result = preparation("revise", probability=.94)
        await self.step(.51)
        await self.step(.26)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])
        self.assertEqual(len(self.judge.preparations), 2)

    async def test_still_ambiguous_review_is_bounded_and_never_discards_content(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("revise", probability=.56)
        await self.feed(CORRECTION)
        for _ in range(42):
            await self.step(.1)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])
        self.assertEqual(len(self.judge.preparations), 2)

    async def test_review_failure_preserves_correction(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("revise", probability=.56)
        await self.feed(CORRECTION)
        await self.step(.81)
        self.judge.error = TimeoutError("offline")
        await self.step(.51)
        for _ in range(26):
            await self.step(.1)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])
        self.assertEqual(len(self.judge.preparations), 2)

    async def test_pending_semantic_review_cannot_lose_to_preparation_timeout(self):
        task = await self.prepare_tts()
        self.judge.result = preparation("revise", probability=.56)
        await self.feed("感じですね。")
        await self.step(.81)
        self.judge.gate = asyncio.Event()
        self.judge.result = preparation("retain")
        await self.step(.51, settle=False)
        await asyncio.sleep(0)
        await self.step(4, settle=False)
        self.assertEqual(self.r._text("preparation"), "感じですね。")
        self.assertEqual(self.core.deliveries, [])
        self.judge.gate.set()
        await self.r.judge_task
        await self.step(.26)
        self.speech.before_first.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(len(self.core.inputs), 1)
        self.assertEqual(len(self.speech.calls), 1)

    async def test_retain_with_unfinished_turn_does_not_discard_extra(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("retain", turn="short")
        extra = "はい、でもそれで"
        await self.feed(extra)
        await self.resolve()
        self.assertEqual(self.r._text("preparation"), extra)
        await self.step(2.0)
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, extra])
        self.assertEqual(self.audio, [])

    async def test_preparation_api_failure_preserves_extra_by_conservative_fallback(self):
        release = await self.begin_first(block=True)
        self.judge.error = TimeoutError("offline")
        await self.feed(CORRECTION)
        await self.resolve()
        self.assertEqual(self.r._text("preparation"), CORRECTION)
        await self.step(1.0)
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])
        self.assertEqual(self.audio, [])

    async def test_retained_segment_reopens_when_late_final_changes_meaning(self):
        release = await self.begin_first(block=True)
        await self.feed("はい")
        await self.resolve()
        self.assertEqual(self.r.segments["extra"]["lane"], "ignored_preparation")
        self.judge.result = preparation("revise")
        full_text = "はい、でもさっきのコンビニというのは間違いでした。"
        await self.feed(full_text)
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, full_text])
        self.assertEqual(self.audio, [])

    async def test_probability_at_threshold_retains_the_same_pcm(self):
        task = await self.prepare_tts()
        self.judge.result = preparation("retain", probability=.80)
        await self.feed(REPETITION)
        await self.resolve()
        self.speech.before_first.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(len(self.core.inputs), 1)
        self.assertEqual(len(self.speech.calls), 1)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertEqual(self.core.deliveries, [])

    async def test_probability_just_below_threshold_keeps_decision_pending(self):
        task = await self.prepare_tts()
        self.judge.result = preparation("retain", probability=.79)
        await self.feed(REPETITION)
        await self.resolve()
        self.speech.before_first.set()
        await self.wait_for(lambda: any(event["type"] == "speech_ready" for event in self.logs()))
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertEqual(self.r._text("preparation"), REPETITION)
        self.assertEqual(len(self.core.inputs), 1)






    async def test_audio_arriving_during_typed_interrupt_uses_normal_lane(self):
        first_tts = await self.prepare_tts()
        entered, release = asyncio.Event(), asyncio.Event()
        original_send = self.r.send_json

        async def block_interrupt(event):
            if event["type"] == "interrupt":
                entered.set()
                await release.wait()
            await original_send(event)

        self.r.send_json = block_interrupt
        typed_text = "さっきの話を訂正します。スーパーでした。"
        typed_task = asyncio.create_task(self.r.enqueue(typed_text))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            extra = "スーパーでパンも買いました。"
            await self.feed(extra)
            self.assertEqual(self.r.segments["extra"]["lane"], "normal")
            self.assertEqual(self.r.input_text, extra)
            self.assertIsNone(self.r.preparation_context)
            release.set()
            self.assertTrue(await asyncio.wait_for(typed_task, 2))
            await asyncio.gather(first_tts, return_exceptions=True)
            await asyncio.wait_for(self.r.jobs.join(), 2)
            await self.resolve()
            await asyncio.wait_for(self.r.jobs.join(), 2)
            self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, typed_text, extra])
            self.assertEqual(self.judge.preparations, [])
            self.assertEqual(self.audio, [])
        finally:
            release.set()
            await asyncio.gather(typed_task, return_exceptions=True)



    async def test_changed_preparation_context_promotes_old_rows_instead_of_reusing_judgment(self):
        release = await self.begin_first(block=True)
        await self.feed(REPETITION)
        old_snapshot = self.r._snapshot("preparation")
        old_id = self.r.segments["extra"]["context"]["id"]
        self.r._new_preparation("今日は図書館に行きました。")
        self.assertNotEqual(self.r.preparation_context["id"], old_id)
        self.assertFalse(self.r._current(old_snapshot))
        await self.step(.81)
        self.assertEqual(self.r.segments["extra"]["lane"], "normal")
        self.assertEqual(self.r.input_text, REPETITION)
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, REPETITION])
        self.assertEqual(self.judge.preparations, [])
        self.assertEqual(self.audio, [])

    async def pending(self, *, item="noise", sequence=2):
        await self.r._accept_transcript({"type": "session.input_transcript.pending",
                                        "item_id": item, "sequence": sequence})

    async def ready_with_sound_only_wait(self):
        task = await self.prepare_tts()
        self.client.pending_audio = True
        await self.pending()
        self.speech.before_first.set()
        await self.wait_for(lambda: self.r.noise_wait_started > 0)
        self.assertEqual(self.audio, [])
        return task


    async def test_old_preparation_row_is_promoted_after_reply_protection_ends(self):
        release = await self.begin_first(block=True)
        await self.feed(REPETITION)
        self.assertEqual(self.r.segments["extra"]["lane"], "preparation")
        self.r.preparation_context["protect_reply"] = False
        await self.step(.81)
        self.assertEqual(self.r.segments["extra"]["lane"], "normal")
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, REPETITION])
        self.assertEqual(self.judge.preparations, [])

    async def test_retain_result_survives_sound_and_empty_asr_without_duplicate_api_call(self):
        release = await self.begin_first(block=True)
        self.judge.gate = asyncio.Event()
        await self.feed(REPETITION)
        await self.step(.81, settle=False)
        await asyncio.wait_for(self.judge.started.wait(), 2)
        judge_task = self.r.judge_task
        await self.r.microphone(b"\x00\x10" * 480)
        await self.pending()
        self.judge.gate.set()
        await asyncio.wait_for(judge_task, 2)
        self.assertEqual(self.r.verdict["choice"], "retain")
        await self.step(.3)
        self.assertEqual(self.r._text("preparation"), REPETITION)
        await self.feed("", item="noise", sequence=2)
        await self.step(.51)
        self.assertFalse(self.r._text("preparation"))
        self.assertEqual(len(self.judge.preparations), 1)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual(len(self.core.inputs), 1)

    async def test_noise_only_wait_is_bounded_with_recent_empty_asr_evidence(self):
        task = await self.ready_with_sound_only_wait()
        started = self.r.noise_wait_started
        self.now = started+1.9
        await self.feed("", item="noise", sequence=2)
        await self.pending(item="late-noise", sequence=3)
        await self.r.microphone(b"\x00\x10" * 480)
        self.now = started+1.99
        await asyncio.sleep(.03)
        self.assertEqual(self.audio, [])
        self.now = started+2.01
        await asyncio.wait_for(task, 2)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertTrue(self.r.segments["late-noise"]["noise_bypassed"])
        self.assertEqual(len(self.core.inputs), 1)
        self.assertEqual(len(self.speech.calls), 1)
        self.assertEqual(sum(event["type"] == "noise_only_wait_released" for event in self.logs()), 1)

    async def test_no_empty_asr_evidence_keeps_slow_recognition_protected(self):
        task = await self.ready_with_sound_only_wait()
        self.now = self.r.noise_wait_started+3
        await asyncio.sleep(.03)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertFalse(self.r._noise_wait_can_release(self.now))

    async def test_stale_empty_evidence_does_not_release_sound_only_wait(self):
        task = await self.ready_with_sound_only_wait()
        started = self.r.noise_wait_started
        self.now = started+.1
        await self.feed("", item="noise", sequence=2)
        self.now = started+2.01
        await asyncio.sleep(.03)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertFalse(self.r._noise_wait_can_release(self.now))

    async def test_real_partial_words_remain_hard_barrier_despite_empty_noise_evidence(self):
        task = await self.ready_with_sound_only_wait()
        started = self.r.noise_wait_started
        await self.feed("はい、でも", final=False)
        self.now = started+2.1
        await self.feed("", item="noise", sequence=2)
        await asyncio.sleep(.03)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertFalse(self.r._noise_wait_can_release(self.now))
        self.assertEqual(self.r._text("preparation"), "はい、でも")

    async def test_transport_backlog_remains_hard_barrier_despite_empty_noise_evidence(self):
        task = await self.ready_with_sound_only_wait()
        self.now = self.r.noise_wait_started+2.1
        await self.feed("", item="noise", sequence=2)
        self.r.microphone_backlog_bytes = 48000
        await asyncio.sleep(.03)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertFalse(self.r._noise_wait_can_release(self.now))
        self.r.microphone_backlog_bytes = 0
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.audio)

    async def test_late_real_words_from_bypassed_empty_segment_can_interrupt_playback(self):
        task = await self.ready_with_sound_only_wait()
        self.now = self.r.noise_wait_started+2.1
        await self.feed("", item="noise", sequence=2)
        await self.pending(item="late", sequence=3)
        await asyncio.wait_for(task, 2)
        await self.feed(CORRECTION, final=False, item="late", sequence=3)
        self.assertEqual(self.r.segments["late"]["lane"], "candidate")
        await self.step(.21)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.input_text, CORRECTION)
        self.assertEqual(self.judge.preparations, [])
        self.client.pending_audio = False
        await self.feed(CORRECTION, item="late", sequence=3)
        await self.resolve()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])

    async def test_new_text_resets_empty_noise_evidence_even_after_text_is_removed(self):
        task = await self.ready_with_sound_only_wait()
        started = self.r.noise_wait_started
        self.now = started+2.1
        await self.feed("", item="noise", sequence=2)
        self.now += .1
        await self.feed("認識途中", final=False)
        self.now += .1
        await self.feed("", final=True)
        self.assertGreater(self.r.input_last_change, self.r.last_empty_asr_at)
        self.assertFalse(self.r._noise_wait_can_release(self.now))
        await asyncio.sleep(.03)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])

    async def test_final_repetition_is_judged_once_during_noise_then_retained_after_bounded_wait(self):
        release = await self.begin_first(block=True)
        revision = self.r.input_revision
        await self.feed(REPETITION)
        self.client.pending_audio = True
        await self.pending()
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(.31)  # Respect the previous turn's shared request interval.
        self.assertEqual(len(self.judge.preparations), 1)
        self.assertEqual(self.r.verdict["choice"], "retain")
        started = self.r.noise_turn_started
        self.now = started+1.9
        await self.feed("", item="noise", sequence=2)
        await self.pending(item="noise-next", sequence=3)
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(.11)
        self.assertFalse(self.r._text("preparation"))
        self.assertEqual(self.r.input_revision, revision)
        self.assertEqual(len(self.judge.preparations), 1)
        self.assertTrue(self.r.segments["noise-next"]["noise_bypassed"])
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual(len(self.core.inputs), 1)

    async def test_noise_without_empty_evidence_does_not_apply_or_repeat_retain_judgment(self):
        release = await self.begin_first(block=True)
        await self.feed(REPETITION)
        self.client.pending_audio = True
        await self.pending()
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(.31)
        self.assertEqual(len(self.judge.preparations), 1)
        for _ in range(4):
            self.now += .81
            await self.r.microphone(b"\x00\x10" * 480)
            await self.step()
        self.assertEqual(self.r._text("preparation"), REPETITION)
        self.assertEqual(len(self.judge.preparations), 1)
        self.assertEqual(self.audio, [])
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)

    async def test_uncertain_extra_is_preserved_despite_repeated_empty_noise(self):
        release = await self.begin_first(block=True)
        self.judge.result = preparation("uncertain")
        await self.feed(CORRECTION)
        self.client.pending_audio = True
        await self.pending()
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step(.31)
        started = self.r.noise_turn_started
        self.now = started+3.1
        await self.feed("", item="noise", sequence=2)
        await self.r.microphone(b"\x00\x10" * 480)
        await self.step()
        self.assertEqual(self.r.input_text, CORRECTION)
        self.assertFalse(self.r._text("preparation"))
        self.client.pending_audio = False
        await self.resolve()
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual([text for text, _ in self.core.inputs], [ORIGINAL, CORRECTION])

    async def test_old_empty_preparation_context_cannot_override_new_recognised_words(self):
        release = await self.begin_first(block=True)
        await self.pending(item="old-empty", sequence=2)
        old_id = self.r.preparation_context["id"]
        self.r._new_preparation(ORIGINAL)
        current_id = self.r.preparation_context["id"]
        self.assertNotEqual(old_id, current_id)
        await self.feed(REPETITION, item="new-words", sequence=3)
        self.assertEqual(self.r._snapshot("preparation")["context"]["id"], current_id)
        # The empty segment remains an ASR barrier; its obsolete context must
        # not cause the new recognised words to be promoted as a stale reply.
        await self.step(.81)
        self.assertEqual(self.r._text("preparation"), REPETITION)
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(len(self.judge.preparations), 1)
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)

    async def test_recorded_empty_asr_timeline_releases_ready_audio_after_two_seconds(self):
        # Offline timestamp replay, not playback or ASR of the user's recording.
        # Session a498...: ready 311.656 s, empty finals 312.297/313.563 s;
        # v0.2.6 did not start playing until the next empty final at 317.328 s.
        task = await self.prepare_tts()
        offset = 1000.0
        self.now = offset+311.656
        self.client.pending_audio = True
        await self.r.microphone(b"\x00\x10" * 480)
        self.speech.before_first.set()
        await self.wait_for(lambda: self.r.noise_wait_started > 0)
        for at, final, item, sequence in ((311.781, False, "recorded-1", 2),
                                          (312.297, True, "recorded-1", 2),
                                          (313.266, False, "recorded-2", 3),
                                          (313.563, True, "recorded-2", 3)):
            self.now = offset+at
            await self.r.microphone(b"\x00\x10" * 480)
            if final:
                await self.feed("", item=item, sequence=sequence)
            else:
                await self.pending(item=item, sequence=sequence)
            await asyncio.sleep(.025)
            self.assertEqual(self.audio, [])
        self.now = offset+313.657
        await self.r.microphone(b"\x00\x10" * 480)
        await asyncio.wait_for(task, 2)
        released = [event for event in self.logs() if event["type"] == "noise_only_wait_released"]
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0]["waited_ms"], 2001)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        self.assertEqual(len(self.core.inputs), 1)
        self.now = offset+316.969
        await self.pending(item="recorded-3", sequence=4)
        self.assertEqual(self.r.segments["recorded-3"]["lane"], "candidate")
        self.now = offset+317.328
        await self.feed("", item="recorded-3", sequence=4)
        self.assertEqual(len(self.speech.calls), 1)


if __name__ == "__main__":
    unittest.main()
