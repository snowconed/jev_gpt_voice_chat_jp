"""Offline sentence playback, exact heard prefixes, and Live context integration."""
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chat_config import ChatSettings
from chat_core import ChatCore
from speech_runtime import SpeechRuntime
from speech_segments import split_speech
from test_speech_runtime import Live


TEXT = "今日は公園でゆっくり過ごしました。  次は『川沿い』を歩いてみたいですね。\n"


class NoAPIResponder:
    def respond(self, history, settings):
        raise AssertionError("Segment playback tests must not generate a model response")

    def close(self):
        pass


class ControlledSpeech:
    def __init__(self, pieces):
        self.pieces = pieces
        self.calls = []
        self.remainder_entered = asyncio.Event()
        self.release_remainder = asyncio.Event()
        self.remainder_error = None
        self.closed_streams = []
        self.closes = 0
        self.pcm = b"\x20\x00" * 12000  # Half a second, long enough to reject early ACKs.

    async def stream(self, text):
        self.calls.append(text)
        try:
            if len(self.pieces) > 1 and text == self.pieces[1]:
                self.remainder_entered.set()
                await self.release_remainder.wait()
                if self.remainder_error:
                    raise self.remainder_error
            yield self.pcm
        finally:
            self.closed_streams.append(text)

    async def close(self):
        self.closes += 1


class ContextLive(Live):
    def __init__(self):
        super().__init__()
        self.thoughts = []

    async def think(self, content, delegation_id=None, *, event_id=None):
        self.thoughts.append({"content": content, "delegation_id": delegation_id, "event_id": event_id})


class SplitSpeechTests(unittest.TestCase):
    def test_exact_text_and_at_most_two_pieces_for_sentences_and_clauses(self):
        samples = [TEXT, "短いです。", "はい。" + "話" * 120,
                   "「引用した長い日本語の文章です！」  続きを省略しません。",
                   "単語" * 16 + "、\t後続にも 空白を残します。",
                   "価格は3.14ドルです。数字や英語のMr. Smithを壊しません。",
                   "段落\n" + "長い文" * 30 + "。次です。"]
        for text in samples:
            with self.subTest(text=text):
                pieces = split_speech(text)
                self.assertIn(len(pieces), (1, 2))
                self.assertEqual("".join(pieces), text)
                self.assertTrue(all(piece.strip() for piece in pieces))
        self.assertEqual(len(split_speech(TEXT)), 2)
        self.assertEqual(split_speech("短いです。"), ["短いです。"])


class SegmentedSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = Path(__file__).resolve().parents[1] / ".runtime" / "segmented_speech_tests"
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.now = time.monotonic()
        self.clock = patch("speech_runtime.time", SimpleNamespace(monotonic=lambda: self.now))
        self.clock.start()
        self.r = None
        self.core = None

    async def asyncTearDown(self):
        try:
            if self.r:
                await self.r.close()
                if self.r.live_context:
                    await self.r.live_context.close()
            if self.core:
                self.core.close()
        finally:
            self.clock.stop()
            self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root.resolve()))
            self.temp.cleanup()

    def make_runtime(self, text=TEXT, *, context=False):
        # Use the actual saved core: interruption assertions cover its history
        # contract instead of only checking a fake delivery call.
        self.core = ChatCore(Path(self.temp.name) / "chat", settings=ChatSettings(greeting=text),
                             responder=NoAPIResponder())
        state = self.core.create()
        self.sid = state["id"]
        self.message = self.core.pending(self.sid)[0]
        self.pieces = split_speech(self.message["text"])
        self.speech = ControlledSpeech(self.pieces)
        self.live = ContextLive() if context else Live()
        self.events, self.audio, self.wire = [], [], []

        async def send_json(event):
            self.events.append(event)
            self.wire.append(("json", event))

        async def send_bytes(pcm):
            self.audio.append(pcm)
            self.wire.append(("pcm", pcm))

        self.r = SpeechRuntime(self.core, self.sid, send_json, send_bytes,
                               Path(self.temp.name), client=self.live, speech=self.speech)
        self.r.ready = True
        return self.r

    async def offer(self):
        await self.r._offer_pending(self.r.input_revision, self.r.output_generation)
        return self.r.speech_task

    async def wait_until(self, condition):
        async with asyncio.timeout(2):
            while not condition():
                await asyncio.sleep(.001)

    def segment_id(self, index):
        return self.message["id"] if len(self.pieces) == 1 else f'{self.message["id"]}:s{index}'

    async def settle_deliveries(self):
        await asyncio.gather(*list(self.r.delivery_tasks))

    def delivery(self):
        return self.core.view(self.sid)["messages"][0]["delivery"]

    async def test_first_sentence_is_sent_before_blocked_remainder_with_exact_order(self):
        self.make_runtime()
        task = await self.offer()
        await self.speech.remainder_entered.wait()
        await self.wait_until(lambda: any(event["type"] == "playback_end" for event in self.events))
        self.assertFalse(task.done())
        self.assertEqual(self.speech.calls, self.pieces)
        self.assertEqual(self.audio, [self.speech.pcm])
        self.assertEqual([event["id"] for event in self.events if event["type"] == "playback"],
                         [self.segment_id(0)])
        self.assertEqual(self.delivery()["status"], "pending")
        self.speech.release_remainder.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual("".join(event["text"] for event in self.events
                                if event["type"] == "output_transcript"), self.message["text"])
        self.assertTrue(all(event["message_id"] == self.message["id"] for event in self.events
                            if event["type"] in ("playback", "output_transcript")))
        wire_types = [event["type"] if kind == "json" else "pcm" for kind, event in self.wire
                      if kind == "pcm" or event["type"] in ("playback", "output_transcript", "playback_end")]
        self.assertEqual(wire_types, ["playback", "output_transcript", "pcm", "playback_end"] * 2)
        self.assertEqual(self.audio, [self.speech.pcm] * 2)

    async def test_interruption_saves_only_the_acknowledged_exact_prefix_in_real_core(self):
        self.make_runtime()
        task = await self.offer()
        await self.wait_until(lambda: any(event["type"] == "playback_end" for event in self.events))
        self.now += 1
        await self.r.played(self.segment_id(0))
        self.assertEqual(self.delivery()["status"], "pending", "a prefix cannot complete the whole message")
        await self.r.interrupt()
        await asyncio.gather(task, return_exceptions=True)
        await self.settle_deliveries()
        self.assertEqual(self.delivery()["status"], "interrupted")
        self.assertEqual(self.delivery()["heard_text"], self.pieces[0])
        self.assertEqual(self.core._history(self.core._sessions[self.sid]),
                         [{"role": "assistant", "content": self.pieces[0]}])
        self.assertEqual(set(self.speech.closed_streams), set(self.pieces))

    async def test_early_out_of_order_duplicate_and_stale_acknowledgments_are_ignored(self):
        self.make_runtime()
        self.speech.release_remainder.set()
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        await self.r.played(self.segment_id(0))
        self.assertEqual(self.r.heard_prefixes[self.message["id"]], "")
        self.now += 2
        await self.r.played(self.segment_id(1))
        await self.r.played("old-message:s0")
        self.assertEqual(self.r.heard_prefixes[self.message["id"]], "")
        await self.r.played(self.segment_id(0))
        await self.r.played(self.segment_id(0))
        self.assertEqual(self.r.heard_prefixes[self.message["id"]], self.pieces[0])
        self.assertEqual(self.delivery()["status"], "pending")
        await self.r.played(self.segment_id(1))
        await self.settle_deliveries()
        before = self.delivery()
        await self.r.played(self.segment_id(1))
        await self.r.played(self.segment_id(0))
        self.assertEqual(self.delivery(), before)
        self.assertEqual(before["status"], "completed")
        self.assertEqual(before["heard_text"], self.message["text"])

    async def test_ack_received_before_a_segment_finishes_sending_is_ignored(self):
        self.make_runtime()
        entered, release = asyncio.Event(), asyncio.Event()
        original_send = self.r.send_bytes

        async def paused_send(pcm):
            await original_send(pcm)
            entered.set()
            await release.wait()

        self.r.send_bytes = paused_send
        task = await self.offer()
        await entered.wait()
        self.now += 10
        await self.r.played(self.segment_id(0))
        self.assertEqual(self.r.heard_prefixes[self.message["id"]], "")
        self.assertFalse(any(event["type"] == "playback_end" for event in self.events))
        await self.r.interrupt()
        await asyncio.gather(task, return_exceptions=True)
        await self.settle_deliveries()
        self.assertEqual(self.delivery()["heard_text"], "")
        release.set()

    async def test_remainder_failure_stops_playback_and_preserves_confirmed_first_sentence(self):
        self.make_runtime()
        self.speech.remainder_error = RuntimeError("private TTS provider detail")
        task = await self.offer()
        await self.wait_until(lambda: any(event["type"] == "playback_end" for event in self.events))
        self.now += 1
        await self.r.played(self.segment_id(0))
        self.speech.release_remainder.set()
        await asyncio.wait_for(task, 2)
        await self.settle_deliveries()
        self.assertTrue(self.r.finished.is_set())
        self.assertIsNone(self.r.playing)
        self.assertIsNone(self.r.approved)
        self.assertEqual(self.delivery()["status"], "interrupted")
        self.assertEqual(self.delivery()["heard_text"], self.pieces[0])
        self.assertTrue(any(event["type"] == "interrupt" for event in self.events))
        self.assertEqual(self.audio, [self.speech.pcm])
        self.assertEqual(set(self.speech.closed_streams), set(self.pieces))
        self.assertNotIn("private TTS", repr(self.events) + str(self.r.failure_message))

    async def test_single_segment_keeps_original_playback_id_and_completes_on_ack(self):
        self.make_runtime("短い挨拶です。")
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.speech.calls, [self.message["text"]])
        self.assertEqual([event["id"] for event in self.events if event["type"] == "playback"],
                         [self.message["id"]])
        self.now += 1
        await self.r.played(self.message["id"])
        await self.settle_deliveries()
        self.assertEqual(self.delivery()["status"], "completed")
        self.assertEqual(self.delivery()["heard_text"], self.message["text"])

    async def test_live_context_distinguishes_started_text_from_confirmed_prefix(self):
        self.make_runtime(context=True)
        task = await self.offer()
        await self.wait_until(lambda: any(event["type"] == "playback_end" for event in self.events))
        await self.r.live_context.flush()
        started = [json.loads(event["content"]) for event in self.live.thoughts]
        self.assertTrue(started)
        self.assertTrue(all(row["status"] == "started" and row["text_role"] == "unconfirmed_playback_text"
                            for row in started))
        self.assertEqual("".join(row["text"] for row in started), self.message["text"])
        # A Live command acknowledgment confirms receipt of context only.
        for event in self.live.thoughts:
            self.assertTrue(self.r.live_context.handle_event(
                {"type": "session.thinking.appended", "client_event_id": event["event_id"]}))
        self.assertEqual(self.r.heard_prefixes[self.message["id"]], "")
        self.assertEqual(self.delivery()["status"], "pending")
        self.now += 1
        await self.r.played(self.segment_id(0))
        await self.r.interrupt()
        await asyncio.gather(task, return_exceptions=True)
        await self.settle_deliveries()
        await self.r.live_context.flush()
        rows = [json.loads(event["content"]) for event in self.live.thoughts]
        self.assertEqual([row["status"] for row in rows if row["part"] == 1],
                         ["started", "progress", "interrupted"])
        text_by_status = defaultdict(str)
        for row in rows:
            text_by_status[row["status"]] += row["text"]
            if row["status"] != "started":
                self.assertEqual(row["text_role"], "confirmed_played_prefix")
        self.assertEqual(text_by_status["progress"], self.pieces[0])
        self.assertEqual(text_by_status["interrupted"], self.pieces[0])
        self.assertNotIn("completed", text_by_status)

    async def test_cancelling_close_during_context_flush_does_not_abandon_connections(self):
        self.make_runtime(context=True)
        context_send_entered, context_flush_entered = asyncio.Event(), asyncio.Event()
        blocked_context = asyncio.Event()

        async def blocked_think(*args, **kwargs):
            context_send_entered.set()
            await blocked_context.wait()

        self.live.think = blocked_think
        original_flush = self.r.live_context.flush

        async def observed_flush():
            context_flush_entered.set()
            await original_flush()

        self.r.live_context.flush = observed_flush
        speech_task = await self.offer()
        await asyncio.wait_for(context_send_entered.wait(), 2)
        await self.wait_until(lambda: any(event["type"] == "playback_end" for event in self.events))
        first_close = asyncio.create_task(self.r.close())
        await asyncio.wait_for(context_flush_entered.wait(), 2)
        first_close.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_close
        # The sender intentionally remains blocked. The bounded shutdown must
        # cancel it itself, and every later caller must join that same cleanup.
        await asyncio.wait_for(self.r.close(), 2)
        await asyncio.wait_for(self.r.close(), 2)
        self.assertTrue(self.r.closed)
        self.assertTrue(self.r.close_done.is_set())
        self.assertTrue(self.r.speech_close_done.is_set())
        self.assertTrue(self.r.live_context.closed)
        self.assertTrue(self.r.live_context.task.done())
        self.assertTrue(speech_task.done())
        self.assertEqual(self.live.disconnects, 1)
        self.assertEqual(self.speech.closes, 1)
        self.assertEqual(set(self.speech.closed_streams), set(self.pieces))
        self.assertEqual(self.delivery()["status"], "interrupted")
        self.assertEqual(self.delivery()["heard_text"], "")

    async def interruption_with_failed_notification(self, *, cancel):
        self.make_runtime()
        speech_task = await self.offer()
        await self.wait_until(lambda: any(event["type"] == "playback_end" for event in self.events))
        self.now += 1
        await self.r.played(self.segment_id(0))
        entered, release = asyncio.Event(), asyncio.Event()
        original_send = self.r.send_json
        first = True

        async def failed_stop(event):
            nonlocal first
            if event["type"] == "interrupt" and first:
                first = False
                entered.set()
                await release.wait()
                raise ConnectionResetError("the browser disconnected")
            await original_send(event)

        self.r.send_json = failed_stop
        interruption = asyncio.create_task(self.r.interrupt())
        await asyncio.wait_for(entered.wait(), 2)
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.delivery()["status"], "pending",
                         "delivery must not precede the attempted physical stop")
        if cancel:
            interruption.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await interruption
        else:
            release.set()
            with self.assertRaises(ConnectionResetError):
                await interruption
        await asyncio.wait_for(self.r.close(), 2)
        self.assertTrue(speech_task.done())
        self.assertTrue(self.r.closed)
        self.assertEqual(self.delivery()["status"], "interrupted")
        self.assertEqual(self.delivery()["heard_text"], self.pieces[0])
        self.assertEqual(self.core._history(self.core._sessions[self.sid]),
                         [{"role": "assistant", "content": self.pieces[0]}])

    async def test_cancelled_interrupt_notification_still_saves_confirmed_prefix_on_close(self):
        await self.interruption_with_failed_notification(cancel=True)

    async def test_failed_interrupt_notification_still_saves_confirmed_prefix_on_close(self):
        await self.interruption_with_failed_notification(cancel=False)


if __name__ == "__main__":
    unittest.main()
