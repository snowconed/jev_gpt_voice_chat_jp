"""Offline buffered speech, concurrent input, delivery and Live-muting tests."""
import asyncio
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from speech_runtime import SpeechRuntime, SPEECH_MODEL


HERE = Path(__file__).resolve().parents[1]
TEXT = "買い物リストを確認できて、ほっとしたんだ。"


class Core:
    def __init__(self):
        self.messages = [{"id": "approved1", "kind": "reply", "text": TEXT}]
        self.deliveries = []

    def pending(self, sid):
        return deepcopy(self.messages)

    def view(self, sid):
        return {"id": sid, "messages": []}

    def delivery(self, sid, mid, status, heard_text=""):
        self.deliveries.append((mid, status, heard_text))
        self.messages = [m for m in self.messages if m["id"] != mid]


class Live:
    session_id = "offline-live"
    usage = None

    def __init__(self):
        self.finalized = asyncio.Event()
        self.queue = asyncio.Queue()
        self.pcm = []
        self.instructions = []
        self.disconnects = 0

    async def connect(self):
        pass

    async def audio(self, pcm):
        self.pcm.append(pcm)

    async def instruct(self, text):
        self.instructions.append(text)

    async def approve(self, _text):
        raise AssertionError("Live must not receive approved text for reading")

    async def events(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event

    async def request_close(self):
        self.finalized.set()
        await self.queue.put(None)

    async def disconnect(self):
        self.disconnects += 1


class Speech:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.before_first = None
        self.after_first = None
        self.paused = asyncio.Event()
        self.error = None
        self.chunks = [b"\x10\x00" * 960, b"\x20\x00" * 960]
        self.streams_closed = 0
        self.closes = 0

    async def stream(self, text):
        self.calls.append(text)
        self.started.set()
        try:
            if self.before_first:
                await self.before_first.wait()
            if self.error:
                raise self.error
            for index, pcm in enumerate(self.chunks):
                yield pcm
                if index == 0 and self.after_first:
                    self.paused.set()
                    await self.after_first.wait()
        finally:
            self.streams_closed += 1

    async def close(self):
        self.closes += 1


class SpeechRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = (HERE / ".runtime" / "speech_tests").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.core, self.live, self.speech = Core(), Live(), Speech()
        self.events, self.audio = [], []
        self.audio_seen = asyncio.Event()
        self.transcript_seen = asyncio.Event()

        async def send_json(event):
            self.events.append(event)
            if event["type"] == "input_transcript":
                self.transcript_seen.set()

        async def send_bytes(pcm):
            self.audio.append(pcm)
            self.audio_seen.set()

        self.r = SpeechRuntime(self.core, "a" * 32, send_json, send_bytes,
                               Path(self.temp.name), client=self.live, speech=self.speech)
        self.r.ready = True

    async def asyncTearDown(self):
        await self.r.close()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root))
        self.temp.cleanup()

    async def offer(self):
        await self.r._offer_pending(self.r.input_revision, self.r.output_generation)
        return self.r.speech_task

    async def settle_deliveries(self):
        await asyncio.gather(*list(self.r.delivery_tasks))

    async def test_all_audio_is_buffered_before_playback_and_delivery_waits_for_ack(self):
        self.speech.after_first = asyncio.Event()
        task = await self.offer()
        await asyncio.wait_for(self.speech.paused.wait(), 2)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        self.assertEqual(self.speech.calls, [TEXT])
        self.assertIsNotNone(self.r.approved)
        self.assertIsNone(self.r.playing)
        self.assertFalse(any(e["type"] == "playback" for e in self.events))
        self.assertFalse(any(e["type"] == "playback_end" for e in self.events))
        await self.r.played("approved1")
        self.assertEqual(self.core.deliveries, [])
        self.speech.after_first.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(b"".join(self.audio), b"".join(self.speech.chunks))
        playback = next(e for e in self.events if e["type"] == "playback")
        self.assertTrue(playback["buffered"])
        self.assertEqual(playback["audio_bytes"], sum(map(len, self.speech.chunks)))
        self.assertTrue(any(e["type"] == "playback_end" for e in self.events))
        self.assertEqual(self.core.deliveries, [])
        await self.r.played("approved1")
        await self.settle_deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])

    async def test_recent_rms_noise_does_not_block_pending_reply(self):
        self.r.input_last_voice = time.monotonic()
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.speech.calls, [TEXT])
        self.assertTrue(self.audio)

    async def test_rms_does_not_cancel_before_first_audio_but_text_does(self):
        self.speech.before_first = asyncio.Event()
        task = await self.offer()
        await asyncio.wait_for(self.speech.started.wait(), 2)
        await self.r.microphone(b"\x00\x04" * 4800)
        self.assertIsNotNone(self.r.approved)
        self.assertFalse(task.done())
        self.assertEqual(self.core.deliveries, [])
        await self.r.enqueue("先ほどの話を訂正します。")
        await asyncio.gather(task, return_exceptions=True)
        await self.settle_deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "cancelled", "")])
        self.assertEqual(self.speech.streams_closed, 1)
        self.assertEqual(self.audio, [])

    async def test_sustained_noise_before_and_during_playback_does_not_interrupt(self):
        # Reproduce the reported failure: an already noisy microphone has
        # accumulated well over 160 ms before the first TTS segment starts.
        noise = b"\x00\x04" * 4800
        for _ in range(6):
            await self.r.microphone(noise)
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        for _ in range(12):
            await self.r.microphone(noise)
        await self.settle_deliveries()
        self.assertEqual(self.live.pcm, [noise] * 18)
        self.assertEqual(self.r.input_text, "")
        self.assertIsNotNone(self.r.playing)
        self.assertEqual(self.core.deliveries, [])
        self.assertEqual(self.speech.streams_closed, 1)
        self.assertFalse(any(e["type"] == "interrupt" for e in self.events))
        await self.r.played("approved1")
        await self.settle_deliveries()
        self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])

    async def test_noise_then_recognized_correction_interrupts_and_preserves_input(self):
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        await self.r.microphone(b"\x00\x10" * 4800)
        self.assertIsNotNone(self.r.playing)
        await self.r._accept_live_transcript({"delta": "違います。少し待って。",
                                              "start_ms": 200, "end_ms": 900}, time.monotonic())
        await self.settle_deliveries()
        self.assertIsNone(self.r.playing)
        self.assertEqual(self.r.input_text, "違います。少し待って。")
        self.assertEqual(self.core.deliveries, [("approved1", "interrupted", "")])
        self.assertEqual(sum(e["type"] == "interrupt" for e in self.events), 1)
        records = [json.loads(line) for line in self.r.path.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()]
        interrupt = next(e for e in records if e["type"] == "recognized_input_interrupt")
        self.assertEqual(interrupt["source"], "live_transcript")

    async def test_non_speech_annotations_and_punctuation_do_not_stop_playback(self):
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        # Annotation fragments may arrive separately from loud breath audio.
        for delta in ("[bre", "athing]", "[sigh]", "[cough]", " 。、！？ "):
            await self.r.microphone(b"\x00\x08" * 4800)
            await self.r._accept_live_transcript({"delta": delta, "end_ms": 800}, time.monotonic())
        await self.r.check_output(time.monotonic())
        self.assertIsNotNone(self.r.playing)
        self.assertEqual(self.r.input_text, "")
        self.assertEqual(self.core.deliveries, [])
        self.assertFalse(any(e["type"] == "interrupt" for e in self.events))

    async def test_noise_while_remaining_segment_is_prepared_keeps_exact_reply(self):
        full_text = "今日は公園でゆっくり過ごしました。明日も天気がよければ散歩したいですね。"
        self.core.messages[0]["text"] = full_text
        tail_entered, release_tail = asyncio.Event(), asyncio.Event()
        seen_text = []

        async def segmented_stream(text):
            seen_text.append(text)
            if len(seen_text) == 2:
                tail_entered.set()
                await release_tail.wait()
            yield b"\x10\x00" * 960

        self.speech.stream = segmented_stream
        task = await self.offer()
        try:
            await asyncio.wait_for(tail_entered.wait(), 2)
            await asyncio.wait_for(self.audio_seen.wait(), 2)
            for _ in range(10):
                await self.r.microphone(b"\x00\x04" * 4800)
            self.assertIsNotNone(self.r.playing)
            self.assertFalse(task.done())
            self.assertFalse(any(e["type"] == "interrupt" for e in self.events))
            release_tail.set()
            await asyncio.wait_for(task, 2)
            self.assertEqual("".join(seen_text), full_text)
            self.assertEqual("".join(e["text"] for e in self.events if e["type"] == "output_transcript"), full_text)
            for playback in (e for e in self.events if e["type"] == "playback"):
                await self.r.played(playback["id"])
            await self.settle_deliveries()
            self.assertEqual(self.core.deliveries, [("approved1", "completed", "")])
        finally:
            release_tail.set()

    async def test_slow_tts_does_not_block_microphone_or_live_transcript(self):
        self.speech.after_first = asyncio.Event()
        task = await self.offer()
        await asyncio.wait_for(self.speech.paused.wait(), 2)
        self.r.tasks = set()
        self.r.receive_task = self.r.spawn(self.r.receive(), self.r.tasks, "receive")
        pcm = b"\x00\x04" * 480
        for _ in range(10):
            await asyncio.wait_for(self.r.microphone(pcm), .5)
        self.assertEqual(self.live.pcm, [pcm] * 10)
        self.assertIsNotNone(self.r.approved)
        self.assertFalse(task.done())
        self.assertEqual(self.audio, [])
        await self.live.queue.put({"type": "session.input_transcript.delta",
                                   "delta": "少し待って。", "end_ms": 200})
        await asyncio.wait_for(self.transcript_seen.wait(), 2)
        await asyncio.gather(task, return_exceptions=True)
        await self.settle_deliveries()
        self.assertEqual(self.r.input_text, "少し待って。")
        self.assertEqual(self.core.deliveries, [("approved1", "cancelled", "")])
        self.assertEqual(self.audio, [])

    async def test_buffering_uses_two_minute_limit_instead_of_legacy_thirty_seconds(self):
        self.speech.after_first = asyncio.Event()
        task = await self.offer()
        await asyncio.wait_for(self.speech.paused.wait(), 2)
        self.assertEqual(self.r.approved_timeout_seconds, 120)
        self.r.approved_at = time.monotonic() - 31
        tick = asyncio.create_task(self.r.tick())
        try:
            await asyncio.sleep(.15)
            self.assertFalse(self.r.finished.is_set())
            self.assertFalse(task.done())
            self.assertIsNotNone(self.r.approved)
        finally:
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)

    async def test_buffer_limit_fails_before_any_playback(self):
        with patch("speech_runtime.MAX_SPEECH_BYTES", 32):
            task = await self.offer()
            await asyncio.wait_for(task, 2)
        self.assertTrue(self.r.finished.is_set())
        self.assertEqual(self.audio, [])
        self.assertFalse(any(e["type"] == "playback" for e in self.events))

    async def test_generation_timeout_is_explicit_and_releases_no_partial_audio(self):
        self.speech.after_first = asyncio.Event()
        with patch("speech_runtime.SPEECH_TIMEOUT_SECONDS", .04):
            task = await self.offer()
            await asyncio.wait_for(task, 2)
        self.assertTrue(self.r.finished.is_set())
        self.assertEqual(self.audio, [])
        self.assertEqual(self.speech.streams_closed, 1)

    async def test_interrupt_during_send_drops_remaining_chunks_and_end(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_send(pcm):
            entered.set()
            await release.wait()
            self.audio.append(pcm)

        self.r.send_bytes = delayed_send
        task = await self.offer()
        await asyncio.wait_for(entered.wait(), 2)
        await self.r.interrupt()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await self.settle_deliveries()
        self.assertEqual(self.audio, [])
        self.assertEqual(self.speech.streams_closed, 1)
        self.assertFalse(any(e["type"] == "playback_end" for e in self.events))

    async def test_new_input_while_pending_lookup_waits_never_starts_tts(self):
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def blocked_pending(_sid):
            snapshot = deepcopy(self.core.messages)
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(3):
                raise TimeoutError("test barrier")
            return snapshot

        self.core.pending = blocked_pending
        task = asyncio.create_task(self.r._offer_pending(0, 0))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await self.r.enqueue("新しい話です。")
        finally:
            release.set()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.speech.calls, [])
        self.assertEqual(self.audio, [])

    async def test_tts_failure_ends_with_safe_explicit_error(self):
        self.speech.error = RuntimeError("private upstream error")
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        await self.settle_deliveries()
        self.assertTrue(self.r.finished.is_set())
        self.assertIn("読み上げ音声", self.r.failure_message)
        self.assertNotIn("private", self.r.failure_message)
        self.assertEqual(self.core.deliveries, [("approved1", "cancelled", "")])
        self.assertEqual(self.audio, [])

    async def test_empty_tts_is_explicit_failure(self):
        self.speech.chunks = []
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        self.assertTrue(self.r.finished.is_set())
        self.assertFalse(any(e["type"] == "playback_end" for e in self.events))

    async def test_live_audio_and_substantive_live_text_are_always_muted(self):
        await self.live.queue.put({"type": "session.output_audio.delta",
                                   "delta": base64.b64encode(b"\x00\x10" * 480).decode()})
        await self.live.queue.put({"type": "session.output_transcript.delta", "delta": "私も旅行しました。"})
        await self.live.queue.put(None)
        await self.r.receive()
        self.assertEqual(self.audio, [])
        self.assertEqual(self.speech.calls, [])
        self.assertFalse(any(e["type"] == "output_transcript" for e in self.events))

    async def test_allowed_backchannel_uses_same_tts_without_fact_commit(self):
        self.core.messages = []
        self.r.live_output_text = "うん。"
        self.r.live_output_last_text = self.r.live_output_last_voice = time.monotonic()-1
        await self.r.check_output(time.monotonic())
        task = self.r.speech_task
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.speech.calls, ["うん。"])
        mid = self.r.playing["id"]
        await self.r.played(mid)
        await self.settle_deliveries()
        self.assertEqual(self.core.deliveries, [])

    async def test_main_reply_takes_priority_over_backchannel(self):
        self.r.live_output_text = "なるほど。"
        self.r.live_output_last_text = self.r.live_output_last_voice = time.monotonic()-1
        await self.r.check_output(time.monotonic())
        self.assertEqual(self.speech.calls, [])
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.speech.calls, [TEXT])

    async def test_metrics_capture_first_audio_source_and_total_bytes(self):
        task = await self.offer()
        await asyncio.wait_for(task, 2)
        records = [json.loads(line) for line in self.r.path.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()]
        first = next(e for e in records if e["type"] == "output_approved")
        ready = next(e for e in records if e["type"] == "speech_ready")
        final = next(e for e in records if e["type"] == "speech_complete")
        self.assertEqual(first["source"], "tts")
        self.assertEqual(first["speech_model"], SPEECH_MODEL)
        self.assertGreaterEqual(first["first_audio_ms"], 0)
        self.assertEqual(final["audio_bytes"], sum(map(len, self.speech.chunks)))
        self.assertGreaterEqual(ready["ready_ms"], first["first_audio_ms"])
        self.assertGreaterEqual(final["sent_ms"], ready["ready_ms"])

    async def test_close_cancels_stream_and_closes_both_clients_once(self):
        self.speech.after_first = asyncio.Event()
        task = await self.offer()
        await asyncio.wait_for(self.speech.paused.wait(), 2)
        await self.r.close()
        await self.r.close()
        self.assertTrue(task.done())
        self.assertEqual(self.speech.streams_closed, 1)
        self.assertEqual(self.speech.closes, 1)
        self.assertEqual(self.live.disconnects, 1)
        self.assertEqual(self.core.deliveries, [("approved1", "cancelled", "")])

    async def test_close_saves_only_safe_speech_request_metadata(self):
        self.speech.calls = [{"model": SPEECH_MODEL, "http_status": 429,
                              "request_id": "req_offline", "status": "http_error",
                              "input_chars": 20, "usage_supported": False,
                              "input": "private text", "api_key": "private key"}]
        await self.r.close()
        text = (self.r.path / "tts_api_calls.json").read_text(encoding="utf-8")
        self.assertNotIn("private", text)
        saved = json.loads(text)
        self.assertEqual(saved[0]["http_status"], 429)
        self.assertEqual(saved[0]["request_id"], "req_offline")
        self.assertFalse(saved[0]["usage_supported"])


if __name__ == "__main__":
    unittest.main()
