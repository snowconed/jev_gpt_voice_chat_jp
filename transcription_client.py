"""Japanese streaming ASR only; application/Jev owns conversational turns.

An ASR segment is closed after 800 ms of acoustic silence, but this is NOT a
user-turn commit. Several segments can belong to one contribution. Item IDs
and sequence numbers let the runtime replace interim text with final text.
"""
from __future__ import annotations

import asyncio
import base64
import os
import time

import aiohttp
from websocket_send import send_serialized, send_snapshot

from voice_runtime import rms

MODEL = "gpt-live-transcribe"
RATE = 24000
ENDPOINT = "wss://api.openai.com/v1/realtime?intent=transcription"


class TranscriptionClient:
    def __init__(self, segment_silence=.8):
        self.segment_silence = segment_silence
        self.http = self.ws = self.session_id = None
        self.closing = False
        self.finalized = asyncio.Event()
        self.usage = {"transcription_items": []}
        self._send_lock = asyncio.Lock()
        self._pending_sends = {}
        self._has_voice = False
        self._silence = 0.0
        self._buffer_seconds = 0.0
        self._items = {}
        self._next_sequence = 0
        self._uncommitted_items = set()
        self._committed_items = set()
        self._unbound_commits = 0
        self._completed_items = set()
        self.transport_metrics = {"audio_frames_sent": 0, "audio_bytes_sent": 0,
                                  "max_audio_send_ms": 0.0, "input_transcript_chars": 0,
                                  "received_events": {}, "send_timeouts": 0,
                                  "asr_segments_committed": 0}

    async def connect(self):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=20))
        try:
            self.ws = await self.http.ws_connect(ENDPOINT, headers={"Authorization": "Bearer " + key},
                                                  heartbeat=20, max_msg_size=4*1024*1024, compress=15)
            await self.send({"type": "session.update", "session": {
                "type": "transcription", "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": RATE},
                    "transcription": {"model": MODEL, "languages": ["ja"], "delay": "low"},
                    "turn_detection": None}}}})
            async with asyncio.timeout(25):
                while True:
                    event = await self.ws.receive_json()
                    if event.get("type") in {"session.updated", "transcription_session.updated"}:
                        self.session_id = event.get("session", {}).get("id")
                        return event
                    if event.get("type") == "error":
                        raise RuntimeError("文字起こしサービスが開始を受け付けませんでした。")
        except BaseException:
            await self.disconnect()
            raise

    @property
    def pending_audio(self):
        """Speech without a final ASR result blocks a conversational commit."""
        return self._has_voice or bool(self._uncommitted_items) or self._commits_inflight > 0

    @property
    def _commits_inflight(self):
        return self._unbound_commits + len(self._committed_items - self._completed_items)

    def _buffer_action(self):
        # ASR can recognize quiet speech below the local energy threshold.
        # Its actual text is evidence too; never clear that buffer as silence.
        has_content = self._has_voice or bool(self._uncommitted_items)
        if has_content and (self._silence + 1e-9 >= self.segment_silence
                            or self._buffer_seconds >= 30):
            return "commit"
        if not has_content and self._buffer_seconds >= 15:
            return "clear"
        return None

    def _reserve_commit(self):
        # Make the association before send yields: a delta/final can arrive
        # before send_json completes or before the committed acknowledgement.
        if self._uncommitted_items:
            self._committed_items.update(self._uncommitted_items)
            self._uncommitted_items.clear()
        else:
            self._unbound_commits += 1

    def _bind_commit(self, item_id):
        if item_id not in self._committed_items:
            if self._unbound_commits:
                self._unbound_commits -= 1
            self._committed_items.add(item_id)
        self._uncommitted_items.discard(item_id)

    async def send(self, event):
        await send_serialized(self, event,
            "文字起こしサービスへの送信が5秒間完了しませんでした。新しい会話を開始してください。")

    def transport_snapshot(self):
        return send_snapshot(self)

    async def audio(self, pcm):
        if self.closing or not pcm:
            return
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > RATE * 2:
            raise ValueError("マイク音声の形式または長さが不正です。")
        started = time.perf_counter()
        try:
            await self._audio(pcm)
        finally:
            self.transport_metrics["max_audio_send_ms"] = max(self.transport_metrics["max_audio_send_ms"],
                round((time.perf_counter()-started)*1000, 3))

    async def _audio(self, pcm):
        # Commit at actual frame boundaries, even when one transport batch
        # includes both the end of a pause and the beginning of new speech.
        pending = bytearray()
        for offset in range(0, len(pcm), 960):
            frame = pcm[offset:offset+960]
            seconds = len(frame) / 48000
            self._buffer_seconds += seconds
            if rms(frame) >= 560:
                self._has_voice = True
                self._silence = 0.0
            else:
                self._silence += seconds
            pending.extend(frame)
            # Bound silence-only buffers and very long unbroken speech. These
            # are ASR housekeeping events, never turn-taking decisions.
            if self._buffer_action():
                await self._append(bytes(pending))
                pending.clear()
                # The append can yield to the receive loop. A newly recognized
                # quiet word must invalidate the earlier silence-only clear.
                action = self._buffer_action()
                if action:
                    if action == "commit":
                        self._reserve_commit()
                    self._has_voice = False
                    self._silence = self._buffer_seconds = 0.0
                    await self.send({"type": "input_audio_buffer." + action})
                if action == "commit":
                    self.transport_metrics["asr_segments_committed"] += 1
        if pending:
            await self._append(bytes(pending))

    async def _append(self, pcm):
        await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        self.transport_metrics["audio_frames_sent"] += 1
        self.transport_metrics["audio_bytes_sent"] += len(pcm)

    def _item(self, item_id):
        if item_id not in self._items:
            if len(self._items) >= 1000:
                raise ValueError("文字起こしの区間数が上限に達しました。")
            self._items[item_id] = {"text": "", "sequence": self._next_sequence}
            self._next_sequence += 1
        return self._items[item_id]

    def normalize_event(self, event):
        typ = event.get("type", "")
        item_id = event.get("item_id")
        if typ == "input_audio_buffer.committed" and item_id:
            item = self._item(item_id)
            if item_id in self._completed_items:
                return None
            self._bind_commit(item_id)
            return {"type": "session.input_transcript.pending", "item_id": item_id,
                    "sequence": item["sequence"], "segment_text": item["text"]}
        if typ in {"conversation.item.input_audio_transcription.delta",
                   "conversation.item.input_audio_transcription.completed"}:
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("文字起こしの区間IDがありません。")
            item = self._item(item_id)
            completed = typ.endswith(".completed")
            value = event.get("transcript" if completed else "delta", "")
            if not isinstance(value, str):
                raise ValueError("文字起こしの形式が不正です。")
            # A delayed acknowledgement/delta for a finished item must not
            # reopen the segment or consume a different commit reservation.
            if item_id in self._completed_items:
                return None
            if item_id not in self._committed_items and self._unbound_commits:
                self._bind_commit(item_id)
            item["text"] = value if completed else item["text"] + value
            if len(item["text"]) > 12000:
                raise ValueError("文字起こしの長さが上限に達しました。")
            if completed and event.get("usage"):
                self.usage["transcription_items"].append(event["usage"])
            if completed:
                self._completed_items.add(item_id)
                self._uncommitted_items.discard(item_id)
            if not completed:
                self.transport_metrics["input_transcript_chars"] += len(value)
                if value.strip() and item_id not in self._committed_items:
                    self._uncommitted_items.add(item_id)
                    # Low-volume words can arrive after a long quiet interval.
                    # Give continuing interim text the same 800 ms pause budget.
                    self._silence = 0.0
                if not item["text"].strip():
                    # Do not create an unfinished runtime row for an empty
                    # draft that silence housekeeping is allowed to clear.
                    return None
            return {"type": "session.input_transcript." + ("completed" if completed else "delta"),
                    "item_id": item_id, "sequence": item["sequence"], "segment_text": item["text"],
                    **({"text": value} if completed else {"delta": value})}
        if typ == "error" or typ == "conversation.item.input_audio_transcription.failed":
            return {"type": "session.error", "error": {"code": "transcription_failed"}}
        return None

    async def events(self):
        async for message in self.ws:
            if message.type == aiohttp.WSMsgType.TEXT:
                event = message.json()
                typ = event.get("type", "unknown")
                counts = self.transport_metrics["received_events"]
                counts[typ] = counts.get(typ, 0) + 1
                normalized = self.normalize_event(event)
                if normalized:
                    yield normalized
            elif message.type in {aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED}:
                return

    async def request_close(self):
        self.closing = True
        # Realtime transcription has no Live session.close/finalized event.
        # Close transport explicitly; do not invent usage-finalization proof.
        if self.ws and not self.ws.closed:
            await self.ws.close()

    async def disconnect(self):
        self.closing = True
        if self.ws:
            await self.ws.close()
        if self.http:
            await self.http.close()
