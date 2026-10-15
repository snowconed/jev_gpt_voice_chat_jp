"""Backend text to buffered TTS; GPT-Live is used only for hearing/turn cues."""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import time
import uuid

from speech_client import SpeechClient, MODEL as SPEECH_MODEL, VOICE as SPEECH_VOICE
from input_transcript import InputTranscriptFilter
from interruption_intent import interruption_kind
from speech_segments import split_speech
from voice_runtime import VoiceRuntime, BACKCHANNELS, normalized, rms

SPEECH_TIMEOUT_SECONDS = 120
MAX_SPEECH_BYTES = 48000 * 120
REPLY_ACK_STABLE_SECONDS = .75
REPLY_ACK_MAX_HOLD_SECONDS = 2.0
REPLY_ACK_QUIET_SECONDS = .55
STANDALONE_PUNCTUATION = frozenset("。、，．,.!?！？")
REPLY_PREPARATION_ACKS = frozenset({
    "はい", "ええ", "うん", "なるほど", "そうですね", "そうなんです", "そうです",
    "そうなんですね", "そうですか", "わかりました", "わかった",
})


class SpeechRuntime(VoiceRuntime):
    """Preserve backend decisions and delivery ACKs without Live read-aloud waits.

All Live audio is discarded. Only approved core text, or an isolated permitted
Live backchannel, is sent to the one TTS voice. No speech duration is converted
into a guessed heard-text prefix.
"""

    def __init__(self, core, session_id, send_json, send_bytes, data_dir: Path,
                 client=None, speech=None):
        super().__init__(core, session_id, send_json, send_bytes, data_dir, client=client)
        self.speech = speech if speech is not None else SpeechClient()
        self.speech_model = getattr(self.speech, "model", SPEECH_MODEL)
        self.speech_voice = getattr(self.speech, "voice", SPEECH_VOICE)
        self.speech_tasks = set()
        self.speech_task = None
        self.speech_closing = False
        self.speech_close_done = asyncio.Event()
        self._speech_shutdown_task = None
        self.interrupting = False
        self.live_output_text = ""
        self.live_output_last_text = 0.0
        self.live_output_last_voice = 0.0
        self.last_backchannel = 0.0
        self.approved_timeout_seconds = SPEECH_TIMEOUT_SECONDS
        self.input_filter = InputTranscriptFilter()
        self.input_last_sound = 0.0
        self.input_interrupted_output = False
        self.waiting_for_continuation = None
        self.reply_input_text = ""
        self.reply_input_started = 0.0
        self.reply_input_changed = 0.0
        self.reply_input_end_ms = 0
        self.reply_input_playback_id = None
        self.filtered_input_end_ms = 0
        self.live_delegation_received_at = 0.0
        self.playback_segments = []
        self.heard_prefixes = {}
        self.live_context = None
        if callable(getattr(self.client, "think", None)):
            from live_context import LiveContextPublisher
            self.live_context = LiveContextPublisher(self.client,
                on_diagnostic=lambda data: self.log("live_context", **data))
        self._processing.configure(ack_stable=REPLY_ACK_STABLE_SECONDS,
                                   ack_quiet=REPLY_ACK_QUIET_SECONDS,
                                   ack_hold=REPLY_ACK_MAX_HOLD_SECONDS)

    async def start(self):
        await super().start()
        if not self.closed:
            self.log("speech_started", source="tts", speech_model=self.speech_model, voice=self.speech_voice)

    async def microphone(self, pcm):
        if self.closed or not self.ready:
            return
        if len(pcm) % 2 or len(pcm) > 48000:
            raise ValueError("音声データの形式が不正です。")
        self.input_samples += len(pcm) // 2
        # Energy is diagnostic activity, not proof of speech. It must neither
        # stop playback nor prevent a recognized turn from ending forever.
        # Live transcript/delegation events supply the speech timing instead.
        if rms(pcm) >= 560:
            self.input_last_sound = self.last_activity = time.monotonic()
        await self.client.audio(pcm)

    async def receive(self):
        try:
            async for event in self.client.events():
                typ = event.get("type", "")
                now = time.monotonic()
                if self.live_context and self.live_context.handle_event(event):
                    continue
                if typ == "session.input_transcript.delta":
                    await self._accept_live_transcript(event, now)
                elif typ == "session.delegation.created":
                    self.delegation = event
                    self.live_delegation_received_at = now
                    self.log(typ, offset_ms=event.get("offset_ms"), delegation=event.get("delegation"))
                elif typ == "session.output_transcript.delta":
                    self.live_output_text += event.get("delta", "")
                    self.live_output_last_text = now
                    candidate = normalized(self.live_output_text)
                    self.log(typ, delta=event.get("delta"), muted=True,
                             start_ms=event.get("start_ms"), end_ms=event.get("end_ms"))
                    if ((self.input_text or self.reply_input_text) and not self.busy and candidate
                            and not any(phrase.startswith(candidate) for phrase in BACKCHANNELS)):
                        self.native_response_boundary = now
                    if len(self.live_output_text) > 8000:
                        self.live_output_text = ""
                elif typ == "session.output_audio.delta":
                    # Observe only whether the muted Live utterance has ended.
                    pcm = base64.b64decode(event["delta"], validate=True)
                    if len(pcm) % 2:
                        raise ValueError("invalid Live PCM")
                    if rms(pcm) >= 100:
                        self.live_output_last_voice = now
                    # Never enqueue, forward, or mix these audio bytes.
                elif typ in ("error", "session.error"):
                    self.log("live_error", code=(event.get("error") or {}).get("code"))
                    self.failure_message = "音声認識サービスでエラーが発生しました。新しい会話を開始してください。"
                    break
                elif typ == "session.closed":
                    self.log("live_closed", usage=event.get("usage"))
                elif typ == "session.instructions.appended":
                    self.log(typ, client_event_id=event.get("client_event_id"))
        finally:
            if not self.closed:
                self.failure_message = self.failure_message or "音声認識の接続が終了しました。新しい会話を開始してください。"
                self.finished.set()

    async def _accept_live_transcript(self, event, now):
        delta = event.get("delta", "")
        if not isinstance(delta, str):
            raise ValueError("invalid input transcript")
        if not delta:
            return
        self.log("session.input_transcript.delta", delta=delta, raw=True,
                 start_ms=event.get("start_ms"), end_ms=event.get("end_ms"))
        self.filtered_input_end_ms = max(self.filtered_input_end_ms, event.get("end_ms", 0))
        text = self.input_filter.feed(delta, now)
        await self._accept_clean_input(text, now, self.filtered_input_end_ms)

    @staticmethod
    def _reply_ack_prefix(text):
        compact = normalized(text)
        return bool(compact) and any(ack.startswith(compact) for ack in REPLY_PREPARATION_ACKS)

    def _clear_reply_input(self):
        self.reply_input_text = ""
        self.reply_input_started = self.reply_input_changed = 0.0
        self.reply_input_end_ms = 0
        self.reply_input_playback_id = None

    async def _accept_clean_input(self, text, now, end_ms):
        punctuation_only = all(char.isspace() or char in STANDALONE_PUNCTUATION for char in text)
        if not text or (punctuation_only and not self.input_text and not self.reply_input_text):
            return
        self.input_last_voice = now
        replying = self.approved or self.playing
        if self.reply_input_text or (replying and self._reply_ack_prefix(text)):
            if not self.reply_input_text:
                self.reply_input_started = now
                self.reply_input_playback_id = self.playing["id"] if self.playing else None
            self.reply_input_text += text
            self.reply_input_changed = now
            self.reply_input_end_ms = max(self.reply_input_end_ms, end_ms)
            await self.send_json({"type": "input_transcript", "text": text})
            if not self._reply_ack_prefix(self.reply_input_text):
                await self._promote_reply_input(now, "continuation")
            return
        await self._accept_substantive_input(text, now, end_ms, caption=True)

    async def _accept_substantive_input(self, text, now, end_ms, *, caption):
        if not self.input_text:
            self.input_interrupted_output = bool(self.playing or self.approved or self.busy or not self.jobs.empty())
        self.cancel_generation()
        self.input_text += text
        if len(self.input_text) > 12000:
            raise ValueError("input transcript limit")
        self.input_end_ms = max(self.input_end_ms, end_ms)
        self.input_last_change = self.last_activity = now
        self.input_revision += 1
        if self.playing or self.approved:
            self.log("recognized_input_interrupt", id=(self.playing or self.approved)["id"],
                     source="live_transcript", input_chars=len(text))
            await self.interrupt()
        self._refresh_interruption_wait()
        if caption:
            await self.send_json({"type": "input_transcript", "text": text})

    def _refresh_interruption_wait(self):
        """Hold explicit short requests, preserving all words for continuation.

        A bare denial during an assistant reply asks it to yield. A bare 'no'
        after the reply is over may simply answer a question and stays input.
        This explicit-phrase rule does not classify arbitrary speech meaning.
        """
        kind = interruption_kind(self.input_text)
        waiting = kind if kind == "pause" or (kind == "correction" and self.input_interrupted_output) else None
        if waiting != self.waiting_for_continuation:
            self.waiting_for_continuation = waiting
            self.log("interruption_wait" if waiting else "interruption_continued",
                     kind=waiting, input_chars=len(self.input_text))
        return waiting

    async def enqueue(self, text, source="typed"):
        # A typed continuation must not leave a spoken 'wait' stranded in the
        # input buffer, where it would suppress the newly requested reply.
        held = self.input_text if source == "typed" and self.waiting_for_continuation else ""
        combined = held + "\n" + text.strip() if held and text.strip() else text
        accepted = await super().enqueue(combined, source)
        if accepted and source == "typed":
            if held:
                self.input_text = self.input_text[len(held):]
            self.input_interrupted_output = False
            self._refresh_interruption_wait()
        return accepted

    async def _promote_reply_input(self, now, reason):
        text, end_ms = self.reply_input_text, self.reply_input_end_ms
        self._clear_reply_input()
        if text:
            self.log("reply_preparation_input_promoted", reason=reason, text=text)
            await self._accept_substantive_input(text, now, end_ms, caption=False)

    async def _resolve_preparation_input(self, now):
        expired = bool(self.reply_input_text and now-self.reply_input_started >= REPLY_ACK_MAX_HOLD_SECONDS)
        text = self.input_filter.flush(now, force=expired)
        if text:
            await self._accept_clean_input(text, now, self.filtered_input_end_ms)
        if not self.reply_input_text:
            return
        if self.input_filter.has_pending:
            return
        compact = normalized(self.reply_input_text)
        stable = now - self.reply_input_changed >= REPLY_ACK_STABLE_SECONDS
        quiet = now - self.input_last_voice >= REPLY_ACK_QUIET_SECONDS
        native = ((self.native_response_boundary >= self.reply_input_started
                   and self.native_response_boundary > 0)
                  or (self.delegation and self.live_delegation_received_at >= self.reply_input_started
                      and self.delegation.get("offset_ms", 0) >= self.reply_input_end_ms - 350))
        if compact in REPLY_PREPARATION_ACKS and stable and (quiet or native):
            if self.reply_input_playback_id and not self.playing:
                # A 'yes' overlapping the end of a question may be its answer,
                # not a backchannel. Once playback ends, preserve that input.
                await self._promote_reply_input(now, "playback_finished")
            else:
                self.log("reply_preparation_ack", text=self.reply_input_text)
                self._clear_reply_input()
        elif now - self.reply_input_started >= REPLY_ACK_MAX_HOLD_SECONDS:
            await self._promote_reply_input(now, "ambiguous_or_not_quiet")

    async def _wait_preparation_input(self, message, revision, generation):
        while self.reply_input_text or self.input_filter.has_pending:
            await self._resolve_preparation_input(time.monotonic())
            if not self._speech_current(message, revision, generation):
                return False
            if self.reply_input_text or self.input_filter.has_pending:
                await asyncio.sleep(.02)
        return self._speech_current(message, revision, generation)

    async def _commit_live_input(self, now):
        if self.input_filter.has_pending or self.reply_input_text:
            return False
        if self._refresh_interruption_wait():
            return False
        accepted = await super()._commit_live_input(now)
        if accepted:
            self.input_interrupted_output = False
            self._refresh_interruption_wait()
        return accepted

    def _offer_current(self, revision, generation):
        # A fan/noisy microphone must not indefinitely suppress approved text.
        # Real transcript text, queued turns and explicit playback interruption
        # remain authoritative barriers.
        return (not self.closed and not self.speech_closing and not self.finished.is_set()
                and not self.busy and not self.interrupting and not self.draining
                and not self.approved and not self.playing and not self.input_text
                and not self.reply_input_text and not self.input_filter.has_pending
                and self.jobs.empty() and not self.delivery_tasks and not self.speech_tasks
                and revision == self.input_revision and generation == self.output_generation)

    async def _offer_pending(self, revision, generation):
        if not self._offer_current(revision, generation):
            return
        pending = await self._pending_if_idle()
        if not self._offer_current(revision, generation) or not pending:
            return
        self._start_speech(pending[0], revision, generation)

    def _start_speech(self, message, revision, generation):
        self.playback_segments = []
        self.heard_prefixes[message["id"]] = ""
        self.approved = message
        self.approved_at = time.monotonic()
        self.approval_revision = revision
        self.retry_count = 0
        self.log("approved_text_sent", id=message["id"], text=message["text"],
                 source="tts", speech_model=self.speech_model, voice=self.speech_voice)
        task = self.spawn(self._stream_speech(message, revision, generation), self.speech_tasks, "tts")
        self.speech_task = task

        def finished(done):
            if self.speech_task is done:
                self.speech_task = None
        task.add_done_callback(finished)

    def _speech_current(self, message, revision, generation):
        return (not self.closed and not self.speech_closing and not self.finished.is_set()
                and not self.interrupting and self.output_generation == generation
                and self.input_revision == revision and not self.input_text
                and (self.approved is message or self.playing is message))

    async def _wait_auxiliary_playback(self, message, revision, generation):
        """Optional cached turn feedback must finish before main audio starts."""
        return self._speech_current(message, revision, generation)

    async def _stream_speech(self, message, revision, generation):
        started = time.monotonic()
        audio_bytes = 0
        first_audio_ms = None
        ready_ms = None
        preparations = []

        async def synthesize(text):
            nonlocal audio_bytes, first_audio_ms
            audio = bytearray()
            stream = self.speech.stream(text)
            try:
                async with asyncio.timeout(SPEECH_TIMEOUT_SECONDS):
                    async for pcm in stream:
                        if not self._speech_current(message, revision, generation):
                            raise asyncio.CancelledError()
                        if not isinstance(pcm, bytes) or len(pcm) % 2:
                            raise ValueError("invalid TTS PCM")
                        if not pcm:
                            continue
                        if first_audio_ms is None:
                            first_audio_ms = round((time.monotonic()-started)*1000)
                            self.log("speech_first_audio", id=message["id"], source="tts",
                                     speech_model=self.speech_model, first_audio_ms=first_audio_ms)
                        audio_bytes += len(pcm)
                        if audio_bytes > MAX_SPEECH_BYTES:
                            raise ValueError("TTS audio limit exceeded")
                        audio.extend(pcm)
                if not audio:
                    raise ValueError("empty TTS output")
                return bytes(audio)
            finally:
                if hasattr(stream, "aclose"):
                    await stream.aclose()

        try:
            await self.status("返事を音声にしています。" if message["kind"] != "backchannel" else "お話を聞いています。")
            if not self._speech_current(message, revision, generation):
                return
            pieces = split_speech(message["text"])
            # Buffer each sentence completely to avoid gaps within it. Prepare
            # the remainder concurrently; no more than two requests per reply.
            preparations = [asyncio.create_task(synthesize(text)) for text in pieces]
            prefix_end = 0
            earliest_end = 0.0
            for index, (text, preparation) in enumerate(zip(pieces, preparations)):
                audio = await preparation
                if not self._speech_current(message, revision, generation):
                    return
                ready_ms = round((time.monotonic()-started)*1000)
                self.log("speech_ready", id=message["id"], source="tts", speech_model=self.speech_model,
                         first_audio_ms=first_audio_ms, ready_ms=ready_ms, segment=index,
                         audio_bytes=len(audio), audio_seconds=len(audio)/48000)
                if index == 0:
                    if not await self._wait_auxiliary_playback(message, revision, generation):
                        return
                    if not await self._wait_preparation_input(message, revision, generation):
                        return
                    self.playing, self.approved = message, None
                    self.playing_started = time.monotonic()
                    self.playing_seconds = float("inf")
                    self.log("output_approved", id=message["id"], kind=message["kind"],
                             text=message["text"], source="tts", speech_model=self.speech_model,
                             voice=self.speech_voice, first_audio_ms=first_audio_ms, ready_ms=ready_ms,
                             audio_bytes=len(audio), segments=len(pieces))
                    self._share_playback(message, "started", message["text"])
                playback_id = message["id"] if len(pieces) == 1 else f'{message["id"]}:s{index}'
                prefix_end += len(text)
                row = {"id": playback_id, "prefix_end": prefix_end,
                       "final": index == len(pieces)-1, "sent": False, "acked": False}
                self.playback_segments.append(row)
                await self.send_json({"type": "playback", "id": playback_id,
                                      "message_id": message["id"], "audio_bytes": len(audio),
                                      "audio_seconds": len(audio)/48000, "buffered": True,
                                      "segment_index": index, "final_segment": row["final"]})
                if not self._speech_current(message, revision, generation):
                    return
                await self.send_json({"type": "output_transcript", "text": text,
                                      "message_id": message["id"]})
                for offset in range(0, len(audio), 48000):
                    if not self._speech_current(message, revision, generation):
                        return
                    await self.send_bytes(audio[offset:offset+48000])
                if not self._speech_current(message, revision, generation):
                    return
                earliest_end = max(earliest_end, time.monotonic()) + len(audio)/48000
                row.update(sent=True, earliest_end=earliest_end)
                if row["final"]:
                    self.playing_seconds = earliest_end-self.playing_started
                await self.send_json({"type": "playback_end", "id": playback_id})
            self.log("speech_complete", id=message["id"], source="tts", speech_model=self.speech_model,
                     first_audio_ms=first_audio_ms, audio_bytes=audio_bytes,
                     ready_ms=ready_ms, sent_ms=round((time.monotonic()-started)*1000),
                     elapsed_ms=round((time.monotonic()-started)*1000))
            if self._speech_current(message, revision, generation):
                await self.status("お話ししています。途中で話しかけても大丈夫です。")
        except asyncio.CancelledError:
            self.log("speech_cancelled", id=message["id"], source="tts", speech_model=self.speech_model,
                     first_audio_ms=first_audio_ms, audio_bytes=audio_bytes)
            raise
        except Exception as error:
            self.log("speech_failed", id=message["id"], error_type=type(error).__name__,
                     source="tts", speech_model=self.speech_model, audio_bytes=audio_bytes)
            if not self.closed and self._speech_current(message, revision, generation):
                await self.interrupt()
                self.failure_message = "読み上げ音声を生成できなかったため停止しました。新しい会話を開始してください。"
                self.finished.set()
        finally:
            for task in preparations:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*preparations, return_exceptions=True)

    def _share_playback(self, message, status, text=""):
        if self.live_context and not self.live_context.closed:
            # Playback facts describe the shared conversation, including local
            # greetings and typed input with no Live delegation of their own.
            self.live_context.publish(message["id"], status, text, delegation_id=None)

    def _heard_progress(self, message, prefix, completed):
        """Hook for turn detectors; use acknowledged text, never duration guesses."""

    async def played(self, playback_id):
        message = self.playing
        if not message:
            return
        row = next((item for item in self.playback_segments if item["id"] == playback_id), None)
        if row is None:
            # Support existing direct/legacy playback and injected runtimes.
            if not self.playback_segments:
                await super().played(playback_id)
            return
        previous = self.playback_segments[:self.playback_segments.index(row)]
        if (row["acked"] or not row["sent"] or any(not item["acked"] for item in previous)
                or time.monotonic()+.15 < row["earliest_end"]):
            return
        row["acked"] = True
        prefix = message["text"][:row["prefix_end"]]
        self.heard_prefixes[message["id"]] = prefix
        self._heard_progress(message, prefix, row["final"])
        self.last_activity = time.monotonic()
        self.log("playback_segment_completed", id=message["id"], playback_id=playback_id,
                 heard_chars=len(prefix), final=row["final"])
        if row["final"]:
            self.playing = None
            self.spawn(self.deliver(message, "completed"), self.delivery_tasks, "delivery")
            self.log("playback_completed", id=message["id"])
        else:
            self._share_playback(message, "progress", prefix)

    async def deliver(self, message, status, heard_text=""):
        if message:
            if status == "interrupted" and not heard_text:
                heard_text = self.heard_prefixes.get(message["id"], "")
            shared = message["text"] if status == "completed" else heard_text
            self._share_playback(message, status, shared)
            self.heard_prefixes.pop(message["id"], None)
        await super().deliver(message, status, heard_text)

    def begin_drain(self):
        # Live audio is always muted; it need not finish/revoke a read-aloud.
        self.output_generation += 1
        self.draining = False
        self.retry_pending = False
        self.clear_output()
        self.live_output_text = ""
        self._clear_reply_input()
        if self.speech_task and self.speech_task is not asyncio.current_task():
            self.speech_task.cancel()

    async def interrupt(self):
        if self.interrupting:
            return
        self.interrupting = True
        try:
            await super().interrupt()
        finally:
            self.interrupting = False

    async def reject_output(self, reason):
        # The inherited tick calls this for an excessive first-byte wait.
        self.log("speech_timeout", reason=reason, source="tts", speech_model=self.speech_model)
        await self.interrupt()
        self.failure_message = "読み上げ音声の準備に時間がかかったため停止しました。新しい会話を開始してください。"
        self.finished.set()

    async def check_output(self, now):
        await self._resolve_preparation_input(now)
        # An isolated completed backchannel may use the same TTS voice. Core
        # replies always take precedence, and all substantive Live text is muted.
        if (not self.live_output_text or now-self.live_output_last_text < .5
                or now-self.live_output_last_voice < .3):
            return
        text, self.live_output_text = self.live_output_text, ""
        if (normalized(text) not in BACKCHANNELS or now-self.last_backchannel < 2
                or self.closed or self.speech_closing or self.finished.is_set()
                or self.approved or self.playing or self.speech_tasks or self.interrupting
                or self.delivery_tasks or self.busy or not self.jobs.empty()):
            return
        revision, generation = self.input_revision, self.output_generation
        pending = await self._pending_if_idle()
        if (pending is None or pending or self.busy or self.approved or self.playing or self.speech_tasks or self.interrupting
                or self.closed or self.speech_closing or self.finished.is_set()
                or revision != self.input_revision or generation != self.output_generation):
            return
        # Do not send a backchannel with a transcript still being assembled: the
        # main reply and its utterance boundary are the first implementation goal.
        if self.input_text:
            return
        self.last_backchannel = now
        self._start_speech({"id": "backchannel_"+uuid.uuid4().hex, "text": text,
                            "kind": "backchannel"}, revision, generation)

    async def close(self):
        # Caller cancellation must not abandon a metered connection or leave a
        # second close waiting forever. One owned shutdown survives all callers.
        if self._speech_shutdown_task is None:
            self.speech_closing = True
            self._speech_shutdown_task = asyncio.create_task(self._close_speech())
        await asyncio.shield(self._speech_shutdown_task)

    async def _close_speech(self):
        self._clear_reply_input()
        self.input_filter.reset()
        tasks = list(self.speech_tasks)
        for task in tasks:
            task.cancel()
        try:
            try:
                if self.live_context:
                    if self.playing:
                        self._share_playback(self.playing, "interrupted",
                                             self.heard_prefixes.get(self.playing["id"], ""))
                    elif self.approved:
                        self._share_playback(self.approved, "cancelled")
                    try:
                        await asyncio.wait_for(self.live_context.flush(), .25)
                    except TimeoutError:
                        self.log("live_context_flush_timeout")
            finally:
                try:
                    if self.live_context:
                        await self.live_context.close()
                finally:
                    await super().close()
        finally:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await self.speech.close()
                finally:
                    keys = {"model", "voice", "input_chars", "request_id", "http_status", "status",
                            "audio_bytes", "received_audio_bytes", "elapsed_seconds",
                            "usage_supported", "usage_note", "first_audio_ms", "chunk_count",
                            "max_chunk_gap_ms", "audio_seconds", "receive_elapsed_seconds",
                            "receive_to_audio_ratio", "queue_seconds", "query_seconds", "synthesis_seconds"}
                    records = [{key: value for key, value in record.items() if key in keys}
                               for record in getattr(self.speech, "calls", []) if isinstance(record, dict)]
                    (self.path / "tts_api_calls.json").write_text(
                        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            finally:
                self.speech_close_done.set()
