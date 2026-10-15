"""GPT-Live turn delegation, conservative content gate, and delivery bookkeeping."""
from __future__ import annotations

import array
import asyncio
import base64
from collections import deque
import json
import math
from pathlib import Path
import sys
import threading
import time
import unicodedata
import uuid

from live_client import LiveClient, MODEL, VOICE
from turn_progress import ProcessingProgress

BACKCHANNELS = {"うん", "はい", "なるほど"}
INPUT_TRANSPORT_BARRIER_BYTES = 19200  # 400 ms of PCM16, 24 kHz, mono.
_SUBMIT_DEFERRED = object()


def normalized(text: str) -> str:
    ignorable = set("。 、，．,.!?！？「」『』\"“”‘’")
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and c not in ignorable)


def rms(pcm: bytes) -> float:
    values = array.array("h", pcm)
    if sys.byteorder != "little":
        values.byteswap()
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


class VoiceRuntime:
    """One browser, one core session, one Live voice. No API key leaves Python.

    Playback is deliberately buffered until a whole authorized utterance or an
    allowed backchannel matches the Live transcript. This is a prototype content
    guard, not proof that a model transcript exactly describes its waveform.
    """

    CAPTURE_VOICE_RMS = 280

    def __init__(self, core, session_id: str, send_json, send_bytes, data_dir: Path,
                 client=None):
        self.core, self.sid = core, session_id
        self.send_json, self.send_bytes = send_json, send_bytes
        self.client = client or LiveClient()
        self.path = Path(data_dir) / "voice" / session_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.logfile = (self.path / "events.jsonl").open("a", encoding="utf-8")
        self.started = time.monotonic()
        self._processing = ProcessingProgress(self.started)
        self.closed = False
        self.ready = False
        self.finished = asyncio.Event()
        self.close_done = asyncio.Event()
        self.failure_message = None
        self.input_text = ""
        self.input_revision = 0
        self.input_last_change = 0.0
        self.input_last_voice = 0.0
        self.input_loud_seconds = 0.0
        self.input_samples = 0
        self.microphone_backlog_bytes = 0
        self.input_end_ms = 0
        self.delegation = None
        self.native_response_boundary = 0.0
        self.jobs = asyncio.Queue(maxsize=16)
        self._input_job_sources = {}
        self.busy = False
        self._active_cancel = None
        self.approved = None
        self.approved_at = 0.0
        self.approved_timeout_seconds = 30.0
        self.approval_revision = 0
        self.retry_count = 0
        self.output_text = ""
        self.output_audio = bytearray()
        self.output_preroll = deque(maxlen=2)
        self.output_last_voice = 0.0
        self.output_last_text = 0.0
        self.output_has_voice = False
        self.output_silence_seconds = 0.0
        self.output_last_packet = 0.0
        self.output_generation = 0
        self.draining = False
        self.drain_started = 0.0
        self.retry_pending = False
        self.control_tasks = set()
        self.control_lock = asyncio.Lock()
        self.playing = None
        self.playing_started = 0.0
        self.playing_seconds = 0.0
        self.delivery_tasks = set()
        self.state_signature = None
        self.last_activity = time.monotonic()
        self.receive_task = None
        self.tasks = []

    def log(self, type: str, **data):
        if self.logfile.closed:
            return
        self._processing.event(type, data, time.monotonic())
        self.logfile.write(json.dumps({"type": type, "t_ms": round((time.monotonic()-self.started)*1000), **data}, ensure_ascii=False) + "\n")
        self.logfile.flush()

    def processing_snapshot(self):
        """Cheap synchronous view for a separately scheduled UI publisher."""
        return self._processing.snapshot(self, time.monotonic())

    def spawn(self, coroutine, group, label):
        task = asyncio.create_task(coroutine, name="voice_" + label)
        group.add(task)

        def completed(done):
            group.discard(done)
            if done.cancelled():
                return
            error = done.exception()  # Always retrieve background exceptions.
            if error is not None and not self.closed:
                self.log("task_error", task=label, error_type=type(error).__name__)
                self.failure_message = "音声対話の処理が停止しました。新しい会話を開始してください。"
                self.finished.set()
        task.add_done_callback(completed)
        return task

    async def control(self, *, text=None, approval=None):
        async with self.control_lock:
            if approval is not None:
                return await self.client.approve(approval)
            return await self.client.instruct(text)

    async def status(self, text):
        await self.send_json({"type": "status", "text": text})

    async def start(self):
        await self.client.connect()
        if self.closed:
            await self.client.disconnect()
            return
        self.ready = True
        self.log("live_started", model=MODEL, voice=VOICE, session_id=self.client.session_id)
        self.tasks = set()
        self.receive_task = self.spawn(self.receive(), self.tasks, "receive")
        self.spawn(self.tick(), self.tasks, "tick")
        self.spawn(self.process_inputs(), self.tasks, "core")
        await self.status("接続しました。お話しください。")

    async def microphone(self, pcm: bytes):
        if self.closed or not self.ready:
            return
        if len(pcm) % 2 or len(pcm) > 48000:
            raise ValueError("音声データの形式が不正です。")
        self.input_samples += len(pcm) // 2
        level = rms(pcm)
        if level >= 280:
            self.input_last_voice = time.monotonic()
            self.last_activity = self.input_last_voice
            self.input_loud_seconds += len(pcm) / 48000
            if self.input_loud_seconds >= .16 and (self.playing or self.approved):
                await self.interrupt()
        else:
            self.input_loud_seconds = 0
        await self.client.audio(pcm)

    async def receive(self):
        try:
            async for event in self.client.events():
                typ = event.get("type", "")
                now = time.monotonic()
                if typ == "session.input_transcript.delta":
                    delta = event.get("delta", "")
                    if not delta:
                        continue
                    self.input_text += delta
                    if len(self.input_text) > 12000:
                        raise ValueError("input transcript limit")
                    self.input_end_ms = max(self.input_end_ms, event.get("end_ms", 0))
                    self.input_last_change = now
                    self.input_revision += 1
                    self.last_activity = now
                    self.log(typ, delta=delta, start_ms=event.get("start_ms"), end_ms=event.get("end_ms"))
                    await self.send_json({"type": "input_transcript", "text": delta})
                    if self.playing or self.approved:
                        await self.interrupt()
                elif typ == "session.delegation.created":
                    self.delegation = event
                    self.log(typ, offset_ms=event.get("offset_ms"), delegation=event.get("delegation"))
                elif typ == "session.output_transcript.delta":
                    self.output_last_text = now
                    self.log(typ, delta=event.get("delta"), start_ms=event.get("start_ms"), end_ms=event.get("end_ms"))
                    if not self.draining:
                        self.output_text += event.get("delta", "")
                        candidate = normalized(self.output_text)
                        if (self.input_text and not self.approved and not self.busy and candidate
                                and not any(phrase.startswith(candidate) for phrase in BACKCHANNELS)):
                            # Live sometimes starts a substantive answer rather
                            # than delegating. Its timing is a useful heuristic;
                            # the unapproved answer itself is NEVER forwarded.
                            self.native_response_boundary = now
                        if len(self.output_text) > 8000:
                            await self.reject_output("transcript_limit")
                elif typ == "session.output_audio.delta":
                    pcm = base64.b64decode(event["delta"], validate=True)
                    if len(pcm) % 2:
                        raise ValueError("invalid output PCM")
                    self.output_last_packet = now
                    # Measure quiet in audio time, rather than mistaking a
                    # delayed network packet for half a second of silence.
                    for offset in range(0, len(pcm), 960):
                        frame = pcm[offset:offset + 960]
                        loud = rms(frame) >= 100
                        if loud:
                            self.output_last_voice = now
                            self.output_silence_seconds = 0.0
                        else:
                            self.output_silence_seconds += len(frame) / 48000
                        if self.draining:
                            continue
                        if loud:
                            if not self.output_has_voice:
                                self.output_audio.extend(b"".join(self.output_preroll))
                            self.output_has_voice = True
                        if self.output_has_voice:
                            self.output_audio.extend(frame)
                        self.output_preroll.append(frame)
                    if len(self.output_audio) > 48000 * 45:
                        await self.reject_output("output_limit")
                elif typ in ("error", "session.error"):
                    # Do not echo raw server errors, input instructions, or headers.
                    self.log("live_error", code=(event.get("error") or {}).get("code"))
                    self.failure_message = "音声サービスでエラーが発生しました。停止して新しい会話を開始してください。"
                    break
                elif typ == "session.closed":
                    self.log("live_closed", usage=event.get("usage"))
                elif typ == "session.instructions.appended":
                    self.log(typ, client_event_id=event.get("client_event_id"))
        finally:
            if not self.closed:
                self.failure_message = self.failure_message or "音声接続が終了しました。新しい会話を開始してください。"
                self.finished.set()

    def clear_output(self):
        self.output_audio.clear()
        self.output_text = ""
        self.output_has_voice = False
        self.output_preroll.clear()
        self.output_silence_seconds = 0.0

    def begin_drain(self):
        """Discard old generation until a quiet boundary after revocation."""
        self.output_generation += 1
        self.draining = True
        self.drain_started = time.monotonic()
        self.clear_output()
        self.spawn(self.control(text="Stop speaking now. The previous APPROVED_TEXT is revoked. Listen and wait for new instructions."),
                   self.control_tasks, "revoke")

    async def deliver(self, message, status, heard_text=""):
        if message and message.get("kind") != "backchannel":
            await asyncio.to_thread(self.core.delivery, self.sid, message["id"], status, heard_text)
            self.log("delivery", id=message["id"], status=status, heard_text=heard_text)

    async def interrupt(self):
        """Clear physical playback first; never wait for the core lock to stop audio."""
        interrupted = self.playing
        cancelled = self.approved
        self.playing = self.approved = None
        self.retry_pending = False
        self.begin_drain()
        try:
            await self.send_json({"type": "interrupt", "revision": self.input_revision})
        finally:
            # Playback was detached above, so close() can no longer discover
            # these messages. Preserve their delivery even if the browser
            # disconnects or cancels this task while the stop is being sent.
            # Subclasses may supply an acknowledged prefix; never guess one.
            for message, status in ((interrupted, "interrupted"), (cancelled, "cancelled")):
                if message:
                    self.spawn(self.deliver(message, status), self.delivery_tasks, "delivery")
            self.log("interrupted", id=(interrupted or cancelled or {}).get("id"))

    async def played(self, playback_id):
        if not self.playing or self.playing["id"] != playback_id:
            return
        # Reject impossible early browser completion reports.
        if time.monotonic() - self.playing_started + .15 < self.playing_seconds:
            self.log("early_played_ignored", id=playback_id)
            return
        message, self.playing = self.playing, None
        # A core API request may hold the session lock. Do not make the browser
        # WebSocket reader (and microphone transport) wait for that lock.
        self.spawn(self.deliver(message, "completed"), self.delivery_tasks, "delivery")
        self.log("playback_completed", id=playback_id)
        self.last_activity = time.monotonic()

    async def enqueue(self, text, source="typed"):
        text = text.strip()
        if not text:
            return
        if len(text) > 2000:
            await self.send_json({"type": "error", "message": "一度のお話が長すぎました。2000文字以内に分けてお話しください。"})
            return
        if self.jobs.full():
            await self.send_json({"type": "error", "message": "入力待ちが多くなっています。少しお待ちください。"})
            return
        automatic = source != "typed"
        revision, last_voice = self.input_revision, self.input_last_voice
        if automatic and self.input_transport_pending:
            return False
        if self.playing or self.approved:
            await self.interrupt()
        # An interrupt may yield to more captured audio or transcript text.
        # Leave the caller's transcript intact so the next tick can reconsider.
        if automatic and (self.input_transport_pending or revision != self.input_revision
                          or last_voice != self.input_last_voice):
            return False
        if self.jobs.full():
            return False
        self.cancel_generation()
        self.input_revision += 1
        self.last_activity = time.monotonic()
        request_id = "voice_" + uuid.uuid4().hex
        self._input_job_sources[request_id] = source
        self.jobs.put_nowait((text, request_id, self.input_revision))
        self.log("input_committed", text=text, source=source, revision=self.input_revision)
        return True

    def cancel_generation(self):
        """Cancel only this session's optional cancellable backend request."""
        if self._active_cancel is not None:
            self._active_cancel.set()

    @property
    def input_transport_pending(self):
        """Locally received audio must not be mistaken for conversational quiet."""
        return self.microphone_backlog_bytes > INPUT_TRANSPORT_BARRIER_BYTES

    async def _wait_input_transport(self, source):
        if source == "typed":
            return True
        while self.input_transport_pending:
            if self.closed or self.finished.is_set():
                return False
            await asyncio.sleep(.02)
        return not self.closed and not self.finished.is_set()

    async def _submit_input(self, text, request_id, source, revision):
        cancelled = threading.Event()
        self._active_cancel = cancelled
        # Input may have arrived while pending delivery/status awaited, before
        # an active cancellation event existed. Persist the earlier input but
        # do not start a paid generation that is already obsolete.
        if revision != self.input_revision or not self.jobs.empty() or self.input_text:
            cancelled.set()
        def submit_if_ready():
            # Executor work may start after the event loop accepted more PCM.
            if source != "typed" and (self.input_transport_pending or self.closed or self.finished.is_set()):
                return _SUBMIT_DEFERRED
            submit = getattr(self.core, "submit_cancellable", None)
            if callable(submit):
                return submit(self.sid, text, request_id, cancelled)
            return self.core.submit(self.sid, text, request_id)

        try:
            while await self._wait_input_transport(source):
                state = await asyncio.to_thread(submit_if_ready)
                if state is not _SUBMIT_DEFERRED:
                    return state
            return _SUBMIT_DEFERRED
        finally:
            cancelled.set()
            if self._active_cancel is cancelled:
                self._active_cancel = None

    async def process_inputs(self):
        while True:
            text, request_id, revision = await self.jobs.get()
            source = self._input_job_sources.get(request_id, "typed")
            self.busy = True
            refresh_reply = False
            try:
                if not await self._wait_input_transport(source):
                    return
                if self.delivery_tasks:
                    await asyncio.gather(*list(self.delivery_tasks))
                pending = await asyncio.to_thread(self.core.pending, self.sid)
                for message in pending:
                    await self.deliver(message, "cancelled")
                await self.status("お話を受け取りました。返事を考えています。")
                self.log("core_submit_started", request_id=request_id)
                state = await self._submit_input(text, request_id, source, revision)
                if state is _SUBMIT_DEFERRED:
                    return
                if not isinstance(state, dict):
                    raise ValueError("invalid core session view")
                if state.get("request_status") == "interrupted":
                    self.log("core_response_cancelled", request_id=request_id)
                    continue
                # A completion callback can publish an old introduction after
                # the pending snapshot above but before submit acquires the
                # core lock. The returned message order tells us exactly which
                # pending outputs precede this newly accepted user turn.
                messages = state.get("messages", [])
                user_index = next((i for i in range(len(messages)-1, -1, -1)
                                   if messages[i].get("role") == "user"), -1)
                for message in messages[:max(0, user_index)]:
                    if (message.get("role") == "assistant" and message.get("kind") != "error"
                            and message.get("delivery", {}).get("status") == "pending"):
                        await self.deliver(message, "cancelled")
                if state.get("error"):
                    # A failed generation is local to this turn. Keep the
                    # microphone and worker alive so the next input can recover
                    # with the already saved conversation history.
                    superseded = (revision != self.input_revision or not self.jobs.empty()
                                  or bool(self.input_text))
                    self.log("core_response_failed", request_id=request_id, superseded=superseded)
                    for message in await asyncio.to_thread(self.core.pending, self.sid):
                        await self.deliver(message, "cancelled")
                    if not superseded:
                        await self.status("返答を生成できませんでした。このまま続けてお話しください。")
                    continue
                if revision != self.input_revision or not self.jobs.empty() or self.input_text:
                    for message in await asyncio.to_thread(self.core.pending, self.sid):
                        await self.deliver(message, "cancelled")
                    self.log("stale_reply_cancelled", revision=revision)
                else:
                    self._observe_core_state(state)
                    refresh_reply = True
            except Exception as exc:
                self.log("core_error", error_type=type(exc).__name__)
                self.failure_message = "応答処理に失敗しました。新しい会話を開始してください。"
                self.finished.set()
                return
            finally:
                self.log("core_submit_finished", request_id=request_id)
                self._input_job_sources.pop(request_id, None)
                self.busy = False
                self.jobs.task_done()
                await self._input_processing_finished(revision, refresh_reply)
            if refresh_reply:
                # A completed reply need not wait for the next 500 ms state
                # poll. Use the existing snapshot and offer guards: newly
                # queued input, interruption, shutdown or another poll can
                # still invalidate it before synthesis is reserved.
                await self._refresh_state()

    async def _input_processing_finished(self, revision, reply_ready):
        """Optional turn feedback can stop when no reply will follow."""

    def _offer_current(self, revision, generation):
        """Recheck the floor after each asynchronous state/pending lookup."""
        return (not self.closed and not self.finished.is_set() and not self.busy and not self.draining
                and not self.approved and not self.playing and not self.input_text
                and self.jobs.empty() and not self.delivery_tasks
                and revision == self.input_revision and generation == self.output_generation
                and time.monotonic()-self.input_last_voice > .7)

    async def _pending_if_idle(self):
        """A periodic voice poll must not wait behind a model request's lock."""
        snapshot = getattr(self.core, "view_if_idle", None)
        if callable(snapshot):
            state = await asyncio.to_thread(snapshot, self.sid)
            if state is None:
                return None
            return [{"id": message["id"], "text": message["content"], "kind": message["kind"]}
                    for message in state.get("messages", []) if message.get("role") == "assistant"
                    and message.get("delivery", {}).get("status") == "pending"]
        return await asyncio.to_thread(self.core.pending, self.sid)

    async def _offer_pending(self, revision, generation):
        if not self._offer_current(revision, generation):
            return
        pending = await self._pending_if_idle()
        if not self._offer_current(revision, generation) or not pending:
            return
        if self.output_has_voice or self.output_text:
            self.begin_drain()
            return
        message = pending[0]
        self.approved = message
        self.approved_at = time.monotonic()
        self.approval_revision = revision
        self.retry_count = 0
        self.clear_output()
        await self.control(approval=message["text"])
        if self.approved is not message or self.output_generation != generation:
            return
        if (self.closed or self.finished.is_set() or self.busy or self.draining or self.input_text or not self.jobs.empty()
                or self.delivery_tasks or self.input_revision != revision
                or time.monotonic()-self.input_last_voice <= .7):
            await self.interrupt()
            return
        self.log("approved_text_sent", id=message["id"], text=message["text"])
        await self.status("返事を音声にしています。")

    def _observe_core_state(self, state):
        """Optional voice-controller state update; never changes core content."""

    async def _refresh_state(self):
        if self.closed or self.finished.is_set() or self.busy:
            return
        revision, generation = self.input_revision, self.output_generation
        view = getattr(self.core, "view_if_idle", None) or self.core.view
        state = await asyncio.to_thread(view, self.sid)
        if (state is None or self.closed or self.finished.is_set() or self.busy or self.input_revision != revision
                or self.output_generation != generation):
            return
        self._observe_core_state(state)
        signature = json.dumps(state, ensure_ascii=False, sort_keys=True)
        if signature != self.state_signature:
            await self.send_json({"type": "session", "session": state})
            self.state_signature = signature
        await self._offer_pending(revision, generation)

    async def reject_output(self, reason):
        if self.draining:
            return
        self.log("output_blocked", reason=reason, actual=self.output_text,
                 expected=self.approved["text"] if self.approved else None)
        self.begin_drain()
        if self.approved:
            if self.retry_count < 1:
                self.retry_count += 1
                self.approved_at = time.monotonic()
                self.retry_pending = True
                await self.status("発話内容を確認しています。")
            else:
                message, self.approved = self.approved, None
                self.retry_pending = False
                self.spawn(self.deliver(message, "cancelled"), self.delivery_tasks, "delivery")
                self.failure_message = "生成した文章と音声が一致しなかったため、読み上げを止めました。承認文は会話欄で確認できます。"
                self.finished.set()

    async def check_output(self, now):
        if self.draining or self.playing or not self.output_has_voice or now - self.output_last_text < .35:
            return
        # Real Live streams include continuous silent PCM. Tests or transports
        # that stop sending after the utterance use the conservative quiet-gap
        # fallback, while a recently received non-silent packet never qualifies.
        quiet = self.output_silence_seconds >= .5 or (
            now - self.output_last_packet > .8 and now - self.output_last_voice > .8)
        if not quiet:
            return
        text = normalized(self.output_text)
        if self.approved and text == normalized(self.approved["text"]):
            if self.approval_revision != self.input_revision or self.input_text or now-self.input_last_voice < .7:
                await self.interrupt()
                return
            message, self.approved = self.approved, None
        elif text in BACKCHANNELS:
            # A separate permitted backchannel does not revoke the approved
            # main reply. Mixed backchannel + main text still fails exact match.
            message = {"id": "backchannel_"+uuid.uuid4().hex, "text": self.output_text, "kind": "backchannel"}
        elif self.approved and normalized(self.approved["text"]).startswith(text) and text:
            # A pause inside the approved sentence is not an output end.
            return
        else:
            await self.reject_output("transcript_mismatch")
            return
        pcm = bytes(self.output_audio)
        self.log("output_approved", id=message["id"], kind=message["kind"], text=self.output_text,
                 buffered_ms=round(len(pcm)/48))
        self.clear_output()
        self.playing = message
        self.playing_started = now
        self.playing_seconds = len(pcm) / 48000
        generation = self.output_generation
        await self.send_json({"type": "playback", "id": message["id"]})
        if generation != self.output_generation or self.playing is not message:
            return
        await self.send_json({"type": "output_transcript", "text": message["text"]})
        for offset in range(0, len(pcm), 48000):
            if generation != self.output_generation or self.playing is not message:
                return
            await self.send_bytes(pcm[offset:offset+48000])
        if generation != self.output_generation or self.playing is not message:
            return
        await self.send_json({"type": "playback_end", "id": message["id"]})
        await self.status("お話ししています。途中で話しかけても大丈夫です。")

    async def _commit_live_input(self, now):
        if (self.input_transport_pending or not self.input_text or self.jobs.full()
                or now-self.input_last_voice <= .55 or now-self.input_last_change <= .45):
            return False
        native = self.delegation and self.delegation.get("offset_ms", 0) >= self.input_end_ms - 350
        response_boundary = self.native_response_boundary >= self.input_last_voice and self.native_response_boundary > 0
        fallback = now-max(self.input_last_voice, self.input_last_change) > 3.5
        if not (native or response_boundary or fallback):
            return False
        source = "live_delegation" if native else "live_response_boundary" if response_boundary else "silence_recovery"
        text = self.input_text
        accepted = await self.enqueue(text, source)
        if accepted is False:
            return False
        self.input_text = self.input_text[len(text):]
        self.delegation = None
        self.native_response_boundary = 0.0
        return accepted is True

    async def tick(self):
        counter = 0
        while True:
            await asyncio.sleep(.1)
            now = time.monotonic()
            if self.draining:
                quiet_audio = self.output_silence_seconds >= .5 or now-self.output_last_packet > .8
                if (not self.control_tasks and now-self.drain_started > .8 and
                        quiet_audio and now-self.output_last_voice > .5 and now-self.output_last_text > .35):
                    self.draining = False
                    self.clear_output()
                    if self.retry_pending and self.approved:
                        self.retry_pending = False
                        self.approved_at = now
                        await self.control(approval=self.approved["text"])
                elif now-self.drain_started > 12:
                    self.failure_message = "音声の停止を確認できなかったため、接続を終了します。"
                    self.finished.set()
                    return
            # Native Live delegation is the main signal. Silence alone only
            # recovers a lost/missing delegation after a conspicuously long gap.
            await self._commit_live_input(now)
            await self.check_output(now)
            if self.approved and not self.draining and now-self.approved_at > self.approved_timeout_seconds:
                await self.reject_output("speech_timeout")
            if self.playing and now-self.playing_started > self.playing_seconds+5:
                await self.interrupt()
                self.failure_message = "音声の再生完了を確認できなかったため停止しました。"
                self.finished.set()
                return
            counter += 1
            if counter % 5 == 0 and not self.busy:
                await self._refresh_state()
            if now-self.last_activity > 120 and not self.busy and not self.playing and not self.approved:
                self.failure_message = "しばらく会話がなかったため音声接続を終了します。"
                self.finished.set()
                return

    async def close(self):
        if self.closed:
            await self.close_done.wait()
            return
        self.closed = True
        self.cancel_generation()
        self.ready = False
        self.finished.set()
        pending = ((self.playing, "interrupted"), (self.approved, "cancelled"))
        self.playing = self.approved = None
        self.clear_output()
        try:
            # Close the metered voice session before waiting for any core lock.
            try:
                await asyncio.wait_for(self.send_json({"type":"interrupt"}), 1)
            except Exception:
                pass
            tasks = [task for task in list(self.tasks) + list(self.control_tasks)
                     if task is not self.receive_task and task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(self.client.request_close(), 2)
                if self.receive_task:
                    await asyncio.wait_for(asyncio.shield(self.receive_task), 5)
            except Exception:
                pass
            finally:
                if self.receive_task and not self.receive_task.done():
                    self.receive_task.cancel()
                    await asyncio.gather(self.receive_task, return_exceptions=True)
                await self.client.disconnect()
            self.log("finalization", complete=self.client.finalized.is_set(), usage=self.client.usage)
            for message, status in pending:
                if message:
                    self.spawn(self.deliver(message, status), self.delivery_tasks, "delivery")
            delivery = list(self.delivery_tasks)
            if delivery:
                try:
                    await asyncio.wait_for(asyncio.gather(*delivery, return_exceptions=True), 5)
                except TimeoutError:
                    self.log("delivery_finalization_incomplete")
        finally:
            self.logfile.close()
            self.close_done.set()
