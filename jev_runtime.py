"""Jev controls turn timing and interruption; the dialogue backend controls content.

ASR segments are not dialogue turns. During playback, tentative user text is
held separately so an acknowledgement cannot revoke an approved TTS reply.
"""
from __future__ import annotations

import asyncio
import json
import time

from speech_runtime import SpeechRuntime
from voice_runtime import normalized, rms
from interruption_intent import interruption_kind
from filler_runtime import FillerPlayback

VAD_SECONDS = .8
STABLE_SECONDS = .2
COMPLETE_GRACE_SECONDS = .25
SHORT_WAIT_SECONDS = 2.0
LONG_WAIT_SECONDS = 3.5
FAILURE_WAIT_SECONDS = 2.0
MAX_HOLD_SECONDS = 8.0
BACKCHANNEL_CONFIDENCE = .85
BACKCHANNEL_DOMINANT_PROBABILITY = .60
BACKCHANNEL_MAX_INTERRUPT_PROBABILITY = .10
PREPARATION_RETAIN_PROBABILITY = .80
PREPARATION_REVISE_PROBABILITY = .80
PREPARATION_MAX_HOLD_SECONDS = 3.0
NOISE_ONLY_WAIT_SECONDS = 2.0
NOISE_EVIDENCE_MAX_AGE_SECONDS = .75
SHORT_ACKS = {"うん", "はい", "ええ", "なるほど", "そっか", "そうなんだ", "そうですね", "わかった"}
CONTINUATION_MARKERS = ("でも", "違", "待", "止め", "やめ", "いや", "いいえ", "だけど", "質問")


class JevSpeechRuntime(SpeechRuntime):
    CAPTURE_VOICE_RMS = 560

    def __init__(self, core, session_id, send_json, send_bytes, data_dir,
                 client=None, speech=None, jev=None, filler_clips=None):
        if client is None:
            from transcription_client import TranscriptionClient
            client = TranscriptionClient()
        if jev is None:
            from jev_client import JevClient
            jev = JevClient()
        super().__init__(core, session_id, send_json, send_bytes, data_dir, client=client, speech=speech)
        self.jev = jev
        self.filler = FillerPlayback(self, filler_clips)
        self.segments = {}
        self.buffer_version = 0
        self.voice_epoch = 0
        self.judge_tasks = set()
        self.judge_task = None
        self.judge_sequence = 0
        self.next_judge_at = 0.0
        self.verdict = None
        self.promoting = False
        self.preparation_promoting = False
        self.preparation_context = None
        self.preparation_sequence = 0
        self.last_committed_text = ""
        self.last_heard_text = ""
        self.noise_wait_started = 0.0
        self.last_empty_asr_at = 0.0
        self.noise_turn_key = None
        self.noise_turn_started = 0.0
        self.jev_closing = False
        self.jev_close_done = asyncio.Event()
        self._processing.configure(mode="jev", quiet=VAD_SECONDS, stable=STABLE_SECONDS,
                                   complete_grace=COMPLETE_GRACE_SECONDS,
                                   short_wait=SHORT_WAIT_SECONDS, long_wait=LONG_WAIT_SECONDS,
                                   failure=FAILURE_WAIT_SECONDS, maximum=MAX_HOLD_SECONDS,
                                   preparation_maximum=PREPARATION_MAX_HOLD_SECONDS,
                                   preparation_failure=FAILURE_WAIT_SECONDS,
                                   preparation_retain_probability=PREPARATION_RETAIN_PROBABILITY,
                                   noise_wait=NOISE_ONLY_WAIT_SECONDS,
                                   noise_evidence_age=NOISE_EVIDENCE_MAX_AGE_SECONDS)

    async def start(self):
        await self.client.connect()
        if self.closed or self.jev_closing:
            await self.client.disconnect()
            return
        self.ready = True
        self.tasks = set()
        self.receive_task = self.spawn(self.receive(), self.tasks, "receive")
        self.spawn(self.tick(), self.tasks, "tick")
        self.spawn(self.process_inputs(), self.tasks, "core")
        self.log("transcription_started", model="gpt-live-transcribe", session_id=self.client.session_id)
        self.log("jev_started", model=getattr(self.jev, "model", "jev-1.13.0"),
                 vad_seconds=VAD_SECONDS, stable_seconds=STABLE_SECONDS,
                 complete_grace_seconds=COMPLETE_GRACE_SECONDS, short_wait_seconds=SHORT_WAIT_SECONDS,
                 long_wait_seconds=LONG_WAIT_SECONDS, maximum_hold_seconds=MAX_HOLD_SECONDS,
                 failure_wait_seconds=FAILURE_WAIT_SECONDS)
        self.log("speech_started", source="tts", speech_model=self.speech_model, voice=self.speech_voice)
        await self.status("接続しました。お話しください。")

    async def microphone(self, pcm):
        if self.closed or self.jev_closing or not self.ready:
            return
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > 48000:
            raise ValueError("音声データの形式が不正です。")
        self.input_samples += len(pcm) // 2
        if rms(pcm) >= 560:
            self.input_last_voice = self.last_activity = time.monotonic()
            self.voice_epoch += 1
        # Sound level invalidates a pending end-of-turn guess, never playback.
        await self.client.audio(pcm)

    def _rows(self, lane):
        return sorted((row for row in self.segments.values() if row["lane"] == lane),
                      key=lambda row: (row["sequence"], row["arrival"]))

    def _text(self, lane):
        return "".join(row["text"] for row in self._rows(lane))

    def _all_final(self, lane):
        rows = self._rows(lane)
        return (not self.input_transport_pending and not getattr(self.client, "pending_audio", False)
                and bool(rows) and all(row["final"] for row in rows))

    def _words_final(self, lane):
        rows = [row for row in self._rows(lane) if row["text"]]
        return bool(rows) and all(row["final"] for row in rows)

    def _turn_noise_recovery(self, lane, now):
        # Keep a bounded acoustic wait for the SAME finalized words; new words
        # or a different reply reset it. Empty ASR segments never reset it.
        context = self._preparation_snapshot_context() if lane == "preparation" else self._context()
        key = (lane, self.buffer_version, self.input_revision, self.output_generation,
               context.get("id"), (self.playing or {}).get("id"))
        if not self._words_final(lane):
            self.noise_turn_key = None
            return False
        if key != self.noise_turn_key:
            self.noise_turn_key, self.noise_turn_started = key, now
        return (not self.input_transport_pending
                and now-self.noise_turn_started >= NOISE_ONLY_WAIT_SECONDS
                and self.last_empty_asr_at >= self.noise_turn_started
                and 0 <= now-self.last_empty_asr_at <= NOISE_EVIDENCE_MAX_AGE_SECONDS
                and not any(row["text"] and not row["final"] for row in self.segments.values()
                            if row["lane"] in ("normal", "candidate", "preparation")))

    def _bypass_empty_rows(self):
        for row in self.segments.values():
            if row["lane"] in ("normal", "candidate", "preparation") and not row["text"]:
                row["noise_bypassed"] = True

    def _context(self):
        return next((row["context"] for row in self._rows("candidate") if row["text"] and row["context"]), {})

    def _new_preparation(self, original_text, *, background=False):
        self.preparation_sequence += 1
        self.preparation_context = {"id": self.preparation_sequence,
                                    "original_text": original_text,
                                    "last_heard_text": self.last_heard_text,
                                    "reply_text": "", "background": background,
                                    "protect_reply": bool(original_text) and not background}

    def _assign_input_lane(self, row):
        # Retained ASR rows remain editable. A correction arriving after actual
        # playback belongs to that playback (or normal listening), not its old
        # preparation window.
        if row["lane"] == "ignored_preparation":
            row["context"] = {}
        if self.preparation_promoting:
            row["lane"], row["context"] = "normal", {}
        elif self.playing or self._text("candidate") or self.promoting or row["context"]:
            row["lane"] = "candidate"
            if not row["context"]:
                active = self.playing or self._context()
                row["context"] = {"id": active.get("id"), "text": active.get("text", "")}
        elif (self.preparation_context is not None
              and self.preparation_context.get("protect_reply", True)
              and not self._text("normal")):
            row["lane"] = "preparation"
            row["context"] = dict(self.preparation_context)
        else:
            row["lane"], row["context"] = "normal", {}

    def _preparation_snapshot_context(self):
        context = next((row["context"] for row in self._rows("preparation") if row["text"] and row["context"]), {})
        if self.preparation_context and context.get("id") == self.preparation_context["id"]:
            context = self.preparation_context
        # A draft the user has never heard cannot change the meaning of their
        # addition. TTS becoming ready must not invalidate an in-flight verdict.
        return {key: value for key, value in context.items() if key != "reply_text"}

    def _snapshot(self, lane):
        context = self._context() if lane == "candidate" else {}
        if lane == "preparation":
            context = self._preparation_snapshot_context()
        snapshot = {"lane": lane, "text": self._text(lane), "version": self.buffer_version,
                    "voice_epoch": self.voice_epoch, "last_voice": self.input_last_voice,
                    "revision": self.input_revision, "generation": self.output_generation,
                    "playing_id": (self.playing or {}).get("id"), "context": dict(context)}
        if lane == "preparation":
            snapshot["preparation_id"] = (self.preparation_context or {}).get("id")
        return snapshot

    def _current(self, snapshot):
        current = self._snapshot(snapshot["lane"])
        # Meaning does not change with sound alone. Keep the classification;
        # _advance and playback still check current acoustic/ASR barriers before
        # applying it. Actual text/context changes invalidate every lane.
        ignored = {"voice_epoch", "last_voice"}
        return (not self.closed and not self.jev_closing and not self.finished.is_set()
                and all(value == current[key] for key, value in snapshot.items() if key not in ignored))

    def _changed(self, *, text_changed=True):
        self.buffer_version += 1
        self.verdict = None
        self.last_activity = time.monotonic()
        if text_changed:
            self.input_last_change = self.last_activity

    def _sync_normal_input(self):
        """Project editable ASR segments into the buffer that gates speech."""
        text = self._text("normal")
        if text != self.input_text:
            self.input_text = text
            self.input_revision += 1
        self._refresh_interruption_wait()

    async def _publish_input_caption(self):
        caption = "".join(row["text"] for row in sorted(self.segments.values(),
                          key=lambda row: (row["sequence"], row["arrival"])))
        await self.send_json({"type": "input_transcript", "text": caption[-12000:], "replace": True})

    async def _accept_transcript(self, event):
        completed = event["type"] == "session.input_transcript.completed"
        pending = event["type"] == "session.input_transcript.pending"
        item_id = event.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("ASR segment id missing")
        sequence = event.get("sequence", len(self.segments))
        if type(sequence) is not int or sequence < 0:
            raise ValueError("ASR segment sequence invalid")
        new_item = item_id not in self.segments
        if new_item:
            if len(self.segments) >= 1000:
                raise ValueError("ASR segment limit")
            self.segments[item_id] = {"id": item_id, "sequence": sequence, "arrival": len(self.segments),
                                      "text": "", "final": False, "lane": "new", "context": {}}
        row = self.segments[item_id]
        if pending:
            if new_item:
                self._assign_input_lane(row)
                self.log("asr_segment_pending", item_id=item_id, sequence=sequence)
            return
        old_text, old_sequence = row["text"], row["sequence"]
        text = event.get("segment_text", event.get("text", "") if completed
                         else old_text + event.get("delta", ""))
        if not isinstance(text, str) or len(text) > 12000:
            raise ValueError("ASR transcript limit")
        if completed:
            self.log("asr_segment_final", item_id=item_id, sequence=sequence,
                     text_chars=len(text), empty=not text)
        changed = text != old_text or old_sequence != sequence
        if row["lane"] == "consumed":
            if changed:
                self.log("late_consumed_transcript_ignored", item_id=item_id)
            return
        row.update(text=text, sequence=sequence, final=completed or (row["final"] and not changed))
        if completed and not text:
            previous_lane = row["lane"]
            row["lane"] = "consumed"
            # An empty noise segment releases the final-ASR barrier but adds no
            # words. Preserve the existing turn's wait clock unless this final
            # actually removes previously recognised interim text.
            if old_text:
                self._changed()
            else:
                self.last_empty_asr_at = time.monotonic()
            # An interim normal segment can disappear in final ASR. Merely
            # consuming its row leaves stale input_text blocking every pending
            # reply forever, while _advance sees no text to commit or clear.
            self._sync_normal_input()
            self.log(event["type"], item_id=item_id, sequence=sequence, segment_text="",
                     lane=row["lane"], previous_lane=previous_lane, final=True)
            await self._publish_input_caption()
            return
        if row["lane"] in ("ignored", "ignored_preparation") and not changed:
            return  # The final event for an already classified acknowledgement.
        if not changed:
            # Keep complete/in-flight answers current. _advance separately
            # refreshes short/long answers made before all ASR became final.
            return
        stale_empty_candidate = (not old_text and row["lane"] == "candidate"
                                 and row["context"].get("id") != (self.playing or {}).get("id"))
        if row.pop("noise_bypassed", False) or stale_empty_candidate:
            # An empty unresolved segment was kept when playback resumed.
            # Delayed real words must belong to the CURRENT floor and cannot
            # remain trapped in the old preparation lane.
            row["lane"], row["context"] = "new", {}
        if row["lane"] in ("new", "ignored", "ignored_preparation"):
            self._assign_input_lane(row)
        self._changed()
        if row["lane"] == "normal":
            self._sync_normal_input()
            if self.approved:
                await self.interrupt()
        if sum(len(self._text(lane)) for lane in ("normal", "candidate", "preparation")) > 12000:
            raise ValueError("Pending transcript limit")
        self.log(event["type"], item_id=item_id, sequence=sequence, segment_text=text,
                 lane=row["lane"], final=row["final"])
        await self._publish_input_caption()
        # Complete short stop/denial phrases take the floor immediately. Keep
        # their editable ASR rows while waiting for the user's continuation;
        # longer corrections still follow ordinary semantic turn judging.
        lane = row["lane"]
        kind = interruption_kind(self._text(lane)) if lane in ("candidate", "preparation") else None
        if kind:
            reason = "local_explicit_stop" if kind == "pause" else "local_explicit_correction"
            self.log(reason, text=self._text(lane))
            if lane == "candidate":
                await self._promote_candidates(reason)
            else:
                await self._promote_preparation(reason)

    async def receive(self):
        try:
            async for event in self.client.events():
                typ = event.get("type", "")
                if typ in ("session.input_transcript.delta", "session.input_transcript.completed", "session.input_transcript.pending"):
                    await self._accept_transcript(event)
                elif typ in ("error", "session.error"):
                    self.log("transcription_error", code=(event.get("error") or {}).get("code"))
                    self.failure_message = "音声認識サービスでエラーが発生しました。新しい会話を開始してください。"
                    break
                elif typ == "session.closed":
                    self.log("transcription_closed", usage=event.get("usage"))
        finally:
            if not self.closed and not self.jev_closing:
                self.failure_message = self.failure_message or "音声認識の接続が終了しました。新しい会話を開始してください。"
                self.finished.set()

    def _offer_current(self, revision, generation):
        # Starting TTS is useful while extra speech is being classified. Only
        # actual playback waits; core/TTS identity remains unchanged until a
        # meaningful addition is promoted to a normal user turn.
        return (not self.jev_closing and not self.promoting and not self._text("candidate")
                and super()._offer_current(revision, generation))

    def _start_speech(self, message, revision, generation):
        # Includes an introduction or a unprompted reply prepared without a
        # currently running foreground submit.
        if self.preparation_context is None:
            # With no outstanding foreground input this is an introduction or
            # a later unprompted reply, not an unanswered user turn to dedupe.
            self._new_preparation(self.last_committed_text, background=True)
        self.preparation_context["background"] = False
        self.preparation_context["reply_text"] = message["text"]
        super()._start_speech(message, revision, generation)

    def _noise_wait_can_release(self, now):
        # Require evidence acquired during THIS ready-audio wait, not a stale
        # empty segment from before a person started speaking. No transcript or
        # transport barrier is bypassed, and no unrecognised row is discarded.
        since = max(self.noise_wait_started, self.input_last_change)
        return (self.noise_wait_started > 0 and now-since >= NOISE_ONLY_WAIT_SECONDS
                and self.last_empty_asr_at >= since
                and 0 <= now-self.last_empty_asr_at <= NOISE_EVIDENCE_MAX_AGE_SECONDS
                and not self.promoting and not self.input_transport_pending
                and not any(self._text(lane) for lane in ("normal", "candidate", "preparation")))

    async def _wait_preparation_input(self, message, revision, generation):
        self.noise_wait_started = time.monotonic()
        try:
            while self._speech_current(message, revision, generation):
                now = time.monotonic()
                hard = (self.promoting or self.input_transport_pending
                        or any(self._text(lane) for lane in ("normal", "candidate", "preparation")))
                sound_only = (getattr(self.client, "pending_audio", False)
                              or any(self._rows(lane) for lane in ("normal", "candidate", "preparation"))
                              or now-self.input_last_voice < .3)
                release_noise = not hard and sound_only and self._noise_wait_can_release(now)
                if not hard and (not sound_only or release_noise):
                    if release_noise:
                        self._bypass_empty_rows()
                        self.log("noise_only_wait_released",
                                 waited_ms=round((now-self.noise_wait_started)*1000),
                                 empty_asr_age_ms=round((now-self.last_empty_asr_at)*1000))
                    # The caller switches to playing before its next await.
                    self.preparation_context = None
                    return True
                await asyncio.sleep(.02)
            return False
        finally:
            self.noise_wait_started = 0.0

    def _heard_progress(self, message, prefix, completed):
        # SpeechRuntime confirms exact segment prefixes, including when the
        # remaining reply is later interrupted. Never substitute unheard text.
        self.last_heard_text = prefix

    async def enqueue(self, text, source="typed"):
        if (source == "typed" and isinstance(text, str) and text.strip()
                and self.waiting_for_continuation and self.input_text):
            text = self.input_text + "\n" + text.strip()
            # Validate the combined turn before consuming any editable ASR.
            # A rejected continuation must leave the held interruption intact.
            if len(text.strip()) > 2000 or self.jobs.full():
                return False
        if source == "typed" and isinstance(text, str) and text.strip() and len(text.strip()) <= 2000 and not self.jobs.full():
            for row in self.segments.values():
                if row["lane"] in ("normal", "candidate", "preparation"):
                    row["lane"] = "consumed"
            self.input_text = ""
            self.buffer_version += 1
            self.verdict = None
            # The interrupt below can await the browser. Speech arriving there
            # must not be judged against the previous, now superseded input.
            self.preparation_context = None
        if (isinstance(text, str) and text.strip() and len(text.strip()) <= 2000
                and not self.jobs.full() and self.filler.active):
            revision, last_voice = self.input_revision, self.input_last_voice
            await self.interrupt()
            if (source != "typed" and (self.input_transport_pending
                    or revision != self.input_revision or last_voice != self.input_last_voice)):
                return False
        accepted = await super().enqueue(text, source)
        if accepted:
            self.last_committed_text = text.strip()
            self._new_preparation(self.last_committed_text)
            if source == "typed":
                self.filler.schedule()
        return accepted

    @property
    def filler_playback(self):
        return self.filler.active

    async def _wait_auxiliary_playback(self, message, revision, generation):
        return await self.filler.before_reply(message, revision, generation)

    async def played(self, playback_id):
        if await self.filler.played(playback_id):
            return
        await super().played(playback_id)

    async def filler_skipped(self, playback_id):
        await self.filler.skipped(playback_id)

    async def _input_processing_finished(self, revision, reply_ready):
        state = self.filler.active
        if not reply_ready and state and state["revision"] == revision:
            await self.filler.stop("no_reply")

    async def interrupt(self):
        self.filler.invalidate()
        try:
            await super().interrupt()
        finally:
            await self.filler.stop("interrupted", notify=False)

    async def _promote_preparation(self, reason):
        if not self._text("preparation"):
            return
        self.cancel_generation()
        self.promoting = self.preparation_promoting = True
        try:
            # Move before the interrupt await so new deltas join normal input.
            for row in self._rows("preparation"):
                row["lane"], row["context"] = "normal", {}
            self.preparation_context = None
            self.input_interrupted_output = True
            self._sync_normal_input()
            self._changed(text_changed=False)
            if self.approved or self.playing or self.filler.active:
                await self.interrupt()
            self.log("preparation_promoted", reason=reason, text=self.input_text)
        finally:
            self.promoting = self.preparation_promoting = False

    def _retain_preparation(self):
        text = self._text("preparation")
        for row in self._rows("preparation"):
            if not row["text"]:
                row["noise_bypassed"] = True
                continue
            row["lane"] = "ignored_preparation"
        self.buffer_version += 1
        self.verdict = None
        self.log("preparation_retained", reason="jev_redundant", text=text)

    async def _promote_candidates(self, reason):
        if not self._text("candidate"):
            return
        self.cancel_generation()
        self.promoting = True
        context = self._context()
        try:
            if self.playing and self.playing["id"] == context.get("id"):
                await self.interrupt()
            # New deltas may have arrived while the interrupt was sent. Move
            # the current complete buffer, not an older classification string.
            for row in self._rows("candidate"):
                row["lane"] = "normal"
            self.input_text = self._text("normal")
            self.input_revision += 1
            self.input_interrupted_output = True
            self._refresh_interruption_wait()
            self.buffer_version += 1
            self.verdict = None
            self.log("interruption_promoted", reason=reason, text=self.input_text)
        finally:
            self.promoting = False

    def _ignore_candidates(self, reason):
        text = self._text("candidate")
        for row in self._rows("candidate"):
            if not row["text"]:
                row["noise_bypassed"] = True
                continue
            # Keep text, ID and original speaker context so a later correction
            # or 'うん、でも違います' extension reopens classification.
            row["lane"] = "ignored"
        self.buffer_version += 1
        self.verdict = None
        self.log("backchannel_ignored", reason=reason, text=text)

    @staticmethod
    def _safe_backchannel(text, result):
        answer = result.get("interruption") or {}
        compact = normalized(text)
        return (answer.get("choice") == "backchannel" and answer.get("confidence", 0) >= BACKCHANNEL_CONFIDENCE
                and answer.get("probabilities", {}).get("backchannel", 0) >= BACKCHANNEL_CONFIDENCE
                and len(compact) <= 30 and not any(marker in compact for marker in CONTINUATION_MARKERS))

    @staticmethod
    def _backchannel_dominant(text, result):
        """A weak acknowledgement is not evidence for a timed interruption.

        Keep the conservative fast-ack threshold above. This weaker evidence
        only protects the bounded recovery path, after final ASR and silence.
        A possible substantive continuation still takes that recovery path.
        """
        answer = result.get("interruption") or {}
        probabilities = answer.get("probabilities", {})
        compact = normalized(text)
        return (answer.get("choice") == "backchannel"
                and probabilities.get("backchannel", 0) >= BACKCHANNEL_DOMINANT_PROBABILITY
                and probabilities.get("interrupt", 1) <= BACKCHANNEL_MAX_INTERRUPT_PROBABILITY
                and probabilities.get("backchannel", 0) > probabilities.get("uncertain", 1)
                and len(compact) <= 30 and not any(marker in compact for marker in CONTINUATION_MARKERS))

    async def _judge(self, snapshot, *, preparation_review=False):
        if snapshot["lane"] == "preparation":
            await self._judge_preparation(snapshot, review_ambiguous=preparation_review)
            return
        previous_verdict = self.verdict
        result = None
        error_type = None
        self.judge_sequence += 1
        request_sequence = self.judge_sequence
        now = time.monotonic()
        timing = {"transcript_final": self._words_final(snapshot["lane"])}
        if snapshot["lane"] == "normal":
            timing = {"transcript_final": self._words_final("normal"),
                      "silence_ms": round(max(0.0, now-self.input_last_voice)*1000, 3),
                      "text_stable_ms": round(max(0.0, now-self.input_last_change)*1000, 3)}
        # Timing describes this request, not snapshot identity: elapsed time
        # must never invalidate a still-current classification on each tick.
        self.log("jev_judge_started", request_sequence=request_sequence,
                 lane=snapshot["lane"], input_chars=len(snapshot["text"]), **timing)
        try:
            result = await self.jev.classify(snapshot["text"],
                     assistant_text=snapshot["context"].get("text", ""),
                     speaking=snapshot["lane"] == "candidate", **timing)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            error_type = type(error).__name__
        if not self._current(snapshot):
            self.log("jev_result_discarded", request_sequence=request_sequence,
                     lane=snapshot["lane"], reason="input_or_playback_changed")
            return
        now = time.monotonic()
        if error_type:
            self.log("jev_failed", request_sequence=request_sequence,
                     error_type=error_type, lane=snapshot["lane"])
            self.verdict = {"snapshot": snapshot, "choice": "error", "due": now,
                            "backchannel_dominant": bool(previous_verdict
                                and self._current(previous_verdict["snapshot"])
                                and previous_verdict.get("backchannel_dominant"))}
            return
        self.log("jev_classification", request_sequence=request_sequence,
                 lane=snapshot["lane"], turn=result.get("turn"),
                 interruption=result.get("interruption"), request_ms=result.get("request_ms"))
        if snapshot["lane"] == "candidate":
            if self._safe_backchannel(snapshot["text"], result):
                self.verdict = {"snapshot": snapshot, "choice": "backchannel", "due": now}
                return
            interruption = (result.get("interruption") or {}).get("choice")
            if interruption == "interrupt":
                await self._promote_candidates("jev_interrupt")
                # The await can admit more text or resumed speech. Reuse the
                # turn prediction only if this exact recognised utterance stayed.
                fresh = self._snapshot("normal")
                if (fresh["text"] != snapshot["text"] or fresh["voice_epoch"] != snapshot["voice_epoch"]
                        or self.closed or self.jev_closing):
                    return
                snapshot = fresh
            else:
                self.verdict = {"snapshot": snapshot, "choice": "uncertain", "due": now + 1.0,
                                "backchannel_dominant": self._backchannel_dominant(snapshot["text"], result),
                                "transcript_final": timing["transcript_final"],
                                "request_sequence": request_sequence}
                return
        choice = (result.get("turn") or {}).get("choice")
        if choice not in ("complete", "short", "long"):
            self.verdict = {"snapshot": snapshot, "choice": "error", "due": now}
        else:
            wait = COMPLETE_GRACE_SECONDS if choice == "complete" else SHORT_WAIT_SECONDS if choice == "short" else LONG_WAIT_SECONDS
            due = now + wait if choice == "complete" else max(now + .5, max(self.input_last_voice, self.input_last_change) + wait)
            self.verdict = {"snapshot": snapshot, "choice": choice, "due": due,
                            "transcript_final": timing.get("transcript_final"),
                            "request_sequence": request_sequence}

    async def _judge_preparation(self, snapshot, *, review_ambiguous=False):
        if not self._current(snapshot) or not self._words_final("preparation") or self.input_transport_pending:
            return
        self.judge_sequence += 1
        request_sequence = self.judge_sequence
        now = time.monotonic()
        timing = {"transcript_final": True,
                  "silence_ms": round(max(0., now-self.input_last_voice)*1000, 3),
                  "text_stable_ms": round(max(0., now-self.input_last_change)*1000, 3)}
        self.log("jev_judge_started", request_sequence=request_sequence,
                 lane="preparation", input_chars=len(snapshot["text"]),
                 review_ambiguous=review_ambiguous, **timing)
        context = snapshot["context"]
        try:
            result = await self.jev.classify_preparation(
                snapshot["text"], original_text=context.get("original_text", ""),
                last_heard_text=context.get("last_heard_text", ""),
                **({"review_ambiguous": True} if review_ambiguous else {}), **timing)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._current(snapshot):
                self.log("jev_failed", request_sequence=request_sequence,
                         lane="preparation", error_type=type(error).__name__)
                self.verdict = {"snapshot": snapshot, "choice": "error", "due": time.monotonic()}
            return
        if not self._current(snapshot):
            self.log("jev_result_discarded", request_sequence=request_sequence,
                     lane="preparation", reason="input_or_reply_changed")
            return
        self.log("jev_preparation_classification", request_sequence=request_sequence,
                 preparation=result.get("preparation"), turn=result.get("turn"),
                 review_ambiguous=review_ambiguous, request_ms=result.get("request_ms"))
        decision = result.get("preparation") or {}
        choice = decision.get("choice")
        turn = (result.get("turn") or {}).get("choice")
        now = time.monotonic()
        probability = decision.get("probabilities", {}).get(choice, 0)
        if choice == "revise" and probability >= PREPARATION_REVISE_PROBABILITY:
            await self._promote_preparation("jev_new_content")
            fresh = self._snapshot("normal")
            if (fresh["text"] != snapshot["text"] or fresh["voice_epoch"] != snapshot["voice_epoch"]
                    or self.closed or self.jev_closing):
                return
            # The same request also judged the end of this extra utterance.
            # Preserve its exact text and reuse that answer after promotion.
            if turn in ("complete", "short", "long"):
                wait = COMPLETE_GRACE_SECONDS if turn == "complete" else SHORT_WAIT_SECONDS if turn == "short" else LONG_WAIT_SECONDS
                due = now+wait if turn == "complete" else max(now+.5, max(self.input_last_voice, self.input_last_change)+wait)
                self.verdict = {"snapshot": fresh, "choice": turn, "due": due,
                                "transcript_final": True, "request_sequence": request_sequence}
            return
        safe_retain = (choice == "retain" and turn == "complete"
                       and decision.get("probabilities", {}).get("retain", 0) >= PREPARATION_RETAIN_PROBABILITY)
        self.verdict = {"snapshot": snapshot, "choice": "retain" if safe_retain else "uncertain",
                        "due": now+(COMPLETE_GRACE_SECONDS if safe_retain else 1.0),
                        # One semantic review, not a periodic poll. A modest
                        # revise score must not immediately cancel ready audio;
                        # a modest retain score must not silently drop content.
                        "needs_review": (not review_ambiguous and choice in ("retain", "revise")
                                         and probability < PREPARATION_REVISE_PROBABILITY),
                        "request_sequence": request_sequence}

    def _start_judge(self, snapshot, *, preparation_review=False):
        self._processing.judge_started(snapshot, time.monotonic())
        self.next_judge_at = time.monotonic() + .5
        task = self.spawn(self._judge(snapshot, preparation_review=preparation_review), self.judge_tasks, "jev")
        self.judge_task = task

        def finished(done):
            if self.judge_task is done:
                self.judge_task = None
        task.add_done_callback(finished)

    async def _commit(self, snapshot, source, *, confirmed_noise=False):
        self._refresh_interruption_wait()
        if self.waiting_for_continuation:
            return False
        if (self.input_transport_pending or not self._current(snapshot)
                or not (self._words_final("normal") if confirmed_noise else self._all_final("normal"))
                or self.jobs.full()):
            return
        text = self.input_text
        if await self.enqueue(text, source) is False:
            return
        for row in self._rows("normal"):
            if row["text"] or not confirmed_noise:
                row["lane"] = "consumed"
            else:
                # Await late real ASR on its current floor, never discard it.
                row["lane"], row["context"] = "preparation", dict(self.preparation_context or {})
                row["noise_bypassed"] = True
        self.input_text = ""
        self.waiting_for_continuation = None
        self.input_interrupted_output = False
        self.buffer_version += 1
        self.verdict = None
        self.filler.schedule()

    async def _advance(self, now):
        """One deterministic timing step, kept separate for race regression tests."""
        if self.closed or self.jev_closing or self.finished.is_set() or self.promoting:
            return
        lane = "candidate" if self._text("candidate") else "preparation" if self._text("preparation") else "normal"
        if lane == "preparation":
            await self._advance_preparation(now)
            return
        snapshot = self._snapshot(lane)
        if not snapshot["text"]:
            return
        if lane == "normal":
            self._refresh_interruption_wait()
            if self.waiting_for_continuation:
                return
        # Candidate interruption is still actionable; only the EOT/ack verdict
        # is blocked by _all_final and _commit until captured audio catches up.
        if self.input_transport_pending and lane == "normal":
            return
        quiet = now - max(self.input_last_voice, self.input_last_change)
        acoustic_quiet = now - self.input_last_voice
        noise_recovery = self._turn_noise_recovery(lane, now)
        if ((lane == "normal" and acoustic_quiet < VAD_SECONDS and not self._words_final(lane))
                or now-self.input_last_change < STABLE_SECONDS):
            return
        verdict = self.verdict
        if verdict and not self._current(verdict["snapshot"]):
            self.verdict = verdict = None
        if verdict:
            choice = verdict["choice"]
            if (not self._all_final(lane) or acoustic_quiet < VAD_SECONDS) and not noise_recovery:
                return
            if noise_recovery:
                quiet = now-self.input_last_change
            if (choice in ("short", "long", "uncertain")
                    and verdict.get("transcript_final") is False
                    and self._words_final(lane)
                    and not verdict.get("final_recheck")):
                # A same-text final does not reset stability or invalidate an
                # in-flight answer. Refresh only the partial-based wait choice,
                # once, with final=True and current elapsed times. Keep the
                # existing recheck rate limit and eight-second fallback below.
                verdict["due"] = min(verdict["due"], now)
                verdict["final_recheck"] = True
                self.log("jev_final_recheck", lane=lane, choice=choice,
                         request_sequence=verdict.get("request_sequence"))
            if verdict.get("final_recheck"):
                if (now >= verdict["due"] and now >= self.next_judge_at
                        and self.judge_task is None):
                    # Keep the previous semantic evidence until the refresh
                    # finishes, and do not let a timeout race that refresh.
                    self._start_judge(snapshot)
                return
            if choice == "backchannel" and (acoustic_quiet >= VAD_SECONDS or noise_recovery):
                if noise_recovery: self._bypass_empty_rows()
                self._ignore_candidates("jev_high_confidence")
                return
            if choice == "complete" and now >= verdict["due"]:
                await self._commit(snapshot, "jev_complete", confirmed_noise=noise_recovery)
                return
            if choice == "error" and quiet >= FAILURE_WAIT_SECONDS:
                if lane == "candidate" and (normalized(snapshot["text"]) in SHORT_ACKS
                                             or verdict.get("backchannel_dominant")):
                    self._ignore_candidates("jev_failure_preserved_backchannel" if verdict.get("backchannel_dominant")
                                            else "jev_failure_short_ack")
                else:
                    if lane == "candidate":
                        await self._promote_candidates("jev_failure_fallback")
                        if self.input_text != snapshot["text"] or self.voice_epoch != snapshot["voice_epoch"]:
                            return
                    await self._commit(self._snapshot("normal"), "jev_failure_silence_fallback", confirmed_noise=noise_recovery)
                return
            if choice in ("short", "long", "uncertain") and quiet >= MAX_HOLD_SECONDS:
                if lane == "candidate" and (normalized(snapshot["text"]) in SHORT_ACKS
                                             or verdict.get("backchannel_dominant")):
                    self._ignore_candidates("bounded_backchannel_dominant" if verdict.get("backchannel_dominant")
                                            else "bounded_short_ack")
                else:
                    if lane == "candidate":
                        await self._promote_candidates("bounded_interruption_fallback")
                        if self.input_text != snapshot["text"] or self.voice_epoch != snapshot["voice_epoch"]:
                            return
                    self.log("jev_bounded_fallback", waited_seconds=quiet, last_choice=choice)
                    await self._commit(self._snapshot("normal"), "jev_bounded_silence_fallback", confirmed_noise=noise_recovery)
                return
            # Reuse a semantic decision until its text/context changes; the
            # single final-ASR refresh above is the only time-based recheck.
            return
        if self.judge_task is None and now >= self.next_judge_at:
            self._start_judge(snapshot)

    async def _advance_preparation(self, now):
        snapshot = self._snapshot("preparation")
        if snapshot["text"] and (self.preparation_context is None
                                 or not self.preparation_context.get("protect_reply", True)
                                 or snapshot["context"].get("id") != self.preparation_context["id"]):
            await self._promote_preparation("reply_no_longer_pending")
            self.next_judge_at = 0.
            return
        if (not snapshot["text"] or not self._words_final("preparation") or self.input_transport_pending
                or now-self.input_last_change < STABLE_SECONDS):
            return
        noise_recovery = self._turn_noise_recovery("preparation", now)
        verdict = self.verdict
        if verdict and not self._current(verdict["snapshot"]):
            self.verdict = verdict = None
        if verdict:
            choice = verdict["choice"]
            quiet = now-max(self.input_last_voice, self.input_last_change)
            if (not self._all_final("preparation") or now-self.input_last_voice < VAD_SECONDS) and not noise_recovery:
                return
            if noise_recovery:
                quiet = now-self.input_last_change
            if choice == "retain" and now >= verdict["due"]:
                if noise_recovery: self._bypass_empty_rows()
                self._retain_preparation()
                return
            if verdict.get("needs_review"):
                if self.judge_task is None and now >= self.next_judge_at:
                    self._start_judge(snapshot, preparation_review=True)
                return
            if ((choice == "error" and quiet >= FAILURE_WAIT_SECONDS)
                    or (choice == "uncertain" and quiet >= PREPARATION_MAX_HOLD_SECONDS)):
                await self._promote_preparation("jev_failure" if choice == "error" else "jev_uncertain_timeout")
                self.next_judge_at = 0.
                return
            # Preparation is classified only after final ASR. Keep an uncertain
            # result until new semantic evidence arrives or its fallback runs.
            return
        if self.judge_task is None and now >= self.next_judge_at:
            self._start_judge(snapshot)

    async def tick(self):
        counter = 0
        while True:
            await asyncio.sleep(.1)
            now = time.monotonic()
            await self._advance(now)
            if self.approved and now-self.approved_at > self.approved_timeout_seconds:
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
        if self.jev_closing:
            await self.jev_close_done.wait()
            return
        self.jev_closing = True
        tasks = list(self.judge_tasks)
        for task in tasks:
            task.cancel()
        try:
            await self.filler.close()
            await asyncio.gather(*tasks, return_exceptions=True)
            await super().close()
        finally:
            try:
                await self.jev.close()
                keys = {"model", "mode", "input_chars", "assistant_chars", "speaking", "status", "http_status",
                        "request_ms", "input_tokens", "output_tokens", "original_chars", "last_heard_chars", "reply_chars",
                        "review_ambiguous"}
                records = [{k: v for k, v in row.items() if k in keys}
                           for row in getattr(self.jev, "calls", []) if isinstance(row, dict)]
                (self.path / "jev_api_calls.json").write_text(
                    json.dumps(records, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            finally:
                self.jev_close_done.set()
