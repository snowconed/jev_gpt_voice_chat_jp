"""Optional cached turn fillers, independent of backend reply delivery.

The browser already owns the PCM. Only its clip ID crosses the live socket;
this controller never synthesizes, rewrites or commits a dialogue message.
"""
from __future__ import annotations

import asyncio
import random
import time
import uuid


FILLER_ACK_GRACE_SECONDS = 1.0


class FillerPlayback:
    def __init__(self, runtime, clips=None):
        self.runtime = runtime
        self.clips = tuple(dict(clip) for clip in (clips or ()))
        self.active = None
        self.last_revision = None
        self.last_clip_id = None
        self.ready_turn = None
        self.tasks = set()
        self.closed = False

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine, name="voice_filler")
        self.tasks.add(task)

        def completed(done):
            self.tasks.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error:
                    self.runtime.log("filler_failed", kind="filler", error_type=type(error).__name__)
        task.add_done_callback(completed)
        return task

    def _current(self, state):
        r = self.runtime
        return (not self.closed and not r.closed and not r.jev_closing
                and not r.finished.is_set() and not state["cancelled"]
                and self.active is state and state["revision"] == r.input_revision
                and state["generation"] == r.output_generation and not r.input_text
                and not r.waiting_for_continuation and not r.playing)

    def schedule(self):
        r = self.runtime
        if not self.clips or self.last_revision == r.input_revision:
            return
        self.last_revision = r.input_revision
        if (self.active or r.input_text or r.playing or r.waiting_for_continuation
                or self.ready_turn == (r.input_revision, r.output_generation)
                or self.closed or r.closed or r.jev_closing or r.finished.is_set()):
            return
        choices = [clip for clip in self.clips if clip["id"] != self.last_clip_id] or self.clips
        clip = random.choice(choices)
        state = {"id": "filler_" + uuid.uuid4().hex, "clip_id": clip["id"],
                 "text": clip["text"], "audio_bytes": clip["audio_bytes"],
                 "audio_seconds": clip["audio_seconds"], "revision": r.input_revision,
                 "generation": r.output_generation, "sent_at": None, "cancelled": False,
                 "done": asyncio.Event(), "dispatched": asyncio.Event(), "timeout": None}
        self.active = state
        r.log("filler_scheduled", kind="filler", id=state["id"], clip_id=state["clip_id"],
              revision=state["revision"])
        self._spawn(self._dispatch(state))

    async def _dispatch(self, state):
        r = self.runtime
        try:
            if (not self._current(state)
                    or self.ready_turn == (state["revision"], state["generation"])):
                self._finish(state, "skipped", reason="reply_ready_or_turn_changed")
                return
            state["sent_at"] = time.monotonic()
            self.last_clip_id = state["clip_id"]
            r.log("filler_sent", kind="filler", id=state["id"], clip_id=state["clip_id"],
                  revision=state["revision"], text=state["text"], audio_seconds=state["audio_seconds"])
            await r.send_json({"type": "filler", **{key: state[key] for key in
                              ("id", "clip_id", "text", "audio_bytes", "audio_seconds", "revision")}})
            if not state["done"].is_set() and not self._current(state):
                # An interruption can overtake a blocked socket send. Stop
                # again after that send so a late filler cannot restart audio.
                await r.send_json({"type": "filler_stop", "id": state["id"], "reason": "turn_changed"})
                self._finish(state, "stopped", reason="turn_changed")
            elif not state["done"].is_set():
                state["timeout"] = self._spawn(self._watchdog(state))
        except asyncio.CancelledError:
            self._finish(state, "stopped", reason="closed")
            raise
        except Exception as error:
            # A filler is optional. Its failure must never discard a valid
            # backend reply or make the user start another conversation.
            r.log("filler_failed", kind="filler", id=state["id"], error_type=type(error).__name__)
            try:
                if state["sent_at"] is not None:
                    # A send may fail after the browser received the command.
                    # Try to clear that audio before releasing the main reply;
                    # this optional notification must not block it indefinitely.
                    try:
                        await asyncio.wait_for(r.send_json({"type": "filler_stop",
                            "id": state["id"], "reason": "send_failed"}), .25)
                    except Exception as stop_error:
                        r.log("filler_stop_failed", kind="filler", id=state["id"],
                              error_type=type(stop_error).__name__)
            finally:
                self._finish(state, "skipped", reason="send_failed")
        finally:
            state["dispatched"].set()

    async def _watchdog(self, state):
        try:
            await asyncio.wait_for(state["done"].wait(),
                                   state["audio_seconds"] + FILLER_ACK_GRACE_SECONDS)
        except TimeoutError:
            await self.stop("ack_timeout", state=state)

    def _finish(self, state, outcome, *, reason=None):
        if state["done"].is_set():
            return
        if self.active is state:
            self.active = None
        state["done"].set()
        timer = state["timeout"]
        if timer and timer is not asyncio.current_task():
            timer.cancel()
        self.runtime.log("filler_" + outcome, kind="filler", id=state["id"],
                         clip_id=state["clip_id"], revision=state["revision"],
                         **({"reason": reason} if reason else {}))

    def invalidate(self):
        if self.active:
            self.active["cancelled"] = True

    async def stop(self, reason, *, state=None, notify=True):
        state = state or self.active
        if not state or state["done"].is_set():
            return
        state["cancelled"] = True
        await state["dispatched"].wait()
        if state["done"].is_set():
            return
        try:
            if notify and state["sent_at"] is not None:
                await self.runtime.send_json({"type": "filler_stop", "id": state["id"], "reason": reason})
        finally:
            self._finish(state, "stopped", reason=reason)

    async def played(self, playback_id):
        state = self.active
        if not state or state["id"] != playback_id:
            return False
        if (state["sent_at"] is None
                or time.monotonic() + .15 < state["sent_at"] + state["audio_seconds"]):
            self.runtime.log("filler_early_played_ignored", kind="filler", id=playback_id)
            return True
        self._finish(state, "completed")
        return True

    async def skipped(self, playback_id):
        state = self.active
        if state and state["id"] == playback_id:
            self._finish(state, "skipped", reason="browser_unavailable")

    async def before_reply(self, message, revision, generation):
        """TTS is ready. Wait only for a filler that has actually been sent."""
        self.ready_turn = (revision, generation)
        state = self.active
        if state and state["sent_at"] is None:
            state["cancelled"] = True
            await state["dispatched"].wait()
        while state and not state["done"].is_set():
            if not self.runtime._speech_current(message, revision, generation):
                return False
            await asyncio.sleep(.02)
        return self.runtime._speech_current(message, revision, generation)

    async def close(self):
        self.closed = True
        self.invalidate()
        if self.active:
            try:
                await asyncio.wait_for(self.stop("closed"), .25)
            except (Exception, asyncio.CancelledError):
                if self.active:
                    self._finish(self.active, "stopped", reason="closed")
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
