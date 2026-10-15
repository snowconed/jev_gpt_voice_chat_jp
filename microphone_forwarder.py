"""Bounded microphone transport with observable waits and gradual recovery.

This tolerates temporary downstream stalls; it does not fix their network cause.
No samples are discarded or reordered during operation. Browser controls remain
independent because push never awaits the downstream transport.
"""
from __future__ import annotations

import asyncio
import math
import time

from voice_runtime import rms

# Give the client's detailed 5s timeout a chance to finish its diagnostic first.
SEND_TIMEOUT_SECONDS = 5.25
MONITOR_SECONDS = 1.0
SLOW_SEND_SECONDS = 1.0


def _safe_transport(value, depth=0):
    """Transport diagnostics contain numeric state only, never remote strings."""
    if depth > 3:
        return None
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if (not isinstance(key, str) or len(key) > 64
                    or not key.replace("_", "").replace(".", "").isalnum()):
                continue
            if key == "send_phase" and isinstance(item, str) and item in {"idle", "wait_lock", "socket"}:
                result[key] = item
            elif item is None or type(item) in (bool, int, float) or isinstance(item, dict):
                result[key] = _safe_transport(item, depth + 1)
        return result
    return None


class MicrophoneForwarder:
    FRAME_BYTES = 4800
    MAX_QUEUED_BYTES = 48000 * 6  # Total accepted, not yet successfully sent PCM.
    CATCHUP_RATE = 1.25
    CATCHUP_START_SECONDS = .3
    CATCHUP_STOP_SECONDS = .15

    def __init__(self, runtime, *, clock=None, sleep=None):
        self.runtime = runtime
        self.runtime.microphone_backlog_bytes = 0
        self._clock = clock or time.perf_counter
        self._sleep = sleep or asyncio.sleep
        self.queue = asyncio.Queue()
        self.queued_bytes = 0
        self.pending_bytes = 0
        self.inflight_bytes = 0
        self.started = self._clock()
        self.last_log = 0.0
        self.next_send = None
        self.last_send_started = None
        self.last_frame_seconds = 0.0
        self.catchup = False
        self.phase = "waiting_input"
        self.phase_started = self.started
        self.send_started = None
        self.monitor_task = None
        self.stopped = False
        self.metrics = {
            "browser_received_bytes": 0, "browser_frames": 0,
            "forwarded_bytes": 0, "forwarded_frames": 0,
            "dropped_before_ready_bytes": 0, "max_queued_audio_ms": 0.0,
            "max_send_wait_ms": 0.0, "max_rms": 0.0, "loud_audio_ms": 0.0,
            "accepted_bytes": 0, "max_total_backlog_ms": 0.0,
            "max_inflight_wait_ms": 0.0, "send_timeouts": 0,
            "catchup_frames": 0, "catchup_entries": 0,
            "abandoned_on_stop_bytes": 0,
        }

    @property
    def total_backlog_bytes(self):
        return self.metrics["accepted_bytes"] - self.metrics["forwarded_bytes"]

    def _phase(self, value):
        self.phase, self.phase_started = value, self._clock()

    def push(self, pcm):
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > 48000:
            raise ValueError("マイク音声の形式が不正です。")
        if not pcm:
            return
        if self.stopped:
            raise ValueError("マイク転送は終了しています。")
        self.metrics["browser_received_bytes"] += len(pcm)
        self.metrics["browser_frames"] += 1
        if not self.runtime.ready:
            self.metrics["dropped_before_ready_bytes"] += len(pcm)
            return
        if self.total_backlog_bytes + len(pcm) > self.MAX_QUEUED_BYTES:
            self.log(force=True, reason="queue_overflow")
            self.runtime.failure_message = "マイク音声の送信待ちが上限を超えたため停止しました。新しい会話を開始してください。"
            self.runtime.finished.set()
            raise ValueError("microphone backlog exceeded six seconds")
        self.queue.put_nowait(pcm)
        self.queued_bytes += len(pcm)
        self.metrics["accepted_bytes"] += len(pcm)
        self.runtime.microphone_backlog_bytes = self.total_backlog_bytes
        # A pending socket write must not conceal speech still arriving from
        # the browser. Use the runtime's clock and acoustic threshold.
        if rms(pcm) >= getattr(self.runtime, "CAPTURE_VOICE_RMS", 560):
            self.runtime.input_last_voice = self.runtime.last_activity = time.monotonic()
        self.metrics["max_queued_audio_ms"] = max(self.metrics["max_queued_audio_ms"], self.queued_bytes / 48)
        self.metrics["max_total_backlog_ms"] = max(self.metrics["max_total_backlog_ms"], self.total_backlog_bytes / 48)

    def _transport(self):
        try:
            snapshot = getattr(self.runtime.client, "transport_snapshot", lambda: {})()
            keys = {"send_pending_ms", "send_waiters", "ws_closed", "ws_compression",
                    "protocol_paused", "tls_write_buffer_bytes", "tcp_write_buffer_bytes",
                    "transport_closing", "last_send_timeout"}
            result = {key: _safe_transport(snapshot[key]) for key in keys if key in snapshot}
            if snapshot.get("send_phase") in {"idle", "wait_lock", "socket"}:
                result["send_phase"] = snapshot["send_phase"]
            return result
        except Exception:
            return {"snapshot_failed": True}

    def log(self, *, force=False, reason="periodic"):
        now = self._clock()
        if not force and self.last_log and now - self.last_log < 2:
            return
        self.last_log = now
        sending_ms = (now - self.send_started) * 1000 if self.send_started is not None else 0.0
        self.runtime.log("microphone_transport", **self.metrics,
            elapsed_ms=round((now - self.started) * 1000, 3),
            browser_audio_ms=round(self.metrics["browser_received_bytes"] / 48, 3),
            forwarded_audio_ms=round(self.metrics["forwarded_bytes"] / 48, 3),
            queued_audio_ms=round(self.queued_bytes / 48, 3),
            pending_audio_ms=round(self.pending_bytes / 48, 3),
            inflight_audio_ms=round(self.inflight_bytes / 48, 3),
            total_backlog_ms=round(self.total_backlog_bytes / 48, 3),
            inflight_wait_ms=round(sending_ms, 3), phase=self.phase,
            phase_elapsed_ms=round((now - self.phase_started) * 1000, 3),
            catchup=self.catchup, target_rate=self.CATCHUP_RATE if self.catchup else 1.0,
            reason=reason, transport=self._transport())

    async def _monitor(self):
        previous = self._clock()
        while True:
            await asyncio.sleep(MONITOR_SECONDS)
            now = self._clock()
            waiting = now - self.send_started if self.send_started is not None else 0.0
            self.metrics["max_inflight_wait_ms"] = max(self.metrics["max_inflight_wait_ms"], round(waiting * 1000, 3))
            self.runtime.log("microphone_monitor", phase=self.phase,
                loop_delay_ms=round(max(0, now - previous - MONITOR_SECONDS) * 1000, 3),
                phase_elapsed_ms=round((now - self.phase_started) * 1000, 3),
                inflight_wait_ms=round(waiting * 1000, 3),
                total_backlog_ms=round(self.total_backlog_bytes / 48, 3),
                transport=self._transport())
            if waiting >= SLOW_SEND_SECONDS:
                self.log(force=True, reason="send_wait")
            previous = now

    async def _forward(self, pcm):
        seconds = len(pcm) / 48000
        backlog = self.total_backlog_bytes / 48000
        if not self.catchup and backlog > self.CATCHUP_START_SECONDS:
            self.catchup = True
            self.metrics["catchup_entries"] += 1
        elif self.catchup and backlog <= self.CATCHUP_STOP_SECONDS:
            self.catchup = False
        now = self._clock()
        rate = self.CATCHUP_RATE if self.catchup else 1.0
        due = now if self.last_send_started is None else self.last_send_started + self.last_frame_seconds / rate
        self.next_send = due
        self.inflight_bytes = len(pcm)
        self._phase("pacing")
        # asyncio's Windows timer may wake slightly early; never exceed the cap.
        while (wait := due - self._clock()) > 0:
            await self._sleep(wait)
        if self.catchup:
            self.metrics["catchup_frames"] += 1
        level = rms(pcm)
        self.metrics["max_rms"] = max(self.metrics["max_rms"], round(level, 3))
        if level >= 560:
            self.metrics["loud_audio_ms"] += len(pcm) / 48
        self._phase("sending")
        started = self._clock()
        self.send_started = self.last_send_started = started
        self.last_frame_seconds = seconds
        try:
            async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
                await self.runtime.microphone(pcm)
            self.metrics["forwarded_bytes"] += len(pcm)
            self.metrics["forwarded_frames"] += 1
            self.runtime.microphone_backlog_bytes = self.total_backlog_bytes
        except TimeoutError:
            self.metrics["send_timeouts"] += 1
            self.runtime.failure_message = "マイク音声の送信が5秒以上停止したため、接続を終了しました。新しい会話を開始してください。"
            raise RuntimeError("microphone downstream timeout") from None
        finally:
            waiting_ms = round((self._clock() - started) * 1000, 3)
            self.metrics["max_send_wait_ms"] = max(self.metrics["max_send_wait_ms"], waiting_ms)
            self.metrics["max_inflight_wait_ms"] = max(self.metrics["max_inflight_wait_ms"], waiting_ms)
            self.send_started = None
            self.inflight_bytes = 0
        self._phase("waiting_input")
        self.log()

    async def run(self):
        pending = bytearray()
        self.monitor_task = asyncio.create_task(self._monitor(), name="microphone_monitor")
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(self.queue.get(), .1)
                except TimeoutError:
                    if pending:
                        frame = bytes(pending)
                        pending.clear()
                        self.pending_bytes = 0
                        await self._forward(frame)
                    else:
                        self.log()
                    continue
                self.queued_bytes -= len(chunk)
                pending.extend(chunk)
                self.pending_bytes = len(pending)
                while len(pending) >= self.FRAME_BYTES:
                    frame = bytes(pending[:self.FRAME_BYTES])
                    del pending[:self.FRAME_BYTES]
                    self.pending_bytes = len(pending)
                    await self._forward(frame)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.runtime.log("microphone_forwarding_error", error_type=type(error).__name__)
            self.runtime.failure_message = self.runtime.failure_message or "マイク音声を送信できませんでした。新しい会話を開始してください。"
            self.runtime.finished.set()
            raise
        finally:
            self.stopped = True
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)
            self.metrics["abandoned_on_stop_bytes"] = self.total_backlog_bytes
            self._phase("stopped")
            self.log(force=True, reason="stopped")
            pending.clear()
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queued_bytes = self.pending_bytes = self.inflight_bytes = 0
            self.runtime.log("live_transport", **_safe_transport(getattr(self.runtime.client, "transport_metrics", {})))
