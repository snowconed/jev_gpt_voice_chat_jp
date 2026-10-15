"""API-free recovery, ordering, rate, bound and cleanup regression tests."""
import asyncio
from contextlib import suppress
import struct
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import microphone_forwarder as module
from microphone_forwarder import MicrophoneForwarder


class Runtime:
    def __init__(self, *, stall_call=None, stall_seconds=0, wait=None, clock=None):
        self.ready = True
        self.failure_message = None
        self.finished = asyncio.Event()
        self.client = SimpleNamespace(transport_metrics={}, transport_snapshot=lambda: {
            "send_phase": "socket", "send_pending_ms": 123,
            "protocol_paused": True, "tls_write_buffer_bytes": 128,
            "Authorization": "never-log-secret", "unexpected": "private body"})
        self.stall_call, self.stall_seconds, self.wait = stall_call, stall_seconds, wait
        self.clock = clock or time.perf_counter
        self.starts = []
        self.audio = bytearray()
        self.logs = []

    def log(self, kind, **data):
        self.logs.append({"type": kind, **data})

    async def microphone(self, pcm):
        self.starts.append((self.clock(), len(pcm)))
        if self.wait is not None:
            await self.wait.wait()
        if len(self.starts) == self.stall_call:
            await asyncio.sleep(self.stall_seconds)
        self.audio.extend(pcm)


async def cancel(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class Clock:
    def __init__(self):
        self.now = 100.0

    def time(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_2300ms_stall_recovers_all_pcm_without_drop_or_reordering(self):
        runtime = Runtime(stall_call=11, stall_seconds=2.3)
        forwarder = MicrophoneForwarder(runtime)
        # Observe the dispatch timestamp that pacing actually constrains.
        # Callback entry follows asyncio.timeout setup and can be delayed by
        # scheduling, making callback-to-callback gaps spuriously too short.
        runtime.clock = lambda: forwarder.last_send_started
        task = asyncio.create_task(forwarder.run())
        self.addAsyncCleanup(cancel, task)
        start = time.perf_counter()
        expected = bytearray()
        # Fourteen seconds permits gradual 1.25x recovery plus a steady tail.
        for i in range(700):
            await asyncio.sleep(max(0, start + (i + 1) * .02 - time.perf_counter()))
            pcm = struct.pack("<h", i) * 480
            expected.extend(pcm)
            forwarder.push(pcm)
        async with asyncio.timeout(2):
            while forwarder.total_backlog_bytes:
                await asyncio.sleep(.02)
        self.assertEqual(runtime.audio, expected)
        self.assertEqual(forwarder.metrics["accepted_bytes"], len(expected))
        self.assertEqual(forwarder.metrics["forwarded_bytes"], len(expected))
        self.assertEqual(runtime.microphone_backlog_bytes, 0)
        self.assertLess(forwarder.metrics["max_total_backlog_ms"], 6000)
        self.assertGreater(forwarder.metrics["max_send_wait_ms"], 2200)
        self.assertGreater(forwarder.metrics["catchup_frames"], 0)
        self.assertFalse(forwarder.catchup)
        self.assertIsNone(runtime.failure_message)
        self.assertFalse(runtime.finished.is_set())
        for previous, current in zip(runtime.starts, runtime.starts[1:]):
            self.assertGreaterEqual(current[0] - previous[0] + 1e-8, previous[1] / 48000 / 1.25)
        tail = [b[0] - a[0] for a, b in zip(runtime.starts[-10:], runtime.starts[-9:])]
        self.assertAlmostEqual(sum(tail) / len(tail), .1, delta=.015)
        self.assertTrue(any(row.get("reason") == "send_wait" for row in runtime.logs))
        self.assertNotIn("never-log-secret", str(runtime.logs))
        self.assertNotIn("Authorization", str(runtime.logs))
        await cancel(task)
        self.assertTrue(forwarder.monitor_task.done())
        self.assertEqual(forwarder.metrics["abandoned_on_stop_bytes"], 0)

    async def test_six_second_bound_includes_pending_and_inflight(self):
        runtime = Runtime(wait=asyncio.Event())
        forwarder = MicrophoneForwarder(runtime)
        # One 1s packet yields 100ms in flight and 900ms locally pending.
        forwarder.push(bytes(48000))
        task = asyncio.create_task(forwarder.run())
        self.addAsyncCleanup(cancel, task)
        async with asyncio.timeout(1):
            while forwarder.phase != "sending":
                await asyncio.sleep(0)
        self.assertEqual(forwarder.inflight_bytes, 4800)
        self.assertEqual(forwarder.pending_bytes, 43200)
        for _ in range(5):
            forwarder.push(bytes(48000))
        self.assertEqual(forwarder.total_backlog_bytes, 288000)
        self.assertEqual(forwarder.queued_bytes, 240000)
        with self.assertRaises(ValueError):
            forwarder.push(b"\x00\x00")
        self.assertEqual(forwarder.metrics["accepted_bytes"], 288000)
        self.assertTrue(runtime.finished.is_set())
        await cancel(task)
        self.assertTrue(forwarder.monitor_task.done())
        self.assertEqual(forwarder.queued_bytes + forwarder.pending_bytes + forwarder.inflight_bytes, 0)
        self.assertEqual(forwarder.metrics["abandoned_on_stop_bytes"], 288000)

    async def test_received_voice_stays_visible_while_network_send_is_blocked(self):
        runtime = Runtime(wait=asyncio.Event())
        runtime.CAPTURE_VOICE_RMS = 280
        runtime.input_last_voice = 0
        forwarder = MicrophoneForwarder(runtime)
        forwarder.push(bytes(4800))
        task = asyncio.create_task(forwarder.run())
        self.addAsyncCleanup(cancel, task)
        async with asyncio.timeout(1):
            while forwarder.phase != "sending":
                await asyncio.sleep(0)
        before = time.monotonic()
        forwarder.push(struct.pack("<h", 400) * 480)
        self.assertGreaterEqual(runtime.input_last_voice, before)
        self.assertEqual(runtime.microphone_backlog_bytes, 5760)
        self.assertFalse(runtime.wait.is_set())
        # Dedicated ASR uses its existing higher acoustic threshold.
        runtime.CAPTURE_VOICE_RMS = 560
        runtime.input_last_voice = 0
        forwarder.push(struct.pack("<h", 400) * 480)
        self.assertEqual(runtime.input_last_voice, 0)

    async def test_timeout_includes_failed_wait_and_monitor_is_reaped(self):
        self.assertEqual(module.SEND_TIMEOUT_SECONDS, 5.25)
        runtime = Runtime(wait=asyncio.Event())
        forwarder = MicrophoneForwarder(runtime)
        forwarder.push(bytes(4800))
        with patch.object(module, "SEND_TIMEOUT_SECONDS", .08), \
                patch.object(module, "MONITOR_SECONDS", .01), \
                patch.object(module, "SLOW_SEND_SECONDS", .02):
            with self.assertRaises(RuntimeError):
                await forwarder.run()
        self.assertGreaterEqual(forwarder.metrics["max_send_wait_ms"], 60)
        self.assertEqual(forwarder.metrics["send_timeouts"], 1)
        self.assertEqual(forwarder.metrics["forwarded_bytes"], 0)
        self.assertTrue(runtime.finished.is_set())
        self.assertTrue(forwarder.monitor_task.done())
        waits = [x for x in runtime.logs if x.get("reason") == "send_wait"]
        self.assertTrue(waits)
        self.assertEqual(waits[0]["transport"]["send_phase"], "socket")
        self.assertTrue(waits[0]["transport"]["protocol_paused"])

    async def test_normal_one_x_and_catchup_rate_with_clock(self):
        clock = Clock()
        runtime = Runtime(clock=clock.time)
        forwarder = MicrophoneForwarder(runtime, clock=clock.time, sleep=clock.sleep)
        # Keep just one captured frame pending in the ordinary path.
        for _ in range(5):
            forwarder.metrics["accepted_bytes"] += 4800
            await forwarder._forward(bytes(4800))
        gaps = [b[0] - a[0] for a, b in zip(runtime.starts, runtime.starts[1:])]
        self.assertTrue(all(abs(gap - .1) < 1e-8 for gap in gaps))
        # Captured data backlogs during a hypothetical transport delay.
        clock.now += 2.3
        forwarder.metrics["accepted_bytes"] += 48000
        for _ in range(10):
            await forwarder._forward(bytes(4800))
        gaps = [b[0] - a[0] for a, b in zip(runtime.starts[5:], runtime.starts[6:])]
        self.assertTrue(all(gap + 1e-8 >= .08 for gap in gaps))
        self.assertFalse(forwarder.catchup)
        self.assertEqual(forwarder.total_backlog_bytes, 0)
        self.assertAlmostEqual(gaps[-1], .1)

    async def test_partial_frame_flush_and_cancel_leave_no_monitor(self):
        runtime = Runtime()
        forwarder = MicrophoneForwarder(runtime)
        pcm = b"\x01\x00" * 480
        forwarder.push(pcm)
        task = asyncio.create_task(forwarder.run())
        self.addAsyncCleanup(cancel, task)
        async with asyncio.timeout(1):
            while forwarder.total_backlog_bytes:
                await asyncio.sleep(.01)
        self.assertEqual(bytes(runtime.audio), pcm)
        await cancel(task)
        self.assertTrue(forwarder.monitor_task.done())
        self.assertFalse(any(t.get_name() == "microphone_monitor" and not t.done()
                             for t in asyncio.all_tasks()))

    async def test_validation_empty_and_pre_ready(self):
        runtime = Runtime()
        forwarder = MicrophoneForwarder(runtime)
        forwarder.push(b"")
        for invalid in (b"x", bytes(48002), "text", None):
            with self.assertRaises(ValueError):
                forwarder.push(invalid)
        self.assertEqual(forwarder.metrics["browser_received_bytes"], 0)
        runtime.ready = False
        forwarder.push(bytes(960))
        self.assertEqual(forwarder.metrics["dropped_before_ready_bytes"], 960)
        self.assertEqual(forwarder.total_backlog_bytes, 0)
        self.assertTrue(forwarder.queue.empty())

    async def test_diagnostics_keep_event_counts_and_safe_timeout_phase(self):
        source = {"received_events": {"session.input_transcript.delta": 12},
                  "last_send_timeout": {"send_phase": "socket", "send_pending_ms": 5001,
                                        "raw_error": "never-log-secret"}}
        sanitized = module._safe_transport(source)
        self.assertEqual(sanitized["received_events"]["session.input_transcript.delta"], 12)
        self.assertEqual(sanitized["last_send_timeout"]["send_phase"], "socket")
        self.assertNotIn("never-log-secret", str(sanitized))
        runtime = Runtime()
        runtime.client.transport_snapshot = lambda: source
        forwarder = MicrophoneForwarder(runtime)
        self.assertEqual(forwarder._transport()["last_send_timeout"]["send_phase"], "socket")


if __name__ == "__main__":
    unittest.main()
