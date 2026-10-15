"""Both input transports expose a blocked write without logging its content."""
import asyncio
from contextlib import suppress
import time
import unittest
from unittest.mock import patch

from live_client import LiveClient
from transcription_client import TranscriptionClient


class BlockingSocket:
    closed = False
    compress = 15

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.sent = []

    async def send_json(self, event):
        self.entered.set()
        await self.release.wait()
        self.sent.append(event)


class SendTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_write_and_waiters_are_observable_and_clear_after_completion(self):
        for constructor in (LiveClient, TranscriptionClient):
            with self.subTest(client=constructor.__name__):
                client = constructor()
                client.ws = BlockingSocket()
                first = asyncio.create_task(client.send({"type": "first", "secret": "hidden"}))
                await client.ws.entered.wait()
                second = asyncio.create_task(client.send({"type": "second"}))
                await asyncio.sleep(.02)
                state = client.transport_snapshot()
                self.assertEqual(state["send_phase"], "socket")
                self.assertEqual(state["send_waiters"], 1)
                self.assertGreaterEqual(state["send_pending_ms"], 15)
                self.assertNotIn("hidden", str(state))
                client.ws.release.set()
                await asyncio.gather(first, second)
                self.assertEqual([row["type"] for row in client.ws.sent], ["first", "second"])
                self.assertEqual(client.transport_snapshot()["send_phase"], "idle")
                self.assertEqual(client.transport_snapshot()["send_waiters"], 0)

    async def test_timeout_records_pending_failure_and_releases_lock_for_both_clients(self):
        timeout_seconds = .06
        # asyncio schedules with monotonic(), while transport diagnostics use
        # perf_counter(). On this Python/Windows runtime their resolutions are
        # 15.625 ms and 0.0001 ms respectively, so the measured wait may fall one
        # monotonic tick below the deadline. Retain the existing 10 ms allowance
        # on finer clocks; the timeout and cleanup assertions remain exact.
        clock_allowance = max(.01, time.get_clock_info("monotonic").resolution)
        minimum_wait_ms = (timeout_seconds-clock_allowance)*1000
        for constructor in (LiveClient, TranscriptionClient):
            with self.subTest(client=constructor.__name__):
                client = constructor()
                client.ws = BlockingSocket()
                with patch("websocket_send.SEND_TIMEOUT_SECONDS", timeout_seconds):
                    with self.assertRaises(RuntimeError):
                        await client.audio(bytes(4800))
                self.assertEqual(client.transport_metrics["send_timeouts"], 1)
                self.assertEqual(client.transport_metrics["audio_bytes_sent"], 0)
                state = client.transport_metrics["last_send_timeout"]
                self.assertEqual(state["send_phase"], "socket")
                self.assertGreaterEqual(state["send_pending_ms"], minimum_wait_ms)
                self.assertGreaterEqual(client.transport_metrics["max_audio_send_ms"], minimum_wait_ms)
                self.assertFalse(client._send_lock.locked())
                self.assertEqual(client._pending_sends, {})

    async def test_cancellation_records_wait_and_leaves_no_pending_send(self):
        for constructor in (LiveClient, TranscriptionClient):
            client = constructor()
            client.ws = BlockingSocket()
            task = asyncio.create_task(client.audio(bytes(4800)))
            await client.ws.entered.wait()
            await asyncio.sleep(.02)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self.assertEqual(client._pending_sends, {})
            self.assertFalse(client._send_lock.locked())
            self.assertEqual(client.transport_metrics["send_timeouts"], 0)
            self.assertGreaterEqual(client.transport_metrics["max_audio_send_ms"], 15)


if __name__ == "__main__":
    unittest.main()
