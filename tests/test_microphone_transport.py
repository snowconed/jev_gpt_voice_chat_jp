"""No-API microphone pacing and browser-control isolation tests."""
import asyncio
from pathlib import Path
import socket
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from server import MicrophoneForwarder, create_app


class Runtime:
    instances = []

    def __init__(self, *args, **kwargs):
        self.ready = True
        self.finished = asyncio.Event()
        self.failure_message = None
        self.client = SimpleNamespace(transport_metrics={"audio_frames_sent": 0})
        self.frames, self.times, self.logs = [], [], []
        self.entered = asyncio.Event()
        self.release = None
        self.ack = asyncio.Event()
        self.closed = asyncio.Event()
        self.playing = None
        self.send_json = args[2] if args else None
        self.instances.append(self)

    async def microphone(self, pcm):
        self.entered.set()
        if self.release:
            await self.release.wait()
        self.frames.append(pcm)
        self.times.append(time.monotonic())

    def log(self, kind, **fields):
        self.logs.append({"type": kind, **fields})

    async def start(self):
        self.ready = True
        self.release = asyncio.Event()
        await self.send_json({"type": "ready"})

    async def played(self, mid):
        self.ack.set()

    async def close(self):
        self.closed.set()


class ForwarderTests(unittest.IsolatedAsyncioTestCase):
    async def test_20ms_frames_are_batched_and_paced(self):
        runtime = Runtime()
        forwarder = MicrophoneForwarder(runtime)
        pcm = b"\x00\x04" * 480
        for _ in range(15):
            forwarder.push(pcm)
        task = asyncio.create_task(forwarder.run())
        try:
            async with asyncio.timeout(2):
                while len(runtime.frames) < 3:
                    await asyncio.sleep(.01)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(runtime.frames, [pcm*5]*3)
        self.assertGreaterEqual(runtime.times[2]-runtime.times[0], .18)
        self.assertEqual(forwarder.metrics["forwarded_bytes"], len(pcm)*15)
        self.assertEqual(forwarder.metrics["browser_received_bytes"], len(pcm)*15)
        self.assertGreater(forwarder.metrics["max_rms"], 560)
        self.assertTrue(any(e["type"] == "microphone_transport" for e in runtime.logs))

    async def test_partial_frame_flushes_without_truncation(self):
        runtime = Runtime()
        forwarder = MicrophoneForwarder(runtime)
        pcm = b"\x01\x02" * 400
        forwarder.push(pcm)
        task = asyncio.create_task(forwarder.run())
        try:
            await asyncio.wait_for(runtime.entered.wait(), 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(runtime.frames, [pcm])

    async def test_not_ready_is_counted_and_not_replayed(self):
        runtime = Runtime()
        runtime.ready = False
        forwarder = MicrophoneForwarder(runtime)
        forwarder.push(bytes(960))
        self.assertTrue(forwarder.queue.empty())
        self.assertEqual(forwarder.metrics["dropped_before_ready_bytes"], 960)

    async def test_queue_overflow_is_explicit_and_preserves_existing_queue(self):
        runtime = Runtime()
        forwarder = MicrophoneForwarder(runtime)
        for _ in range(6):
            forwarder.push(bytes(48000))
        with self.assertRaises(ValueError):
            forwarder.push(bytes(960))
        self.assertTrue(runtime.finished.is_set())
        self.assertIn("送信待ちが上限", runtime.failure_message)
        self.assertEqual(forwarder.queued_bytes, 288000)

    async def test_cancel_interrupts_a_blocked_send(self):
        runtime = Runtime()
        runtime.release = asyncio.Event()
        forwarder = MicrophoneForwarder(runtime)
        forwarder.push(bytes(4800))
        task = asyncio.create_task(forwarder.run())
        await asyncio.wait_for(runtime.entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        self.assertEqual(runtime.frames, [])
        self.assertTrue(any(e["type"] == "microphone_transport" and e["reason"] == "stopped" for e in runtime.logs))


class Core:
    def view(self, sid):
        return {"id": sid, "messages": []}

    def close(self):
        pass


class BrowserControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_ack_and_stop_do_not_wait_for_mic_send(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        app = create_app(Path(__file__).resolve().parents[1] / ".runtime/http_test", port=port, core=Core(), turn_mode="jev")
        client = TestClient(TestServer(app, host="127.0.0.1", port=port))
        await client.start_server()
        try:
            config = await (await client.get("/api/config")).json()
            self.assertEqual(config["version"], "1.0.0")
            with patch("server.JevSpeechRuntime", Runtime):
                ws = await client.ws_connect("/api/voice/" + "a"*32 + "?token=" + config["csrf_token"])
                self.assertEqual((await ws.receive_json())["type"], "ready")
                runtime = Runtime.instances[-1]
                await ws.send_bytes(bytes(4800))
                await asyncio.wait_for(runtime.entered.wait(), 1)
                await ws.send_json({"type": "played", "id": "approved1"})
                await asyncio.wait_for(runtime.ack.wait(), 1)
                self.assertFalse(runtime.release.is_set())
                await ws.send_json({"type": "stop"})
                await asyncio.wait_for(runtime.closed.wait(), 1)
                await ws.close()
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
