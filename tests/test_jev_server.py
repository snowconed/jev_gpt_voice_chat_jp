"""Offline WebSocket routing/control checks for the selectable Jev path."""
import asyncio
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

import server


class Core:
    def create(self, role_id):
        return {"id": "b" * 32, "messages": [], "role_id": role_id}

    def view(self, sid):
        return {"id": sid, "messages": []}

    def close(self):
        pass


class Runtime:
    created = []

    def __init__(self, core, sid, send_json, send_bytes, directory):
        self.core, self.sid, self.directory = core, sid, directory
        self.send_json, self.send_bytes = send_json, send_bytes
        self.ready = False
        self.finished = asyncio.Event()
        self.started = asyncio.Event()
        self.got_microphone = asyncio.Event()
        self.got_played = asyncio.Event()
        self.got_text = asyncio.Event()
        self.closed = asyncio.Event()
        self.failure_message = None
        self.playing = {"id": "approved-reply"}
        self.client = SimpleNamespace(transport_metrics={})
        self.audio = []
        self.acks = []
        self.texts = []
        self.interrupt_count = 0
        self.close_count = 0
        self.logs = []
        type(self).created.append(self)

    def log(self, kind, **data):
        self.logs.append({"kind": kind, **data})

    async def start(self):
        self.ready = True
        self.started.set()

    async def microphone(self, pcm):
        self.audio.append(pcm)
        self.got_microphone.set()

    async def played(self, identifier):
        self.acks.append(identifier)
        self.got_played.set()

    async def enqueue(self, text):
        self.texts.append(text)
        self.got_text.set()

    async def interrupt(self):
        self.interrupt_count += 1

    async def close(self):
        self.close_count += 1
        self.ready = False
        self.closed.set()


class JevWebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        Runtime.created = []
        self.patches = [patch("server.JevSpeechRuntime", Runtime),
                        patch("jev_client.has_key", return_value=True),
                        patch.dict(os.environ, {"OPENAI_API_KEY": "offline-not-a-key"})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.core = Core()
        app = server.create_app(Path(__file__).resolve().parents[1] / ".runtime" / "http_test",
                                port=self.port, core=self.core,
                                turn_mode="jev", service_tier="ultrafast")
        self.client = TestClient(TestServer(app, host="127.0.0.1", port=self.port))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.config = await (await self.client.get("/api/config")).json()

    async def connect(self):
        response = await self.client.post("/api/sessions", json={"role_id": "chat"},
                                         headers={"X-CSRF-Token": self.config["csrf_token"]})
        self.assertEqual(response.status, 200)
        sid = (await response.json())["id"]
        ws = await self.client.ws_connect(f"/api/voice/{sid}?token={self.config['csrf_token']}")
        self.addAsyncCleanup(ws.close)
        for _ in range(10):
            if Runtime.created:
                break
            await asyncio.sleep(0)
        self.assertEqual(len(Runtime.created), 1)
        runtime = Runtime.created[0]
        await asyncio.wait_for(runtime.started.wait(), 1)
        return ws, runtime

    async def test_jev_selection_preserves_audio_and_browser_ack(self):
        self.assertEqual(self.config["interruption_policy"], "semantic")
        self.assertEqual(self.config["turn_mode"], "jev")
        self.assertEqual(self.config["service_tier"], "ultrafast")
        with patch("speech_runtime.SpeechRuntime", side_effect=AssertionError("Wrong runtime selected")):
            ws, runtime = await self.connect()
            pcm = b"\x10\x27" * 2400
            await ws.send_bytes(pcm)
            await ws.send_json({"type": "user_activity"})
            await ws.send_json({"type": "played", "id": "approved-reply"})
            await asyncio.wait_for(runtime.got_microphone.wait(), 1)
            await asyncio.wait_for(runtime.got_played.wait(), 1)
            self.assertEqual(b"".join(runtime.audio), pcm)
            self.assertEqual(runtime.acks, ["approved-reply"])
            self.assertEqual(runtime.interrupt_count, 0)
            self.assertIs(runtime.core, self.core)
            self.assertIn({"kind": "voice_configuration", "turn_mode": "jev", "service_tier": "ultrafast", "tts_backend": "gpt"}, runtime.logs)
            await ws.send_json({"type": "stop"})
            await asyncio.wait_for(runtime.closed.wait(), 1)
            event = await ws.receive_json(timeout=1)
            self.assertEqual(event, {"type": "closed"})
            self.assertEqual(runtime.close_count, 1)

    async def test_typed_input_and_disconnect_are_forwarded_once(self):
        ws, runtime = await self.connect()
        await ws.send_json({"type": "text", "text": "入力をそのまま送ります。"})
        await asyncio.wait_for(runtime.got_text.wait(), 1)
        self.assertEqual(runtime.texts, ["入力をそのまま送ります。"])
        await ws.close()
        await asyncio.wait_for(runtime.closed.wait(), 1)
        self.assertEqual(runtime.close_count, 1)




if __name__ == "__main__":
    unittest.main()
