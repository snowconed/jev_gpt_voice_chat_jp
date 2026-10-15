"""Offline API contract checks for per-session filler selection and cached PCM."""
import asyncio
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from filler_cache import write_filler_clips
import server
import test_jev_server as fixtures


class Runtime(fixtures.Runtime):
    def __init__(self, *args, filler_clips=None, speech=None):
        super().__init__(*args)
        self.filler_clips = filler_clips
        self.speech = speech
        self.skipped = []
        self.got_skipped = asyncio.Event()

    async def filler_skipped(self, identifier):
        self.skipped.append(identifier)
        self.got_skipped.set()


class FillerServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        Runtime.created = []
        for item in (patch("server.JevSpeechRuntime", Runtime),
                     patch("jev_client.has_key", return_value=True),
                     patch.dict(os.environ, {"OPENAI_API_KEY":"offline-not-a-key"})):
            item.start()
            self.addCleanup(item.stop)

    def assets(self, *, model="gpt-4o-mini-tts", voice="marin"):
        return write_filler_clips(self.root, "gpt", model=model, voice=voice,
                                 clips=[("ええと", b"\x10\x27" * 12000)])

    async def start(self, **kwargs):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        app = server.create_app(self.root / "sessions", port=port, core=fixtures.Core(),
                                filler_cache_dir=self.root, **kwargs)
        self.client = TestClient(TestServer(app, host="127.0.0.1", port=port))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.config = await (await self.client.get("/api/config")).json()
        self.headers = {"X-CSRF-Token":self.config["csrf_token"]}

    async def create(self, body):
        return await self.client.post("/api/sessions", json=body, headers=self.headers)

    async def connect(self, state):
        ws = await self.client.ws_connect(f"/api/voice/{state['id']}?token={self.config['csrf_token']}")
        self.addAsyncCleanup(ws.close)
        for _ in range(100):
            if Runtime.created:
                break
            await asyncio.sleep(0)
        runtime = Runtime.created[-1]
        await asyncio.wait_for(runtime.started.wait(), 1)
        return ws, runtime

    async def test_gpt_catalog_pcm_and_enabled_session(self):
        clips = self.assets()
        await self.start()
        self.assertTrue(self.config["filler_supported"])
        self.assertEqual(len(self.config["filler_clips"]), 1)
        advertised = self.config["filler_clips"][0]
        response = await self.client.get(advertised["url"])
        self.assertEqual(await response.read(), clips[0].pcm)
        self.assertEqual(advertised["audio_bytes"], len(clips[0].pcm))
        self.assertEqual(response.headers["Content-Type"], "application/octet-stream")
        state = await (await self.create({"filler_enabled":True})).json()
        self.assertTrue(state["filler_enabled"])
        _, runtime = await self.connect(state)
        self.assertEqual(runtime.filler_clips, [clips[0].metadata()])

    async def test_other_voice_cache_is_not_served(self):
        clips = self.assets(voice="alloy")
        await self.start()
        self.assertEqual(self.config["filler_clips"], [])
        self.assertEqual((await self.client.get(f"/api/fillers/{clips[0].id}.pcm")).status, 404)
        self.assertFalse((await (await self.create({"filler_enabled":True})).json())["filler_enabled"])

    async def test_other_model_cache_is_not_served(self):
        clips = self.assets(model="offline-other-model")
        await self.start()
        self.assertEqual(self.config["filler_clips"], [])
        self.assertEqual((await self.client.get(f"/api/fillers/{clips[0].id}.pcm")).status, 404)
        self.assertFalse((await (await self.create({"filler_enabled":True})).json())["filler_enabled"])

    async def test_missing_cache_downgrades_without_blocking_conversation(self):
        await self.start()
        state = await (await self.create({"filler_enabled":True})).json()
        self.assertFalse(state["filler_enabled"])
        _, runtime = await self.connect(state)
        self.assertIsNone(runtime.filler_clips)

    async def test_off_session_does_not_inject_clips_or_allow_hot_toggle(self):
        self.assets()
        await self.start()
        state = await (await self.create({"filler_enabled":False})).json()
        self.assertFalse(state["filler_enabled"])
        ws, runtime = await self.connect(state)
        await ws.send_json({"type":"filler_enabled", "enabled":True})
        await ws.send_json({"type":"text", "text":"確認です", "client_input_id":"input-1"})
        self.assertEqual(await ws.receive_json(timeout=1),
                         {"type":"input_accepted", "client_input_id":"input-1"})
        self.assertIsNone(runtime.filler_clips)
        self.assertEqual(runtime.texts, ["確認です"])

    async def test_old_clients_default_to_off(self):
        self.assets()
        await self.start()
        state = await (await self.create({})).json()
        self.assertFalse(state["filler_enabled"])

    async def test_selection_requires_boolean_and_existing_csrf(self):
        self.assets()
        await self.start()
        for value in ("true", "false", 0, 1, None, [], {}):
            self.assertEqual((await self.create({"filler_enabled":value})).status, 400)
        response = await self.client.post("/api/sessions", json={"filler_enabled":True})
        self.assertEqual(response.status, 403)

    async def test_skip_and_physical_ack_have_separate_dispatch(self):
        self.assets()
        await self.start()
        state = await (await self.create({"filler_enabled":True})).json()
        ws, runtime = await self.connect(state)
        await ws.send_json({"type":"filler_skipped", "id":"filler-one"})
        await ws.send_json({"type":"played", "id":"main-one"})
        await asyncio.wait_for(runtime.got_skipped.wait(), 1)
        await asyncio.wait_for(runtime.got_played.wait(), 1)
        self.assertEqual(runtime.skipped, ["filler-one"])
        self.assertEqual(runtime.acks, ["main-one"])

    async def test_live_mode_is_rejected_before_startup(self):
        with self.assertRaisesRegex(ValueError, "Only Jev"):
            server.create_app(core=fixtures.Core(), turn_mode="live")

    async def test_non_gpt_backends_are_not_available(self):
        for backend in ("voicevox", "piper", "unknown"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "Only GPT speech"):
                    server.create_app(core=fixtures.Core(), tts=backend)



if __name__ == "__main__":
    unittest.main()
