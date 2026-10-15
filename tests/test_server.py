import asyncio
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from server import create_app
from test_jev_server import Runtime as RoutingRuntime


class Core:
    def create(self,role_id): return {"id":"a"*32,"messages":[],"role_id":role_id}
    def view(self,sid): return {"id":sid,"messages":[]}
    def close(self): pass


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",0))
            self.port=probe.getsockname()[1]
        key_patch = patch("jev_client.has_key", return_value=True)
        key_patch.start()
        self.addCleanup(key_patch.stop)
        app=create_app(Path(__file__).resolve().parents[1]/".runtime"/"http_test",port=self.port,core=Core(),turn_mode="jev")
        self.client=TestClient(TestServer(app,host="127.0.0.1",port=self.port))
        await self.client.start_server()
    async def asyncTearDown(self): await self.client.close()

    async def test_local_configuration_has_role_labels_and_no_key(self):
        response=await self.client.get("/api/config")
        self.assertEqual(response.status,200)
        data=await response.json()
        self.assertEqual(data["model"],"gpt-live-transcribe")
        self.assertEqual(data["voice"],"marin")
        self.assertEqual(data["interruption_policy"],"semantic")
        self.assertEqual([role["id"] for role in data["roles"]],["chat"])
        self.assertTrue(all(role["label"] for role in data["roles"]))
        self.assertNotIn(os.environ.get("OPENAI_API_KEY","missing-secret-value"),str(data))

    async def test_cross_origin_rejected(self):
        response=await self.client.get("/api/config",headers={"Origin":"https://example.com"})
        self.assertEqual(response.status,403)

    async def test_foreign_host_rejected(self):
        response=await self.client.get("/api/config",headers={"Host":"example.com"})
        self.assertEqual(response.status,403)

    async def test_create_requires_token(self):
        response=await self.client.post("/api/sessions",json={"role_id":"chat"})
        self.assertEqual(response.status,403)
        config=await (await self.client.get("/api/config")).json()
        with patch.dict(os.environ,{"OPENAI_API_KEY":"test-not-a-real-key"}):
            response=await self.client.post("/api/sessions",json={"role_id":"chat"},headers={"X-CSRF-Token":config["csrf_token"]})
        self.assertEqual(response.status,200)
        self.assertEqual((await response.json())["id"],"a"*32)

    async def test_static_paths_do_not_expose_research_data(self):
        for name in ("/","/app.js","/style.css","/pcm-worklet.js"):
            response=await self.client.get(name)
            self.assertEqual(response.status,200)
            await response.read()
        response=await self.client.get("/core_origin.json")
        self.assertEqual(response.status,404)

    async def test_old_browser_energy_activity_cannot_interrupt_jev_playback(self):
        RoutingRuntime.created = []
        config = await (await self.client.get("/api/config")).json()
        with patch("server.JevSpeechRuntime", RoutingRuntime):
            ws = await self.client.ws_connect(f"/api/voice/{'a' * 32}?token={config['csrf_token']}")
            self.addAsyncCleanup(ws.close)
            async with asyncio.timeout(2):
                while not RoutingRuntime.created:
                    await asyncio.sleep(.005)
            runtime = RoutingRuntime.created[0]
            await asyncio.wait_for(runtime.started.wait(), 1)
            for _ in range(20):
                await ws.send_json({"type": "user_activity"})
            pcm = b"\xff\x7f" * 2400
            await ws.send_bytes(pcm)
            # This ordered ACK proves the preceding activity events were read.
            await ws.send_json({"type": "played", "id": "segment-one"})
            await asyncio.wait_for(runtime.got_played.wait(), 1)
            await asyncio.wait_for(runtime.got_microphone.wait(), 1)
            self.assertEqual(runtime.interrupt_count, 0)
            self.assertIsNotNone(runtime.playing)
            self.assertEqual(b"".join(runtime.audio), pcm)
            await ws.send_json({"type": "text", "text": "訂正します。"})
            await asyncio.wait_for(runtime.got_text.wait(), 1)
            self.assertEqual(runtime.texts, ["訂正します。"])
            await ws.send_json({"type": "stop"})
            await asyncio.wait_for(runtime.closed.wait(), 1)
            self.assertEqual(runtime.close_count, 1)


class JevServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_jev_configuration_and_missing_key_are_explicit(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",0))
            port = probe.getsockname()[1]
        app = create_app(port=port, core=Core(), turn_mode="jev", service_tier="ultrafast")
        client = TestClient(TestServer(app,host="127.0.0.1",port=port))
        await client.start_server()
        try:
            with patch("jev_client.has_key", return_value=False), patch.dict(os.environ, {"OPENAI_API_KEY":"offline-key"}):
                cfg = await (await client.get("/api/config")).json()
                self.assertEqual(cfg["model"], "gpt-live-transcribe")
                self.assertEqual(cfg["turn_mode"], "jev")
                self.assertEqual(cfg["interruption_policy"], "semantic")
                self.assertEqual(cfg["service_tier"], "ultrafast")
                self.assertFalse(cfg["turn_key_configured"])
                self.assertNotIn("offline-key", str(cfg))
                response = await client.post("/api/sessions", json={"role_id":"chat"},
                    headers={"X-CSRF-Token":cfg["csrf_token"]})
                self.assertEqual(response.status, 503)
                self.assertIn("TYPESAFE_API_KEY", (await response.json())["error"])
        finally:
            await client.close()


if __name__=="__main__": unittest.main()
