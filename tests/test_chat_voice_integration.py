"""Offline HTTP/WebSocket -> real runtime -> chat -> buffered PCM integration.

Only the remote speech, transcription and chat providers are fake. Jev server
routing, browser delivery acknowledgements and same-port restarts run locally.
"""
import asyncio
from copy import deepcopy
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

import server
from jev_runtime import JevSpeechRuntime
from test_jev_runtime import Judge
from test_speech_runtime import Live, Speech


class Chat:
    def __init__(self):
        self.sessions = {}
        self.inputs = []
        self.deliveries = []
        self.close_count = 0
        self.lock = threading.RLock()

    def create(self, role_id="chat"):
        if role_id != "chat":
            raise ValueError("Unknown role")
        sid = uuid.uuid4().hex
        self.sessions[sid] = {"id": sid, "role_id": "chat", "messages": [
            {"id": "greeting", "role": "assistant", "kind": "reply", "content": "こんにちは。今日はどうでしたか？",
             "delivery": {"status": "pending"}}]}
        return self.view(sid)

    def view(self, sid):
        with self.lock:
            return deepcopy(self.sessions[sid])

    def pending(self, sid):
        return [{"id": message["id"], "kind": message["kind"], "text": message["content"]}
                for message in self.view(sid)["messages"]
                if message["role"] == "assistant" and message["delivery"]["status"] == "pending"]

    def submit(self, sid, text, request_id):
        with self.lock:
            self.inputs.append(text)
            self.sessions[sid]["messages"].extend([
                {"id": request_id, "role": "user", "kind": "user", "content": text},
                {"id": f"reply-{len(self.inputs)}", "role": "assistant", "kind": "reply",
                 "content": f"お話は「{text}」ですね。", "delivery": {"status": "pending"}}])
            return self.view(sid)

    def delivery(self, sid, mid, status, heard_text=""):
        with self.lock:
            self.deliveries.append((mid, status, heard_text))
            for message in self.sessions[sid]["messages"]:
                if message["id"] == mid:
                    message["delivery"]["status"] = status

    def close(self):
        self.close_count += 1


class JevRuntime(JevSpeechRuntime):
    created = []

    def __init__(self, core, sid, send_json, send_bytes, directory):
        client, speech = Live(), Speech()
        client.pending_audio = False
        super().__init__(core, sid, send_json, send_bytes, directory,
                         client=client, speech=speech, jev=Judge())
        type(self).created.append(self)


class ChatVoiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        parent = Path(__file__).resolve().parents[1] / ".runtime" / "chat_voice_tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.message_segments = {}
        for item in (patch("server.JevSpeechRuntime", JevRuntime),
                     patch("jev_client.has_key", return_value=True),
                     patch.dict(os.environ, {"OPENAI_API_KEY": "offline-not-an-api-key"})):
            item.start()
            self.addCleanup(item.stop)

    async def wait_for(self, predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(.005)

    async def connect(self, mode, port):
        cls = JevRuntime
        cls.created = []
        core = Chat()
        app = server.create_app(Path(self.temp.name), port=port, core=core, turn_mode=mode)
        client = TestClient(TestServer(app, host="127.0.0.1", port=port))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        config = await (await client.get("/api/config")).json()
        response = await client.post("/api/sessions", json={"role_id": "chat"},
                                     headers={"X-CSRF-Token": config["csrf_token"]})
        self.assertEqual(response.status, 200)
        sid = (await response.json())["id"]
        ws = await client.ws_connect(f"/api/voice/{sid}?token={config['csrf_token']}")
        self.addAsyncCleanup(ws.close)
        await self.wait_for(lambda: cls.created and cls.created[0].ready)
        return client, ws, cls.created[0], core, config

    async def playback(self, ws, expected_id):
        audio, text = bytearray(), ""
        segments, active = [], None
        queued_end = 0.0
        async with asyncio.timeout(4):
            while True:
                message = await ws.receive()
                if message.type == WSMsgType.BINARY:
                    self.assertIsNotNone(active)
                    audio.extend(message.data)
                    active["received_bytes"] += len(message.data)
                    continue
                self.assertEqual(message.type, WSMsgType.TEXT)
                event = message.json()
                self.assertNotIn(event["type"], ("error", "closed"), event)
                if event["type"] == "playback":
                    self.assertEqual(event["message_id"], expected_id)
                    self.assertEqual(event["segment_index"], len(segments))
                    self.assertTrue(event["buffered"])
                    self.assertIsNone(active)
                    active = {**event, "received_bytes": 0}
                    segments.append(active)
                elif event["type"] == "output_transcript":
                    self.assertEqual(event["message_id"], expected_id)
                    text += event["text"]
                elif event["type"] == "playback_end":
                    self.assertIsNotNone(active)
                    self.assertEqual(event["id"], active["id"])
                    self.assertEqual(active["received_bytes"], active["audio_bytes"])
                    queued_end = max(queued_end, time.monotonic()) + active["audio_seconds"]
                    active["played_at"] = queued_end
                    final = active["final_segment"]
                    active = None
                    if final:
                        self.assertLessEqual(len(segments), 2)
                        self.message_segments[expected_id] = segments
                        return bytes(audio), text

    async def acknowledge(self, ws, core, mid):
        for segment in self.message_segments[mid]:
            self.assertNotIn((mid, "completed", ""), core.deliveries)
            await asyncio.sleep(max(0, segment["played_at"] + .02 - time.monotonic()))
            await ws.send_json({"type": "played", "id": segment["id"]})
        await self.wait_for(lambda: (mid, "completed", "") in core.deliveries)

    async def test_jev_streams_exact_chat_reply_and_restart_on_same_port(self):
        for mode in ("jev",):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            previous_token = None
            for repeat in range(2):
                with self.subTest(mode=mode, repeat=repeat):
                    client, ws, runtime, core, config = await self.connect(mode, port)
                    try:
                        self.assertNotEqual(config["csrf_token"], previous_token)
                        previous_token = config["csrf_token"]
                        pcm, text = await self.playback(ws, "greeting")
                        self.assertEqual(pcm, b"".join(runtime.speech.chunks))
                        self.assertEqual(text, "こんにちは。今日はどうでしたか？")
                        self.assertEqual(core.deliveries, [])
                        await self.acknowledge(ws, core, "greeting")
                        microphone = bytes(4800)
                        await ws.send_bytes(microphone)
                        await self.wait_for(lambda: len(b"".join(runtime.client.pcm)) >= len(microphone))
                        self.assertEqual(b"".join(runtime.client.pcm), microphone)
                        user_text = "今日は散歩に行きました。"
                        if mode == "jev":
                            await runtime.client.queue.put({"type": "session.input_transcript.completed", "item_id": "user-1",
                                                            "sequence": 0, "segment_text": user_text, "text": user_text})
                        else:
                            await runtime.client.queue.put({"type": "session.input_transcript.delta", "delta": user_text,
                                                            "start_ms": 0, "end_ms": 1000})
                            await runtime.client.queue.put({"type": "session.delegation.created", "offset_ms": 1000})
                        pcm, text = await self.playback(ws, "reply-1")
                        self.assertEqual(core.inputs, [user_text])
                        self.assertEqual(text, f"お話は「{user_text}」ですね。")
                        self.assertEqual(pcm, b"".join(runtime.speech.chunks) * len(self.message_segments["reply-1"]))
                        self.assertEqual("".join(runtime.speech.calls[1:]), text)
                        self.assertNotIn(("reply-1", "completed", ""), core.deliveries)
                        await self.acknowledge(ws, core, "reply-1")
                        await ws.send_json({"type": "stop"})
                        await self.wait_for(lambda: runtime.closed)
                    finally:
                        await client.close()
                    self.assertEqual(core.close_count, 1)
                    self.assertEqual(runtime.speech.closes, 1)
                    self.assertEqual(runtime.client.disconnects, 1)

    async def test_typed_correction_cancels_prepared_pcm_in_jev(self):
        for mode in ("jev",):
            with self.subTest(mode=mode):
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                client, ws, runtime, core, _ = await self.connect(mode, port)
                try:
                    await self.playback(ws, "greeting")
                    await self.acknowledge(ws, core, "greeting")
                    prior_calls = len(runtime.speech.calls)
                    runtime.speech.before_first = asyncio.Event()
                    await ws.send_json({"type": "text", "text": "最初のお話です。"})
                    await self.wait_for(lambda: len(runtime.speech.calls) >= prior_calls + 2)
                    self.assertEqual(runtime.speech.calls[prior_calls:], ["お話は「最初のお話です。」", "ですね。"])
                    original_generation = runtime.output_generation
                    await ws.send_json({"type": "text", "text": "訂正したお話です。"})
                    await self.wait_for(lambda: ("reply-1", "cancelled", "") in core.deliveries)
                    await self.wait_for(lambda: len(runtime.speech.calls) >= prior_calls + 4)
                    self.assertGreater(runtime.output_generation, original_generation)
                    runtime.speech.before_first.set()
                    # The reader rejects any stale reply-1 playback; only the
                    # corrected parent message may produce audible segments.
                    pcm, text = await self.playback(ws, "reply-2")
                    self.assertEqual(core.inputs, ["最初のお話です。", "訂正したお話です。"])
                    self.assertEqual(text, "お話は「訂正したお話です。」ですね。")
                    self.assertEqual("".join(runtime.speech.calls[prior_calls + 2:]), text)
                    self.assertEqual(pcm, b"".join(runtime.speech.chunks) * len(self.message_segments["reply-2"]))
                    await self.acknowledge(ws, core, "reply-2")
                finally:
                    await client.close()


if __name__ == "__main__":
    unittest.main()
