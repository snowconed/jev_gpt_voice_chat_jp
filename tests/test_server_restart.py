"""No-API shutdown/restart checks on real local HTTP/WebSocket connections."""
import asyncio
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, ClientTimeout, WSServerHandshakeError, WSMsgType, web

import server
from test_jev_server import Core as BaseCore, Runtime as BaseRuntime


class Core(BaseCore):
    def __init__(self):
        self.events = []
        self.view_gate = None
        self.view_entered = threading.Event()
        self.close_count = 0

    def view(self, sid):
        self.view_entered.set()
        if self.view_gate is not None:
            self.view_gate.wait(3)
        return super().view(sid)

    def close(self):
        self.close_count += 1
        self.events.append("core_closed")


class Runtime(BaseRuntime):
    created = []
    close_gate = None
    start_gate = None

    async def start(self):
        self.started.set()
        if self.start_gate is not None:
            try:
                await self.start_gate.wait()
            except asyncio.CancelledError:
                self.core.events.append("start_cancelled")
                raise
        self.ready = True

    async def close(self):
        self.core.events.append("runtime_close_started")
        if self.close_gate is not None:
            await self.close_gate.wait()
        await super().close()
        self.core.events.append("runtime_closed")


class RestartTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        Runtime.created, Runtime.close_gate, Runtime.start_gate = [], None, None
        for item in (patch("server.JevSpeechRuntime", Runtime),
                     patch("jev_client.has_key", return_value=True)):
            item.start()
            self.addCleanup(item.stop)
        self.client = ClientSession(timeout=ClientTimeout(total=3))
        self.addAsyncCleanup(self.client.close)

    async def serve(self, mode="jev", port=None, core=None):
        if port is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        core = core or Core()
        app = server.create_app(Path(__file__).resolve().parents[1] / ".runtime" / "http_test",
                                port=port, core=core, turn_mode=mode)
        # Baseline without on_shutdown only finishes after this handler timeout.
        runner = web.AppRunner(app, shutdown_timeout=.5)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        self.addAsyncCleanup(runner.cleanup)
        base = f"http://127.0.0.1:{port}"
        async with self.client.get(base + "/api/config") as response:
            config = await response.json()
        return app, runner, core, base, config["csrf_token"], port

    async def connect(self, base, token):
        ws = await self.client.ws_connect(base + f"/api/voice/{'b' * 32}?token={token}")
        for _ in range(50):
            if Runtime.created:
                break
            await asyncio.sleep(.01)
        runtime = Runtime.created[-1]
        await asyncio.wait_for(runtime.started.wait(), 1)
        return ws, runtime

    async def consume(self, ws, events, closed_notice=None):
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                event = message.json()
                events.append(event)
                if event.get("type") == "closed" and closed_notice is not None:
                    closed_notice.set()

    async def test_active_browser_closes_promptly_and_same_port_restarts_jev(self):
        for mode in ("jev",):
            port = None
            for repeat in range(2):
                with self.subTest(mode=mode, repeat=repeat):
                    Runtime.created = []
                    _, runner, core, base, token, port = await self.serve(mode, port)
                    ws, runtime = await self.connect(base, token)
                    events = []
                    receiver = asyncio.create_task(self.consume(ws, events))
                    pcm = b"\x10\x27" * 2400
                    await ws.send_bytes(pcm)
                    await asyncio.wait_for(runtime.got_microphone.wait(), 1)
                    self.assertEqual(b"".join(runtime.audio), pcm)
                    started = time.perf_counter()
                    await asyncio.wait_for(runner.cleanup(), 2)
                    self.assertLess(time.perf_counter() - started, .4)
                    await asyncio.wait_for(receiver, 1)
                    self.assertTrue(any(row.get("type") == "closed" for row in events))
                    self.assertEqual(runtime.close_count, 1)
                    self.assertEqual(core.close_count, 1)
                    self.assertLess(core.events.index("runtime_closed"), core.events.index("core_closed"))

    async def test_browser_stops_before_remote_finalization_and_core_closes_last(self):
        Runtime.close_gate = asyncio.Event()
        _, runner, core, base, token, _ = await self.serve()
        ws, runtime = await self.connect(base, token)
        received_closed = asyncio.Event()
        receiver = asyncio.create_task(self.consume(ws, [], received_closed))
        stopping = asyncio.create_task(runner.cleanup())
        try:
            await asyncio.wait_for(received_closed.wait(), .4)
            self.assertEqual(core.close_count, 0)
            self.assertFalse(runtime.closed.is_set())
        finally:
            Runtime.close_gate.set()
            await asyncio.wait_for(stopping, 2)
            await asyncio.wait_for(receiver, 1)
        self.assertEqual(runtime.close_count, 1)
        self.assertEqual(core.events[-1], "core_closed")

    async def test_shutdown_during_reserved_core_view_never_constructs_runtime(self):
        core = Core()
        core.view_gate = threading.Event()
        _, runner, core, base, token, _ = await self.serve(core=core)
        connecting = asyncio.create_task(self.client.ws_connect(base + f"/api/voice/{'b' * 32}?token={token}"))
        try:
            for _ in range(100):
                if core.view_entered.is_set():
                    break
                await asyncio.sleep(.005)
            self.assertTrue(core.view_entered.is_set())
            await asyncio.wait_for(runner.cleanup(), 1)
            with self.assertRaises(WSServerHandshakeError) as raised:
                await connecting
            self.assertEqual(raised.exception.status, 503)
            self.assertEqual(Runtime.created, [])
            self.assertEqual(core.close_count, 1)
        finally:
            core.view_gate.set()
            await asyncio.gather(connecting, return_exceptions=True)

    async def test_shutdown_during_websocket_prepare_does_not_start_remote_session(self):
        original = web.WebSocketResponse.prepare
        for headers_sent in (False, True):
            with self.subTest(headers_sent=headers_sent):
                Runtime.created = []
                entered, gate = asyncio.Event(), asyncio.Event()

                async def gated_prepare(ws, request):
                    if headers_sent:
                        result = await original(ws, request)
                    entered.set()
                    await gate.wait()
                    return result if headers_sent else await original(ws, request)

                _, runner, core, base, token, _ = await self.serve()
                with patch.object(web.WebSocketResponse, "prepare", gated_prepare):
                    connecting = asyncio.create_task(self.client.ws_connect(base + f"/api/voice/{'b' * 32}?token={token}"))
                    await asyncio.wait_for(entered.wait(), 1)
                    receiver = None
                    if headers_sent:
                        ws = await connecting
                        receiver = asyncio.create_task(self.consume(ws, []))
                    await asyncio.wait_for(runner.cleanup(), 2)
                    if headers_sent:
                        await asyncio.wait_for(receiver, 1)
                    else:
                        with self.assertRaises(WSServerHandshakeError) as raised:
                            await connecting
                        self.assertEqual(raised.exception.status, 503)
                    self.assertEqual(Runtime.created, [])
                    self.assertEqual(core.close_count, 1)

    async def test_new_requests_are_rejected_after_shutdown_begins(self):
        app, runner, core, base, token, _ = await self.serve()
        await app.shutdown()
        async with self.client.get(base + "/api/config") as response:
            self.assertEqual(response.status, 503)
        with self.assertRaises(WSServerHandshakeError) as raised:
            await self.client.ws_connect(base + f"/api/voice/{'b' * 32}?token={token}")
        self.assertEqual(raised.exception.status, 503)
        self.assertEqual(Runtime.created, [])
        await runner.cleanup()
        self.assertEqual(core.close_count, 1)

    async def test_shutdown_cancels_remote_handshake_before_finalization(self):
        Runtime.start_gate = asyncio.Event()
        _, runner, core, base, token, _ = await self.serve()
        ws, runtime = await self.connect(base, token)
        receiver = asyncio.create_task(self.consume(ws, []))
        await asyncio.wait_for(runner.cleanup(), 2)
        await asyncio.wait_for(receiver, 1)
        self.assertIn("start_cancelled", core.events)
        self.assertEqual(runtime.close_count, 1)
        self.assertEqual(core.events[-1], "core_closed")


if __name__ == "__main__":
    unittest.main()
