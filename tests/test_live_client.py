"""No-API checks of safe, serialized Live sends and transport accounting."""
import asyncio
import base64
from types import SimpleNamespace
import unittest

import aiohttp
from live_client import LiveClient, INSTRUCTIONS


class Socket:
    closed = False

    def __init__(self):
        self.sent = []
        self.active = 0
        self.max_active = 0

    async def send_json(self, event):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(.01)
            self.sent.append(event)
        finally:
            self.active -= 1

    def __aiter__(self):
        async def events():
            for value in ({"type": "session.input_transcript.delta", "delta": "今日は"},
                          {"type": "session.closed", "usage": {"seconds": 1}}):
                yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, json=lambda v=value: v)
        return events()


class LiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_bytes_are_unchanged_and_sends_serialized(self):
        client = LiveClient()
        client.ws = Socket()
        pcm = b"\x01\x02"*2400
        await asyncio.gather(client.audio(pcm), client.audio(pcm), client.instruct("wait"))
        self.assertEqual(client.ws.max_active, 1)
        frames = [e for e in client.ws.sent if e["type"] == "session.input_audio.append"]
        self.assertEqual([base64.b64decode(e["audio"]) for e in frames], [pcm, pcm])
        self.assertEqual(client.transport_metrics["audio_bytes_sent"], len(pcm)*2)
        self.assertEqual(client.transport_metrics["audio_frames_sent"], 2)

    async def test_received_event_diagnostics_and_finalization(self):
        client = LiveClient()
        client.ws = Socket()
        events = [event async for event in client.events()]
        self.assertEqual(len(events), 2)
        self.assertEqual(client.transport_metrics["input_transcript_chars"], 3)
        self.assertTrue(client.finalized.is_set())

    async def test_closing_does_not_send_more_audio(self):
        client = LiveClient()
        client.ws = Socket()
        await client.request_close()
        await client.audio(bytes(4800))
        self.assertEqual([e["type"] for e in client.ws.sent], ["session.close"])

    async def test_pcm_shape_is_checked(self):
        client = LiveClient()
        client.ws = Socket()
        for pcm in (b"\x00", bytes(48002)):
            with self.assertRaises(ValueError):
                await client.audio(pcm)
        self.assertEqual(client.ws.sent, [])

    async def test_thinking_context_uses_data_event_and_requested_id(self):
        client = LiveClient()
        client.ws = Socket()
        content = '{"text":"  そのまま読んでください。\\n"}'
        event_id = await client.think(content, "item_known", event_id="context_test")
        self.assertEqual(event_id, "context_test")
        self.assertEqual(client.ws.sent, [{"type": "session.thinking.append", "event_id": "context_test",
                                         "delegation_id": "item_known", "content": content}])

    async def test_thinking_context_rejects_oversized_utf8_and_closed_session(self):
        client = LiveClient()
        client.ws = Socket()
        await client.think("あ" * 150)
        with self.assertRaises(ValueError):
            await client.think("あ" * 151)
        client.closing = True
        with self.assertRaises(RuntimeError):
            await client.think("context")
        self.assertEqual(len(client.ws.sent), 1)

    def test_default_prompt_describes_external_playback_not_approved_read_aloud(self):
        self.assertIn("generated audio is muted", INSTRUCTIONS)
        self.assertIn("external_playback", INSTRUCTIONS)
        self.assertNotIn("APPROVED_TEXT", INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
