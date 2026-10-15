"""Offline checks for byte integrity, cancellation, bounds, and safe failures."""
import asyncio
import json
import os
import unittest
from unittest.mock import patch

import aiohttp

from speech_client import ENDPOINT, MAX_AUDIO_BYTES, RATE, SpeechClient, SpeechError


class Content:
    def __init__(self, chunks, *, error=None, wait=None):
        self.chunks, self.error, self.wait = chunks, error, wait

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk
        if self.wait is not None:
            await self.wait.wait()
        if self.error is not None:
            raise self.error


class Response:
    def __init__(self, chunks=(), *, status=200, request_id="req_safe123", error=None, wait=None):
        self.content = Content(chunks, error=error, wait=wait)
        self.status = status
        self.headers = {"x-request-id": request_id}
        self.closed = False
        self.close_count = 0

    def close(self):
        self.closed = True
        self.close_count += 1


class Session:
    def __init__(self, responses, **kwargs):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False
        self.close_count = 0
        self.options = kwargs

    async def post(self, endpoint, **kwargs):
        self.requests.append({"endpoint": endpoint, **kwargs})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self):
        self.closed = True
        self.close_count += 1


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key = patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret-do-not-serialize"})
        self.key.start()
        self.addCleanup(self.key.stop)

    def client(self, responses):
        created = []
        def factory(**kwargs):
            created.append(Session(responses, **kwargs))
            return created[-1]
        return SpeechClient(session_factory=factory), created

    async def test_exact_input_odd_chunks_and_reused_session(self):
        first, second = Response([b"\x01", b"\x02\x03", b"\x04"]), Response([b"\x05\x06"])
        client, made = self.client([first, second])
        text = "  原文を、そのまま。\n"
        chunks = [piece async for piece in client.stream(text)]
        self.assertEqual(b"".join(chunks), bytes([1, 2, 3, 4]))
        self.assertTrue(all(len(chunk) % 2 == 0 for chunk in chunks))
        self.assertEqual([piece async for piece in client.stream("次の文")], [b"\x05\x06"])
        self.assertEqual(len(made), 1)
        request = made[0].requests[0]
        self.assertEqual(request["endpoint"], ENDPOINT)
        self.assertEqual(request["json"]["input"], text)
        self.assertEqual(request["json"]["response_format"], "pcm")
        self.assertEqual(request["json"]["voice"], "marin")
        self.assertEqual(RATE, 24000)
        self.assertEqual(made[0].options["timeout"].total, 120)
        self.assertEqual(made[0].options["timeout"].sock_read, 15)
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(client.calls[0]["audio_bytes"], 4)
        self.assertEqual(client.calls[0]["request_id"], "req_safe123")
        self.assertEqual(client.calls[0]["chunk_count"], 3)
        self.assertIsNotNone(client.calls[0]["first_audio_ms"])
        self.assertIsNotNone(client.calls[0]["receive_to_audio_ratio"])
        self.assertFalse(client.calls[0]["usage_supported"])
        self.assertNotIn(text, json.dumps(client.calls, ensure_ascii=False))
        self.assertNotIn("test-secret", json.dumps(client.calls))
        await client.close()
        await client.close()
        self.assertEqual(made[0].close_count, 1)

    async def test_aclose_closes_response_while_session_can_be_reused(self):
        response = Response([b"\x00\x00", b"\x01\x00"])
        client, made = self.client([response])
        stream = client.stream("途中で止める")
        await anext(stream)
        await stream.aclose()
        self.assertTrue(response.closed)
        self.assertEqual(client.calls[0]["status"], "cancelled")
        self.assertFalse(made[0].closed)
        await client.close()

    async def test_task_cancel_closes_response(self):
        event = asyncio.Event()
        response = Response(wait=event)
        client, made = self.client([response])
        async def consume():
            return [piece async for piece in client.stream("取消")]
        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(response.closed)
        self.assertEqual(client.calls[0]["status"], "cancelled")
        await client.close()

    async def test_http_error_does_not_expose_body_or_key(self):
        response = Response(status=429, request_id="Authorization: test-secret-do-not-serialize")
        client, _ = self.client([response])
        with self.assertRaises(SpeechError) as error:
            _ = [chunk async for chunk in client.stream("失敗")]
        self.assertIn("429", str(error.exception))
        self.assertNotIn("test-secret", str(error.exception))
        self.assertIsNone(client.calls[0]["request_id"])
        self.assertEqual(client.calls[0]["status"], "http_error")
        self.assertTrue(response.closed)
        await client.close()

    async def test_remote_connection_detail_is_not_exposed(self):
        client, _ = self.client([aiohttp.ClientConnectionError("private secret payload")])
        with self.assertRaises(SpeechError) as error:
            _ = [chunk async for chunk in client.stream("失敗")]
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(client.calls[0]["status"], "connection_error")
        await client.close()

    async def test_timeout_status_and_response_closed(self):
        response = Response(error=TimeoutError("secret"))
        client, _ = self.client([response])
        with self.assertRaises(SpeechError):
            _ = [chunk async for chunk in client.stream("遅延")]
        self.assertEqual(client.calls[0]["status"], "timeout")
        self.assertTrue(response.closed)
        await client.close()

    async def test_final_odd_byte_is_rejected(self):
        response = Response([b"\x00\x01\x02"])
        client, _ = self.client([response])
        with self.assertRaises(SpeechError):
            _ = [chunk async for chunk in client.stream("不正PCM")]
        self.assertEqual(client.calls[0]["status"], "invalid_pcm")
        self.assertTrue(response.closed)
        await client.close()

    async def test_empty_audio_is_rejected(self):
        response = Response([])
        client, _ = self.client([response])
        with self.assertRaises(SpeechError):
            _ = [chunk async for chunk in client.stream("空音声")]
        self.assertEqual(client.calls[0]["status"], "empty_audio")
        await client.close()

    async def test_audio_length_limit(self):
        response = Response([bytes(MAX_AUDIO_BYTES + 2)])
        client, _ = self.client([response])
        with self.assertRaises(SpeechError):
            _ = [chunk async for chunk in client.stream("長い音声")]
        self.assertEqual(client.calls[0]["status"], "audio_limit")
        self.assertEqual(client.calls[0]["audio_bytes"], 0)
        self.assertTrue(response.closed)
        await client.close()

    async def test_invalid_input_never_opens_network(self):
        client, made = self.client([])
        for value in ("", "  ", "あ" * 2001, None):
            with self.assertRaises(ValueError):
                _ = [chunk async for chunk in client.stream(value)]
        self.assertEqual(made, [])
        self.assertEqual(client.calls, [])

    async def test_closed_client_never_reopens(self):
        client, made = self.client([])
        await client.close()
        with self.assertRaises(SpeechError):
            _ = [chunk async for chunk in client.stream("再開")]
        self.assertEqual(made, [])

    async def test_close_closes_active_response(self):
        response = Response([b"\x01\x02", b"\x03\x04"])
        client, made = self.client([response])
        stream = client.stream("途中")
        await anext(stream)
        await client.close()
        self.assertTrue(response.closed)
        self.assertTrue(made[0].closed)
        with self.assertRaises(SpeechError):
            await anext(stream)
        self.assertEqual(client.calls[0]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
