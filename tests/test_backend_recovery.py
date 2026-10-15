"""Bounded real-SDK transport retries and safe per-worker diagnostics (offline)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import unittest

import httpx
from openai import AsyncOpenAI, OpenAI

from chat_config import ChatSettings
from chat_model import APIResponder, ChatGenerationCancelled, ChatModelError


def completed(text="確認しました。"):
    return httpx.Response(200, headers={"x-request-id": "req_offline"}, json={
        "id": "resp_offline", "object": "response", "created_at": 1,
        "status": "completed", "output": [{"id": "msg_offline", "type": "message",
            "status": "completed", "role": "assistant", "content": [
                {"type": "output_text", "text": text, "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 12}}})


class BackendRecoveryTests(unittest.TestCase):
    history = [{"role": "user", "content": "こんにちは。"}]

    def model(self, handler):
        def factory(**kwargs):
            return AsyncOpenAI(api_key="offline-only", **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        model = APIResponder(async_client_factory=factory)
        self.addCleanup(model.close)
        return model

    def test_stalled_request_is_cancelled_then_retried_once_without_duplicate_output(self):
        calls, cancelled = [], threading.Event()
        async def handler(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return completed()
        model = self.model(handler)
        settings = ChatSettings(attempt_timeout_seconds=.1, timeout_seconds=1)
        started = time.monotonic()
        self.assertEqual(model.respond(self.history, settings), "確認しました。")
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        record = model.get_last_call()
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["attempts"][0]["error_type"], "timeout")
        self.assertEqual(record["reasoning_tokens"], 12)
        self.assertEqual(record["provider_request_id"], "req_offline")
        record["status"] = "mutated"
        self.assertEqual(model.get_last_call()["status"], "completed")

    def test_retry_limit_and_no_exception_body_leak(self):
        calls = []
        async def handler(request):
            calls.append(request)
            return httpx.Response(503, headers={"x-request-id": "req_failed"},
                json={"error": {"message": "SECRET_PRIVATE_BODY", "type": "server_error"}})
        model = self.model(handler)
        with self.assertRaises(ChatModelError) as error:
            model.respond(self.history, ChatSettings())
        self.assertEqual(len(calls), 2)
        record = model.get_last_call()
        self.assertEqual(record["http_status"], 503)
        self.assertEqual(record["provider_request_id"], "req_failed")
        self.assertNotIn("SECRET", str(record) + str(error.exception))

    def test_auth_quota_long_retry_after_and_incomplete_are_not_retried(self):
        for status, code, headers in ((401, "invalid_api_key", {}),
            (429, "insufficient_quota", {}), (503, "server_error", {"retry-after": "60"}),
            (503, "server_error", {"retry-after-ms": "60000"}),
            (503, "server_error", {"x-should-retry": "false"}),
            (200, "incomplete", {})):
            with self.subTest(status=status, code=code):
                calls = []
                async def handler(request):
                    calls.append(request)
                    if status == 200:
                        response = completed()
                        body = json.loads(response.content)
                        body["status"] = "incomplete"
                        body["incomplete_details"] = {"reason": "max_output_tokens"}
                        return httpx.Response(200, json=body)
                    return httpx.Response(status, headers=headers,
                        json={"error": {"message": "PRIVATE", "code": code}})
                model = self.model(handler)
                with self.assertRaises(ChatModelError):
                    model.respond(self.history, ChatSettings())
                self.assertEqual(len(calls), 1)
                if status == 200:
                    self.assertEqual(model.get_last_call()["incomplete_reason"], "max_output_tokens")

    def test_cancellation_during_retry_backoff_prevents_second_request(self):
        calls, failed = [], threading.Event()
        async def handler(request):
            calls.append(request)
            failed.set()
            return httpx.Response(503, headers={"retry-after": "1"}, json={"error": {"message": "offline"}})
        model = self.model(handler)
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(model.respond_cancellable, self.history, ChatSettings(), cancel)
            self.assertTrue(failed.wait(1))
            cancel.set()
            with self.assertRaises(ChatGenerationCancelled):
                future.result(1)
        self.assertEqual(len(calls), 1)

    def test_diagnostics_are_local_to_the_calling_worker(self):
        async def handler(request):
            return completed()
        model = self.model(handler)
        def request():
            model.respond(self.history, ChatSettings())
            return model.get_last_call()
        with ThreadPoolExecutor(max_workers=1) as pool:
            record = pool.submit(request).result(1)
        self.assertEqual(record["status"], "completed")
        self.assertIsNone(model.get_last_call())

    def test_injected_sync_client_cancellation_does_not_issue_retry(self):
        calls, cancel = [], threading.Event()
        def handler(request):
            calls.append(request)
            cancel.set()
            return httpx.Response(503, json={"error": {"message": "offline"}})
        client = OpenAI(api_key="offline-only", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        model = APIResponder(client=client)
        self.addCleanup(model.close)
        with self.assertRaises(ChatGenerationCancelled):
            model.respond_cancellable(self.history, ChatSettings(), cancel)
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.get_last_call()["status"], "cancelled")

    def test_total_deadline_stops_second_attempt_and_disables_sdk_retries(self):
        calls = []
        async def handler(request):
            calls.append(request)
            await asyncio.Event().wait()
        model = self.model(handler)
        started = time.monotonic()
        with self.assertRaises(ChatModelError):
            model.respond(self.history, ChatSettings(timeout_seconds=1, attempt_timeout_seconds=.8))
        self.assertLess(time.monotonic() - started, 1.3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.get_last_call()["error_type"], "timeout")

    def test_last_attempt_can_finish_beyond_first_attempt_deadline(self):
        calls = []
        async def handler(request):
            calls.append(request)
            if len(calls) == 1:
                await asyncio.Event().wait()
            await asyncio.sleep(.2)
            return completed()
        model = self.model(handler)
        self.assertEqual(model.respond(self.history,
            ChatSettings(timeout_seconds=1, attempt_timeout_seconds=.1)), "確認しました。")
        self.assertEqual(len(calls), 2)
        self.assertGreater(model.get_last_call()["attempts"][1]["request_ms"], 100)


if __name__ == "__main__":
    unittest.main()
