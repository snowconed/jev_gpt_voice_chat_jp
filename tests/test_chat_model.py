from copy import deepcopy
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace
import unittest

import httpx
from openai import AsyncOpenAI

from chat_config import ChatSettings
from chat_model import APIResponder, ChatGenerationCancelled, ChatModelError


class FakeClient:
    def __init__(self, *, text="そうなんですね。", status="completed", error=None):
        self.response = SimpleNamespace(output_text=text, status=status,
            usage=SimpleNamespace(input_tokens=12, output_tokens=5, total_tokens=17))
        self.error = error
        self.requests = []
        self.closed = 0
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed += 1


class FakeAsyncClient:
    def __init__(self, *, error=None):
        self.responses = self
        self.error = error
        self.requests = []
        self.started = {name: threading.Event() for name in ("blocked-1", "blocked-2", "fast")}
        self.cancelled = []
        self.closed = 0

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        name = kwargs["input"][-1]["content"]
        self.started[name].set()
        try:
            if name.startswith("blocked"):
                await asyncio.Event().wait()
            if self.error:
                raise self.error
            return SimpleNamespace(status="completed", output_text="新しい返答です。", usage=None)
        except asyncio.CancelledError:
            self.cancelled.append(name)
            raise

    async def close(self):
        self.closed += 1


class ChatModelTests(unittest.TestCase):
    @staticmethod
    def history(text):
        return [{"role": "user", "content": text}]

    def test_lazy_client_current_responses_contract_and_no_retries(self):
        created = []
        client = FakeClient()
        def factory(**kwargs):
            created.append(kwargs)
            return client
        model = APIResponder(client_factory=factory)
        self.assertEqual(created, [])
        settings = ChatSettings(system_prompt="音声向けの短い返答")
        history = [{"role": "user", "content": "今日の話です。"}]
        before = deepcopy(history)
        self.assertEqual(model.respond(history, settings), "そうなんですね。")
        self.assertEqual(created, [{"max_retries": 0, "timeout": 30.0}])
        request = client.requests[0]
        self.assertEqual(request, {"model": "gpt-6.1-sol", "instructions": settings.system_prompt,
            "input": history, "reasoning": {"effort": "low"}, "service_tier": "ultrafast",
            "max_output_tokens": 2000, "store": False, "timeout": 8.0})
        request["input"][0]["content"] = "mutated by fake provider"
        self.assertEqual(history, before)
        model.respond(history, settings)
        self.assertEqual(len(created), 1)

    def test_metadata_contains_counts_not_text_or_prompt(self):
        model = APIResponder(client=FakeClient())
        model.respond([{"role": "user", "content": "PRIVATE_USER_TEXT"}],
                      ChatSettings(system_prompt="PRIVATE_PROMPT"))
        record = model.calls[0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["input_tokens"], 12)
        self.assertGreaterEqual(record["request_ms"], 0)
        self.assertNotIn("PRIVATE", str(record))

    def test_provider_failure_is_sanitized_and_not_retried(self):
        client = FakeClient(error=RuntimeError("Bearer SECRET_KEY / PRIVATE_TEXT"))
        model = APIResponder(client=client)
        with self.assertRaises(ChatModelError) as error:
            model.respond([{"role": "user", "content": "こんにちは"}], ChatSettings())
        self.assertNotIn("SECRET", str(error.exception))
        self.assertNotIn("PRIVATE", str(model.calls))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(model.calls[0]["status"], "failed")

    def test_incomplete_empty_and_oversized_outputs_are_not_spoken(self):
        for text, status in (("途中", "incomplete"), ("", "completed"),
                             ("  ", "completed"), (None, "completed"),
                             ("あ" * 2001, "completed")):
            with self.subTest(status=status, length=len(text or "")):
                model = APIResponder(client=FakeClient(text=text, status=status))
                with self.assertRaises(ChatModelError):
                    model.respond([{"role": "user", "content": "話"}], ChatSettings())

    def test_invalid_history_does_not_create_client(self):
        def forbidden(**kwargs):
            self.fail("Network client must not be constructed")
        for history in ([], None, [{"role": "system", "content": "x"}],
                        [{"role": "user", "content": "x", "secret": "x"}],
                        [{"role": "user", "content": " "}],
                        [{"role": "user", "content": "x" * 2001}]):
            with self.subTest(history_type=type(history).__name__):
                with self.assertRaises(ValueError):
                    APIResponder(client_factory=forbidden).respond(history, ChatSettings())

    def test_missing_key_factory_failure_is_sanitized(self):
        def factory(**kwargs):
            raise ValueError("SECRET API KEY")
        with self.assertRaises(ChatModelError) as error:
            APIResponder(client_factory=factory).respond([{"role": "user", "content": "話"}], ChatSettings())
        self.assertNotIn("SECRET", str(error.exception))

    def test_close_is_idempotent_and_prevents_requests(self):
        client = FakeClient()
        model = APIResponder(client=client)
        model.close()
        model.close()
        self.assertEqual(client.closed, 1)
        with self.assertRaises(ChatModelError):
            model.respond([{"role": "user", "content": "話"}], ChatSettings())
        self.assertEqual(client.requests, [])

    def test_cancel_one_request_reuses_pool_and_keeps_other_request_alive(self):
        client, created = FakeAsyncClient(), []
        def factory(**kwargs):
            created.append(kwargs)
            return client
        model = APIResponder(async_client_factory=factory)
        self.addCleanup(model.close)
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(model.respond_cancellable, self.history("blocked-1"), ChatSettings(), cancel)
            self.assertTrue(client.started["blocked-1"].wait(2))
            second = pool.submit(model.respond_cancellable, self.history("fast"), ChatSettings(), threading.Event())
            self.assertEqual(second.result(2), "新しい返答です。")
            cancel.set()
            with self.assertRaises(ChatGenerationCancelled):
                first.result(2)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0], {"max_retries": 0, "timeout": 8.0})
        self.assertEqual(client.cancelled, ["blocked-1"])
        self.assertEqual(client.closed, 0)
        self.assertEqual([call["status"] for call in model.calls], ["cancelled", "completed"])
        self.assertEqual(model.respond_cancellable(self.history("fast"), ChatSettings(), threading.Event()), "新しい返答です。")
        self.assertEqual(len(created), 1)
        model.close()
        self.assertEqual(client.closed, 1)
        self.assertFalse(model._async_thread.is_alive())

    def test_cancel_before_start_does_not_construct_transport(self):
        def forbidden(**kwargs):
            self.fail("Cancelled requests must not construct a transport")
        model = APIResponder(async_client_factory=forbidden)
        self.addCleanup(model.close)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(ChatGenerationCancelled):
            model.respond_cancellable(self.history("fast"), ChatSettings(), cancel)
        self.assertIsNone(model._async_thread)
        self.assertEqual(model.calls[0]["status"], "cancelled")

    def test_close_cancels_async_requests_and_closes_pool_once(self):
        client = FakeAsyncClient()
        model = APIResponder(async_client_factory=lambda **kwargs: client)
        self.addCleanup(model.close)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(model.respond_cancellable, self.history(name), ChatSettings(), threading.Event())
                       for name in ("blocked-1", "blocked-2")]
            for name in ("blocked-1", "blocked-2"):
                self.assertTrue(client.started[name].wait(2))
            model.close()
            for future in pending:
                with self.assertRaises(ChatGenerationCancelled):
                    future.result(2)
        model.close()
        self.assertEqual(client.closed, 1)
        self.assertCountEqual(client.cancelled, ["blocked-1", "blocked-2"])
        self.assertFalse(model._async_thread.is_alive())
        with self.assertRaises(ChatModelError):
            model.respond_cancellable(self.history("fast"), ChatSettings(), threading.Event())
        self.assertEqual(len(client.requests), 2)

    def test_async_provider_failure_is_sanitized_without_retry(self):
        client = FakeAsyncClient(error=RuntimeError("Bearer SECRET / PRIVATE"))
        model = APIResponder(async_client_factory=lambda **kwargs: client)
        self.addCleanup(model.close)
        with self.assertRaises(ChatModelError) as error:
            model.respond_cancellable(self.history("fast"), ChatSettings(), threading.Event())
        self.assertNotIn("SECRET", str(error.exception))
        self.assertNotIn("PRIVATE", str(model.calls))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(model.calls[0]["status"], "failed")

    def test_injected_sync_client_is_not_replaced_by_default_async_transport(self):
        client = FakeClient()
        model = APIResponder(client=client)
        self.addCleanup(model.close)
        self.assertEqual(model.respond_cancellable(self.history("fast"), ChatSettings(), threading.Event()), "そうなんですね。")
        self.assertEqual(len(client.requests), 1)
        self.assertIsNone(model._async_thread)

    def test_real_sdk_cancels_mock_http_request_and_reuses_the_client(self):
        entered, http_cancelled = threading.Event(), threading.Event()
        clients = []

        async def handler(request):
            body = json.loads(request.content)
            if body["input"][-1]["content"] == "blocked-1":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    http_cancelled.set()
                    raise
            return httpx.Response(200, json={"id": "resp_offline", "object": "response",
                "created_at": 1, "status": "completed", "output": [{"id": "msg_offline",
                "type": "message", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": "確認しました。", "annotations": []}]}]})

        def factory(**kwargs):
            client = AsyncOpenAI(api_key="offline-test-no-network", **kwargs,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client

        model = APIResponder(async_client_factory=factory)
        self.addCleanup(model.close)
        cancel = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(model.respond_cancellable, self.history("blocked-1"), ChatSettings(), cancel)
            self.assertTrue(entered.wait(2))
            cancel.set()
            with self.assertRaises(ChatGenerationCancelled):
                request.result(2)
        self.assertTrue(http_cancelled.is_set())
        self.assertEqual(model.respond_cancellable(self.history("fast"), ChatSettings(), threading.Event()), "確認しました。")
        self.assertEqual(len(clients), 1)
        self.assertFalse(clients[0].is_closed())
        model.close()
        self.assertTrue(clients[0].is_closed())


if __name__ == "__main__":
    unittest.main()
