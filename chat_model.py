"""Replaceable synchronous Responses API adapter for ChatCore.

APIResponder.respond(history, settings) -> str. History contains only user
messages and assistant text confirmed as heard. No tools, server-side storage,
API call on construction. One bounded retry is allowed for transient transport
failures, with no partial reply delivered. close() is idempotent.
respond_cancellable(history, settings, cancel_event) is an optional synchronous
extension using a reusable asynchronous connection pool on one owned thread.
Official API: https://developers.openai.com/api/reference/python/resources/responses/methods/create
"""
from copy import deepcopy
import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import threading
import time
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import AsyncOpenAI, OpenAI, APIConnectionError, APIStatusError

MAX_REPLY_CHARS = 2000


class ChatModelError(RuntimeError):
    """Safe public failure; provider bodies and exception details are omitted."""


class ChatGenerationCancelled(ChatModelError):
    """The caller no longer needs this reply; this is not a model failure."""


class APIResponder:
    def __init__(self, *, client=None, client_factory=None, async_client_factory=None):
        self._client = client
        self._factory = client_factory or OpenAI
        self._lock = threading.RLock()
        self._closed = False
        self.calls = []
        self._async_factory = async_client_factory or AsyncOpenAI
        self._async_enabled = async_client_factory is not None or (client is None and client_factory is None)
        self._async_client = None
        self._async_loop = None
        self._async_thread = None
        self._async_tasks = set()
        self._cancel_events = set()
        self._local = threading.local()

    def get_last_call(self):
        """Safe metadata for the call made by this worker, never another session."""
        return deepcopy(getattr(self._local, "record", None))

    def _ensure_async_loop(self):
        # Called with _lock held. The loop owns its client, connection pool and
        # request tasks; workers never close a client shared by other sessions.
        if self._closed:
            raise ChatModelError("会話モデルの接続は終了しています。")
        if self._async_loop is not None:
            return self._async_loop
        ready = threading.Event()
        start_error = []

        def run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._async_loop = loop
            except Exception:
                start_error.append(True)
                ready.set()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._async_thread = threading.Thread(target=run, name="chat-model", daemon=True)
        self._async_thread.start()
        ready.wait()
        if start_error:
            raise ChatModelError("会話モデルへの接続処理を開始できません。")
        return self._async_loop

    async def _request_cancellable(self, parameters, cancel_event):
        owner = asyncio.current_task()
        self._async_tasks.add(owner)
        request = None
        try:
            if cancel_event.is_set():
                raise ChatGenerationCancelled("返答の生成を中断しました。")
            if self._async_client is None:
                self._async_client = self._async_factory(max_retries=0, timeout=parameters["timeout"])
            request = asyncio.create_task(self._async_client.responses.create(**parameters))
            # SDK timeouts are per network operation, not a wall-clock deadline.
            async with asyncio.timeout(parameters["timeout"]):
                while True:
                    done, _ = await asyncio.wait((request,), timeout=.05)
                    if cancel_event.is_set():
                        raise ChatGenerationCancelled("返答の生成を中断しました。")
                    if done:
                        return request.result()
        finally:
            if request is not None and not request.done():
                request.cancel()
            if request is not None:
                await asyncio.gather(request, return_exceptions=True)
            self._async_tasks.discard(owner)

    def _cancellable_response(self, parameters, cancel_event):
        if cancel_event.is_set():
            raise ChatGenerationCancelled("返答の生成を中断しました。")
        with self._lock:
            loop = self._ensure_async_loop()
            self._cancel_events.add(cancel_event)
            future = asyncio.run_coroutine_threadsafe(self._request_cancellable(parameters, cancel_event), loop)
        try:
            return future.result()
        except FutureCancelledError:
            raise ChatGenerationCancelled("返答の生成を中断しました。") from None
        finally:
            with self._lock:
                self._cancel_events.discard(cancel_event)

    def _get_client(self, settings):
        with self._lock:
            if self._closed:
                raise ChatModelError("会話モデルの接続は終了しています。")
            if self._client is None:
                try:
                    self._client = self._factory(max_retries=0, timeout=settings.timeout_seconds)
                except Exception:
                    raise ChatModelError("会話モデルへ接続できません。APIキーと設定を確認してください。") from None
            return self._client

    def respond(self, history, settings):
        if self._async_enabled:
            return self.respond_cancellable(history, settings, threading.Event())
        return self._respond(history, settings)

    def respond_cancellable(self, history, settings, cancel_event):
        """Optional synchronous contract; cancellation only affects this call.

        Cancelling the HTTP request stops local waiting and discards its output.
        It does not prove the provider stopped computation or guarantee a refund.
        """
        # An injected synchronous client may have its own credentials, base URL
        # or test transport. Never silently replace it with the default service.
        # Such clients retain their contract; cancellation discards late output.
        return self._respond(history, settings, cancel_event)

    def _respond(self, history, settings, cancel_event=None):
        self._local.record = None
        if (not isinstance(history, list) or not history or len(history) > 100
                or any(not isinstance(item, dict) or set(item) != {"role", "content"}
                       or item["role"] not in ("user", "assistant")
                       or not isinstance(item["content"], str) or not item["content"].strip()
                       or len(item["content"]) > 2000 for item in history)):
            raise ValueError("会話履歴の形式が不正です。")
        record = {"model": settings.model, "reasoning_effort": settings.reasoning_effort,
                  "service_tier": settings.service_tier, "input_messages": len(history),
                  "input_chars": sum(len(item["content"]) for item in history),
                  "status": "requesting", "request_ms": None, "attempts": []}
        self._local.record = record
        with self._lock:
            self.calls.append(record)
        started = time.monotonic()
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise ChatGenerationCancelled("返答の生成を中断しました。")
            client = self._get_client(settings) if not self._async_enabled else None
            parameters = dict(model=settings.model,
                instructions=settings.system_prompt, input=deepcopy(history),
                reasoning={"effort": settings.reasoning_effort}, service_tier=settings.service_tier,
                max_output_tokens=settings.max_output_tokens, store=False,
                timeout=settings.timeout_seconds)
            for number in range(settings.max_retries + 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise ChatGenerationCancelled("返答の生成を中断しました。")
                remaining = settings.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError()
                # Fail a stalled first attempt early; the last attempt gets the
                # remaining total budget so two healthy but slower responses
                # are not both discarded at the same short threshold.
                parameters["timeout"] = (min(settings.attempt_timeout_seconds, remaining)
                                         if number == 0 else remaining)
                attempt = {"number": number + 1, "status": "requesting"}
                record["attempts"].append(attempt)
                attempt_started = time.monotonic()
                try:
                    response = (client.responses.create(**parameters) if client is not None else
                                self._cancellable_response(parameters, cancel_event))
                    self._response_metadata(response, attempt)
                    attempt["status"] = "received"
                    break
                except ChatGenerationCancelled:
                    attempt["status"] = "cancelled"
                    raise
                except Exception as exc:
                    attempt["status"] = "failed"
                    self._error_metadata(exc, attempt)
                    delay = self._retry_delay(exc)
                    remaining = settings.timeout_seconds - (time.monotonic() - started)
                    if (number >= settings.max_retries or delay is None
                            or remaining <= delay + .1):
                        raise
                    attempt["retry_delay_ms"] = round(delay * 1000)
                finally:
                    attempt["request_ms"] = round((time.monotonic() - attempt_started) * 1000, 3)
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        raise ChatGenerationCancelled("返答の生成を中断しました。")
                else:
                    time.sleep(delay)
            if cancel_event is not None and cancel_event.is_set():
                raise ChatGenerationCancelled("返答の生成を中断しました。")
            self._response_metadata(response, record)
            if getattr(response, "status", None) != "completed":
                raise ChatModelError("会話の返答を最後まで生成できませんでした。")
            text = response.output_text
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_REPLY_CHARS:
                raise ChatModelError("会話の返答を正しく受け取れませんでした。")
            with self._lock:
                if self._closed:
                    raise ChatModelError("会話モデルの接続が終了しました。")
                record["status"] = "completed"
            return text.strip()
        except ChatGenerationCancelled:
            record["status"] = "cancelled"
            raise
        except ChatModelError as exc:
            record["status"] = "failed"
            self._error_metadata(exc, record)
            raise
        except Exception as exc:
            record["status"] = "failed"
            self._error_metadata(exc, record)
            raise ChatModelError("会話モデルへの問い合わせに失敗しました。") from None
        finally:
            record["request_ms"] = round((time.monotonic()-started)*1000, 3)

    @staticmethod
    def _safe_request_id(value):
        return value if isinstance(value, str) and re.fullmatch(r"req_[A-Za-z0-9_-]{1,120}", value) else None

    @classmethod
    def _error_metadata(cls, exc, record):
        # Never serialize exception messages, HTTP bodies/headers, or URLs.
        record["error_type"] = ("timeout" if isinstance(exc, TimeoutError)
            or type(exc).__name__ == "APITimeoutError" else
            "connection" if isinstance(exc, APIConnectionError) else
            "http_status" if isinstance(exc, APIStatusError) else
            "invalid_response" if isinstance(exc, ChatModelError) else "unexpected")
        status = getattr(exc, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            record["http_status"] = status
        request_id = cls._safe_request_id(getattr(exc, "request_id", None))
        if request_id:
            record["provider_request_id"] = request_id

    @classmethod
    def _response_metadata(cls, response, record):
        status = getattr(response, "status", None)
        if status in ("completed", "incomplete", "failed", "cancelled", "queued", "in_progress"):
            record["response_status"] = status
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        if reason in ("max_output_tokens", "content_filter"):
            record["incomplete_reason"] = reason
        request_id = cls._safe_request_id(getattr(response, "_request_id", None))
        if request_id:
            record["provider_request_id"] = request_id
        usage = getattr(response, "usage", None)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if type(value) is int and value >= 0:
                record[key] = value
        reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
        if type(reasoning) is int and reasoning >= 0:
            record["reasoning_tokens"] = reasoning

    @staticmethod
    def _retry_delay(exc):
        transient = isinstance(exc, (APIConnectionError, TimeoutError))
        if isinstance(exc, APIStatusError):
            transient = exc.status_code in (408, 409, 429) or exc.status_code >= 500
            if getattr(exc, "code", None) in ("insufficient_quota", "billing_hard_limit_reached"):
                transient = False
        if not transient:
            return None
        headers = getattr(getattr(exc, "response", None), "headers", {})
        if headers.get("x-should-retry") == "false":
            return None
        delay = .25
        milliseconds = headers.get("retry-after-ms")
        if milliseconds:
            try:
                delay = float(milliseconds) / 1000
                return max(.25, delay) if 0 <= delay <= 2 else None
            except (TypeError, ValueError):
                pass
        raw = headers.get("retry-after")
        if raw:
            try:
                delay = float(raw)
            except (TypeError, ValueError):
                try:
                    delay = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    delay = .25
        # Honor a long server delay by declining this interactive retry.
        if not 0 <= delay <= 2:
            return None
        return max(.25, delay)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            for event in self._cancel_events:
                event.set()
            loop, thread = self._async_loop, self._async_thread
            shutdown = (asyncio.run_coroutine_threadsafe(self._close_async(), loop)
                        if loop is not None else None)
        if shutdown is not None:
            try:
                shutdown.result(timeout=5)
            finally:
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=5)
        if client is not None:
            client.close()

    async def _close_async(self):
        tasks = tuple(self._async_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._async_client is not None:
            await self._async_client.close()
