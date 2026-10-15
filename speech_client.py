"""Stream approved text through the official PCM speech endpoint.

No dialogue decisions are made here. Text is sent unchanged, and request logs
contain only timing/counts and a validated request identifier, never credentials,
the source text, response bodies, or response headers.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
import re
import time

import aiohttp

MODEL = "gpt-4o-mini-tts"
VOICE = "marin"
RATE = 24000
MAX_TEXT_CHARS = 2000
MAX_AUDIO_BYTES = RATE * 2 * 120
ENDPOINT = "https://api.openai.com/v1/audio/speech"
INSTRUCTIONS = (
    "日本語で、落ち着いた親しみのある自然な会話として話してください。"
    "話す速さ、声の調子、話し方を一貫させてください。"
    "入力された本文を原文どおりに読み上げてください。"
    "言い換え、前置き、質問、感想、説明を追加せず、本文を省略しないでください。"
)
_REQUEST_ID = re.compile(r"req_[A-Za-z0-9_-]{1,192}\Z")


class SpeechError(RuntimeError):
    """A safe, user-facing speech failure; never wraps remote text or headers."""


class SpeechClient:
    def __init__(self, voice: str = VOICE, model: str = MODEL, *, session_factory=None):
        if not isinstance(voice, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", voice):
            raise ValueError("音声の種類の指定が不正です。")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", model):
            raise ValueError("音声モデルの指定が不正です。")
        self.voice, self.model = voice, model
        self.calls: list[dict] = []
        self._session_factory = session_factory or aiohttp.ClientSession
        self._session = None
        self._closed = False
        self._responses = set()

    def _get_session(self):
        if self._closed:
            raise SpeechError("音声接続は終了しています。新しい会話を開始してください。")
        if self._session is None:
            self._session = self._session_factory(timeout=aiohttp.ClientTimeout(total=120, sock_read=15))
        return self._session

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield even-length mono PCM16 little-endian chunks at 24 kHz.

        Caller should ``aclose`` the async generator when abandoning playback.
        Cancellation and generator exit close the active HTTP response promptly.
        The HTTP session remains available for subsequent utterances.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("読み上げる本文がありません。")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("読み上げる本文は2000文字以内にしてください。")
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key.strip():
            raise SpeechError("環境変数 OPENAI_API_KEY が設定されていません。")
        session = self._get_session()
        record = {
            "model": self.model, "voice": self.voice, "input_chars": len(text),
            "request_id": None, "http_status": None, "status": "requesting",
            "audio_bytes": 0, "received_audio_bytes": 0, "elapsed_seconds": None,
            "first_audio_ms": None, "chunk_count": 0, "max_chunk_gap_ms": 0.0,
            "audio_seconds": 0.0, "receive_elapsed_seconds": None, "receive_to_audio_ratio": None,
            "usage_supported": False,
            "usage_note": "PCM応答から使用量は取得できません。文字数・音声量から料金を確定しません。",
        }
        self.calls.append(record)
        started, response = time.monotonic(), None
        remainder = b""
        first_chunk_at = previous_chunk_at = None
        try:
            response = await session.post(
                ENDPOINT,
                headers={"Authorization": "Bearer " + key},
                json={"model": self.model, "voice": self.voice, "input": text,
                      "response_format": "pcm", "instructions": INSTRUCTIONS},
            )
            self._responses.add(response)
            record["http_status"] = response.status
            identifier = response.headers.get("x-request-id", "")
            if _REQUEST_ID.fullmatch(identifier):
                record["request_id"] = identifier
            if response.status != 200:
                record["status"] = "http_error"
                # Deliberately do not read the remote error body.
                raise SpeechError(f"音声の生成に失敗しました (HTTP {response.status})。")
            record["status"] = "streaming"
            async for chunk in response.content.iter_chunked(4096):
                if self._closed:
                    record["status"] = "cancelled"
                    raise SpeechError("音声接続が終了したため、読み上げを中止しました。")
                if not chunk:
                    continue
                now = time.monotonic()
                if first_chunk_at is None:
                    first_chunk_at = now
                    record["first_audio_ms"] = round((now-started)*1000, 3)
                if previous_chunk_at is not None:
                    record["max_chunk_gap_ms"] = max(record["max_chunk_gap_ms"], round((now-previous_chunk_at)*1000, 3))
                previous_chunk_at = now
                record["chunk_count"] += 1
                record["received_audio_bytes"] += len(chunk)
                if record["received_audio_bytes"] > MAX_AUDIO_BYTES:
                    record["status"] = "audio_limit"
                    raise SpeechError("音声が2分の上限を超えたため、読み上げを中止しました。")
                aligned = remainder + chunk
                even_length = len(aligned) - len(aligned) % 2
                remainder = aligned[even_length:]
                if even_length:
                    pcm = aligned[:even_length]
                    record["audio_bytes"] += len(pcm)
                    yield pcm
            if remainder:
                record["status"] = "invalid_pcm"
                raise SpeechError("音声データを正しく受信できませんでした。")
            if not record["audio_bytes"]:
                record["status"] = "empty_audio"
                raise SpeechError("読み上げ音声が返されませんでした。")
            record["status"] = "completed"
        except (asyncio.CancelledError, GeneratorExit):
            record["status"] = "cancelled"
            raise
        except TimeoutError:
            record["status"] = "timeout"
            raise SpeechError("音声生成の待ち時間を超えたため、読み上げを中止しました。") from None
        except aiohttp.ClientError:
            record["status"] = "connection_error"
            raise SpeechError("音声サービスへの接続に失敗しました。") from None
        except SpeechError:
            raise
        except Exception:
            record["status"] = "failed"
            raise SpeechError("音声の生成中にエラーが発生しました。") from None
        finally:
            if response is not None:
                response.close()
                self._responses.discard(response)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["audio_seconds"] = round(record["audio_bytes"] / (RATE*2), 4)
            if first_chunk_at is not None and previous_chunk_at is not None:
                span = previous_chunk_at - first_chunk_at
                record["receive_elapsed_seconds"] = round(span, 3)
                if record["audio_bytes"]:
                    record["receive_to_audio_ratio"] = round(span / (record["audio_bytes"] / (RATE*2)), 3)

    async def close(self):
        """Close metered responses first, then release the reusable session."""
        if self._closed:
            return
        self._closed = True
        for response in tuple(self._responses):
            response.close()
        self._responses.clear()
        if self._session is not None:
            await self._session.close()
