"""Validated, offline-only GPT speech filler clips for Jev conversations.

Loading a missing or stale cache disables fillers; it never invokes synthesis or
prevents a conversation. Only the explicit preparation command writes assets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from speech_client import INSTRUCTIONS, RATE

GENERATION_VERSION = 1
FILLER_TEXTS = ("ええと", "えーと", "えっと", "うーん", "んーと", "あの")
MAX_CLIPS = 6
MIN_AUDIO_SECONDS = 0.1
MAX_AUDIO_SECONDS = 3.0
MAX_AUDIO_BYTES = int(RATE * 2 * MAX_AUDIO_SECONDS)
MAX_MANIFEST_BYTES = 32768
TRIM_POLICY = "outer-silence-peak64-padding40-60ms-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Za-z0-9_.:+-]{1,100}\Z")


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def filler_profile(backend: str, *, model: str, voice: str) -> dict:
    """Identify the current synthesis settings, without environment secrets."""
    if (backend != "gpt" or not isinstance(model, str)
            or not _LABEL.fullmatch(model) or not isinstance(voice, str)
            or not _LABEL.fullmatch(voice)):
        raise ValueError("フィラー音声設定が不正です。")
    settings = {"instructions": INSTRUCTIONS, "response_format": "pcm"}
    # Keep the existing GPT cache fingerprint compatible with prepared clips.
    return {"backend": backend, "model": model, "voice": voice, "speaker_id": None,
            "settings": settings, "rate": RATE, "channels": 1, "sample_width": 2,
            "generation_version": GENERATION_VERSION, "trim_policy": TRIM_POLICY,
            "texts": list(FILLER_TEXTS)}


def profile_key(profile: dict) -> str:
    return hashlib.sha256(_canonical(profile)).hexdigest()


def _clip_id(profile: dict, text: str, pcm_hash: str) -> str:
    return hashlib.sha256(_canonical({"profile": profile, "text": text, "pcm_sha256": pcm_hash})).hexdigest()


@dataclass(frozen=True)
class FillerClip:
    id: str
    text: str
    pcm: bytes
    backend: str
    model: str
    voice: str

    @property
    def audio_bytes(self) -> int:
        return len(self.pcm)

    @property
    def audio_seconds(self) -> float:
        return len(self.pcm) / (RATE * 2)

    def metadata(self) -> dict:
        return {"id": self.id, "text": self.text, "audio_bytes": self.audio_bytes,
                "audio_seconds": self.audio_seconds, "rate": RATE,
                "channels": 1, "sample_width": 2}


def _validate_pcm(pcm: bytes) -> None:
    if (not isinstance(pcm, bytes) or len(pcm) % 2
            or not int(RATE * 2 * MIN_AUDIO_SECONDS) <= len(pcm) <= MAX_AUDIO_BYTES):
        raise ValueError("フィラー音声の長さまたはPCM形式が不正です。")
    # Do not accept a silent asset as a usable response.
    if not any(abs(int.from_bytes(pcm[n:n + 2], "little", signed=True)) > 64
               for n in range(0, len(pcm), 2)):
        raise ValueError("フィラー音声が無音です。")


def _bounded_read(path: Path, limit: int) -> bytes:
    with path.open("rb") as source:
        result = source.read(limit + 1)
    if len(result) > limit:
        raise ValueError("フィラーのファイルサイズが上限を超えています。")
    return result


def load_filler_clips(root, backend: str, *, model: str, voice: str) -> tuple[FillerClip, ...]:
    """Return complete matching assets, or () on any cache/configuration error.

    File names are derived from validated hashes, not trusted manifest paths.
    Resolved paths must stay inside the requested cache and profile directory.
    """
    try:
        profile = filler_profile(backend, model=model, voice=voice)
        cache_root = Path(root).resolve()
        directory = (cache_root / profile_key(profile)).resolve()
        if directory.parent != cache_root:
            return ()
        manifest_path = (directory / "manifest.json").resolve()
        if manifest_path.parent != directory:
            return ()
        manifest = json.loads(_bounded_read(manifest_path, MAX_MANIFEST_BYTES))
        if (not isinstance(manifest, dict) or manifest.get("profile") != profile
                or not isinstance(manifest.get("clips"), list)
                or not 1 <= len(manifest["clips"]) <= MAX_CLIPS):
            return ()
        result, seen_ids, seen_texts = [], set(), set()
        for item in manifest["clips"]:
            if not isinstance(item, dict):
                return ()
            identifier, text, pcm_hash = item.get("id"), item.get("text"), item.get("pcm_sha256")
            if (not isinstance(identifier, str) or not _HEX.fullmatch(identifier)
                    or not isinstance(pcm_hash, str) or not _HEX.fullmatch(pcm_hash)
                    or text not in FILLER_TEXTS or identifier in seen_ids or text in seen_texts
                    or identifier != _clip_id(profile, text, pcm_hash)):
                return ()
            pcm_path = (directory / (identifier + ".pcm")).resolve()
            if pcm_path.parent != directory:
                return ()
            pcm = _bounded_read(pcm_path, MAX_AUDIO_BYTES)
            _validate_pcm(pcm)
            seconds = item.get("audio_seconds")
            if (type(item.get("audio_bytes")) is not int or item["audio_bytes"] != len(pcm)
                    or type(seconds) not in (int, float) or not math.isfinite(seconds)
                    or abs(seconds - len(pcm) / (RATE * 2)) > 1e-6
                    or hashlib.sha256(pcm).hexdigest() != pcm_hash):
                return ()
            result.append(FillerClip(identifier, text, pcm, backend, model, voice))
            seen_ids.add(identifier)
            seen_texts.add(text)
        return tuple(result)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return ()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".filler-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_filler_clips(root, backend: str, *, model: str, voice: str, clips) -> tuple[FillerClip, ...]:
    """Explicit preparer only: atomically publish bounded, fully validated PCM."""
    profile = filler_profile(backend, model=model, voice=voice)
    inputs = list(clips)
    if not 1 <= len(inputs) <= MAX_CLIPS:
        raise ValueError("フィラー数が不正です。")
    items, seen = [], set()
    for text, pcm in inputs:
        if text not in FILLER_TEXTS or text in seen:
            raise ValueError("フィラー本文が不正です。")
        _validate_pcm(pcm)
        pcm_hash = hashlib.sha256(pcm).hexdigest()
        items.append(({"id": _clip_id(profile, text, pcm_hash), "text": text,
                       "pcm_sha256": pcm_hash, "audio_bytes": len(pcm),
                       "audio_seconds": len(pcm) / (RATE * 2)}, pcm))
        seen.add(text)
    cache_root = Path(root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    directory = (cache_root / profile_key(profile)).resolve()
    if directory.parent != cache_root:
        raise ValueError("フィラー保存先がキャッシュの外にあります。")
    directory.mkdir(exist_ok=True)
    for item, pcm in items:
        destination = (directory / (item["id"] + ".pcm")).resolve()
        if destination.parent != directory:
            raise ValueError("フィラー保存先が不正です。")
        _atomic_write(destination, pcm)
    manifest_path = (directory / "manifest.json").resolve()
    if manifest_path.parent != directory:
        raise ValueError("フィラー保存先が不正です。")
    manifest_bytes = _canonical({"profile": profile, "clips": [item for item, _ in items]}) + b"\n"
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("フィラー設定が大きすぎます。")
    _atomic_write(manifest_path, manifest_bytes)
    result = load_filler_clips(cache_root, backend, model=model, voice=voice)
    if len(result) != len(items):
        raise ValueError("保存したフィラーの検証に失敗しました。")
    return result
