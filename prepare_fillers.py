"""Explicit offline preparation; conversation startup never runs this script."""
from __future__ import annotations

import argparse
import asyncio
from array import array
import json
from pathlib import Path
import sys
import time

from filler_cache import (FILLER_TEXTS, MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS,
                          TRIM_POLICY, load_filler_clips, write_filler_clips)
from speech_client import MODEL, RATE, VOICE, SpeechClient

ROOT = Path(__file__).resolve().parent


def trim_outer_silence(pcm: bytes) -> tuple[bytes, dict]:
    """Remove only low-amplitude outer silence, retaining 40/60 ms padding.

    No resampling, speed/pitch changes, internal edits, or maximum-duration cuts.
    Longer speech remains whole and is rejected if it exceeds the cache limit.
    """
    if len(pcm) % 2:
        raise ValueError("PCM音声のサイズが不正です。")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    active = [index for index, sample in enumerate(samples) if abs(sample) > 64]
    if not active:
        raise ValueError("生成されたフィラーは無音です。")
    start = max(0, active[0] - int(RATE * 0.04))
    end = min(len(samples), active[-1] + 1 + int(RATE * 0.06))
    trimmed = pcm[start * 2:end * 2]
    if len(trimmed) > MAX_AUDIO_BYTES:
        raise ValueError(f"生成されたフィラーが{MAX_AUDIO_SECONDS}秒を超えました。語尾を切らずに中止します。")
    return trimmed, {"original_seconds": len(pcm) / (RATE * 2),
                     "audio_seconds": len(trimmed) / (RATE * 2),
                     "trimmed_leading_seconds": start / RATE,
                     "trimmed_trailing_seconds": (len(samples) - end) / RATE,
                     "peak": max(abs(sample) for sample in samples),
                     "full_scale_samples": sum(abs(sample) >= 32767 for sample in samples),
                     "trim_policy": TRIM_POLICY}


async def prepare_gpt(args) -> dict:
    backend = "gpt"
    client = SpeechClient(model=args.gpt_model, voice=args.gpt_voice)
    started = time.monotonic()
    try:
        existing = load_filler_clips(args.output, backend, model=client.model,
                                     voice=client.voice)
        if existing and not args.force:
            return {"backend": backend, "model": client.model, "voice": client.voice,
                    "status": "reused", "clips": [clip.metadata() for clip in existing],
                    "synthesis_calls": 0}
        rows, clips = [], []
        for text in FILLER_TEXTS:
            pcm = b"".join([chunk async for chunk in client.stream(text)])
            prepared, metrics = trim_outer_silence(pcm)
            clips.append((text, prepared))
            rows.append({"text": text, **metrics})
        saved = write_filler_clips(args.output, backend, model=client.model,
                                   voice=client.voice, clips=clips)
        return {"backend": backend, "model": client.model, "voice": client.voice,
                "status": "generated",
                "elapsed_seconds": round(time.monotonic() - started, 4),
                "clips": [{**clip.metadata(), **row} for clip, row in zip(saved, rows)],
                "synthesis_calls": len(client.calls),
                "human_listening_verified": False,
                "note": "本文と同じ声・合成設定を使用。外側の無音のみ除去。自然さの人間による試聴評価は未実施。",
                "calls": client.calls}
    finally:
        await client.close()


async def run(args):
    result = await prepare_gpt(args)
    results = [result]
    print(json.dumps({key: value for key, value in result.items() if key != "calls"}, ensure_ascii=False))
    if args.report:
        from filler_cache import _atomic_write
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(args.report, json.dumps({"results": results}, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")


def main():
    parser = argparse.ArgumentParser(description="Jev用フィラーを事前生成します。GPT音声は少量のAPI料金が発生します。")
    parser.add_argument("--backend", choices=("gpt",), default="gpt")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "fillers")
    parser.add_argument("--gpt-model", default=MODEL)
    parser.add_argument("--gpt-voice", default=VOICE)
    parser.add_argument("--force", action="store_true", help="検証済みキャッシュがあっても再生成する")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        # Never print exceptions from network libraries, URLs, or credentials.
        print(f"フィラーの準備に失敗しました ({type(exc).__name__})。音声サービスと設定を確認してください。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
