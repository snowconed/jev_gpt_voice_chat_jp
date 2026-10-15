"""Offline asset integrity, voice isolation, and silence-only trimming tests."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from filler_cache import (FILLER_TEXTS, MAX_AUDIO_BYTES, MAX_MANIFEST_BYTES,
                          filler_profile, load_filler_clips, profile_key,
                          write_filler_clips)
from prepare_fillers import prepare_gpt, trim_outer_silence
from speech_client import MODEL, VOICE


def audio(samples=12000, value=1000):
    return struct.pack("<h", value) * samples


class FillerCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "fillers"
        self.settings = {"model": MODEL, "voice": VOICE}

    def write(self, **settings):
        settings = self.settings | settings
        return write_filler_clips(self.root, "gpt", clips=[(text, audio()) for text in FILLER_TEXTS], **settings)

    def load(self, **settings):
        return load_filler_clips(self.root, "gpt", **(self.settings | settings))

    def manifest(self):
        return self.root / profile_key(filler_profile("gpt", **self.settings)) / "manifest.json"

    def change_manifest(self, update):
        path = self.manifest()
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_roundtrip_and_browser_metadata(self):
        written = self.write()
        self.assertEqual(self.load(), written)
        self.assertEqual(tuple(clip.text for clip in written), FILLER_TEXTS)
        self.assertNotEqual(written[0].id, written[1].id)
        self.assertEqual(written[0].audio_seconds, 0.5)
        self.assertEqual(written[0].metadata()["audio_bytes"], 24000)
        self.assertEqual(written[0].metadata()["rate"], 24000)
        self.assertNotIn("pcm", written[0].metadata())

    def test_missing_cache_is_disabled(self):
        self.assertEqual(self.load(), ())

    def test_changed_voice_model_or_instructions_never_reuses_assets(self):
        self.write()
        self.assertEqual(self.load(voice="coral"), ())
        self.assertEqual(self.load(model="another-tts"), ())
        with patch("filler_cache.INSTRUCTIONS", "違う読み上げ設定"):
            self.assertEqual(self.load(), ())

    def test_unsupported_backend_cannot_read_or_write_assets(self):
        self.write()
        for backend in ("voicevox", "piper", "both"):
            with self.subTest(backend=backend):
                self.assertEqual(load_filler_clips(self.root, backend, **self.settings), ())
                with self.assertRaises(ValueError):
                    write_filler_clips(self.root, backend,
                                       clips=[(FILLER_TEXTS[0], audio())], **self.settings)

    def test_corrupted_pcm_disables_whole_set(self):
        clips = self.write()
        path = self.manifest().parent / (clips[0].id + ".pcm")
        path.write_bytes(audio(value=2000))
        self.assertEqual(self.load(), ())

    def test_missing_one_clip_disables_whole_set(self):
        clips = self.write()
        (self.manifest().parent / (clips[0].id + ".pcm")).unlink()
        self.assertEqual(self.load(), ())

    def test_path_traversal_not_used(self):
        self.write()
        self.change_manifest(lambda value: value["clips"][0].update(id="../../secret"))
        self.assertEqual(self.load(), ())
        for model, voice in (("../model", VOICE), (MODEL, "../../voice")):
            self.assertEqual(self.load(model=model, voice=voice), ())

    def test_invalid_metadata_fails_closed(self):
        for change in (lambda x: x["profile"].update(voice="coral"),
                       lambda x: x["clips"][0].update(audio_bytes=True),
                       lambda x: x["clips"][0].update(audio_seconds=float("nan")),
                       lambda x: x["clips"][0].update(audio_seconds=1),
                       lambda x: x["clips"].append(x["clips"][0]),
                       lambda x: x["clips"][0].update(text="なるほど"),
                       lambda x: x.update(clips=[])):
            with self.subTest(change=change):
                self.write()
                self.change_manifest(change)
                self.assertEqual(self.load(), ())

    def test_bounded_manifest(self):
        self.write()
        self.manifest().write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
        self.assertEqual(self.load(), ())

    def test_invalid_writer_input_does_not_replace_existing_manifest(self):
        before = self.write()
        for clips in ([], [("なるほど", audio())], [(FILLER_TEXTS[0], b"\0" * 24000)],
                      [(FILLER_TEXTS[0], audio(1))], [(FILLER_TEXTS[0], b"x" * (MAX_AUDIO_BYTES + 2))],
                      [(FILLER_TEXTS[0], b"x" * 24001)], [(FILLER_TEXTS[0], audio())] * 2):
            with self.subTest(clips_length=len(clips)):
                with self.assertRaises(ValueError):
                    write_filler_clips(self.root, "gpt", clips=clips, **self.settings)
                self.assertEqual(self.load(), before)

    def test_atomic_failed_write_preserves_previous_manifest_and_cleans_temporary(self):
        before = self.write()
        with patch("filler_cache.os.replace", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                self.write()
        self.assertEqual(self.load(), before)
        self.assertEqual(list(self.root.rglob(".filler-*")), [])

    def test_pcm_hash_is_recorded(self):
        self.write()
        data = json.loads(self.manifest().read_text(encoding="utf-8"))
        self.assertEqual(data["clips"][0]["pcm_sha256"], hashlib.sha256(audio()).hexdigest())


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpt_generation_and_reuse_keep_the_selected_voice(self):
        clients = []

        class Speech:
            def __init__(self, *, model, voice):
                self.model, self.voice = model, voice
                self.calls, self.closed = [], False
                clients.append(self)

            async def stream(self, text):
                self.calls.append(text)
                yield audio()

            async def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), gpt_model=MODEL,
                                   gpt_voice=VOICE, force=False)
            with patch("prepare_fillers.SpeechClient", Speech):
                generated = await prepare_gpt(args)
                reused = await prepare_gpt(args)
            self.assertEqual(generated["backend"], "gpt")
            self.assertEqual((generated["model"], generated["voice"]), (MODEL, VOICE))
            self.assertEqual(clients[0].calls, list(FILLER_TEXTS))
            self.assertEqual(generated["synthesis_calls"], len(FILLER_TEXTS))
            self.assertEqual(reused["status"], "reused")
            self.assertEqual(reused["synthesis_calls"], 0)
            self.assertEqual(clients[1].calls, [])
            self.assertTrue(all(client.closed for client in clients))
            self.assertEqual(len(load_filler_clips(directory, "gpt", model=MODEL, voice=VOICE)),
                             len(FILLER_TEXTS))

    async def test_failed_generation_closes_client_without_publishing_partial_set(self):
        clients = []

        class Speech:
            def __init__(self, *, model, voice):
                self.model, self.voice = model, voice
                self.calls, self.closed = [], False
                clients.append(self)

            async def stream(self, text):
                self.calls.append(text)
                if len(self.calls) == 2:
                    raise RuntimeError("offline test synthesis failure")
                yield audio()

            async def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), gpt_model=MODEL,
                                   gpt_voice=VOICE, force=False)
            with patch("prepare_fillers.SpeechClient", Speech), self.assertRaises(RuntimeError):
                await prepare_gpt(args)
            self.assertTrue(clients[0].closed)
            self.assertEqual(list(Path(directory).rglob("*.pcm")), [])
            self.assertEqual(list(Path(directory).rglob("manifest.json")), [])


class TrimTests(unittest.TestCase):
    def test_only_outer_silence_trimmed_with_padding(self):
        speech = audio(2400) + b"\0\0" * 1200 + audio(2400, 2000)
        original = b"\0\0" * 4800 + speech + b"\0\0" * 7200
        trimmed, metrics = trim_outer_silence(original)
        self.assertEqual(trimmed, b"\0\0" * 960 + speech + b"\0\0" * 1440)
        self.assertAlmostEqual(metrics["trimmed_leading_seconds"], 0.16)
        self.assertAlmostEqual(metrics["trimmed_trailing_seconds"], 0.24)
        self.assertEqual(metrics["full_scale_samples"], 0)

    def test_speech_at_edges_is_not_cut(self):
        original = audio()
        self.assertEqual(trim_outer_silence(original)[0], original)

    def test_long_speech_is_rejected_not_cut(self):
        with self.assertRaises(ValueError):
            trim_outer_silence(audio(80000))

    def test_silence_and_odd_bytes_rejected(self):
        for original in (b"\0\0" * 12000, b"x"):
            with self.assertRaises(ValueError):
                trim_outer_silence(original)


if __name__ == "__main__":
    unittest.main()
