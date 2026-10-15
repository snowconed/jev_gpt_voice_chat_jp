"""Regression checks for publishing only source, never printing credentials."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.check_publication import inspect_repository, main, private_path, scan_text


class PublicationTests(unittest.TestCase):
    def test_generated_and_private_paths_are_rejected(self):
        for name in (".env", ".env.local", ".env.production", "data/session.json", "data_jev/history.json",
                     "models/engine.bin", ".runtime/server.json", "validation/result.json", "logs/trace.txt",
                     "assets/fillers/profile/manifest.json", "audio.pcm", "private.key", "credentials.json"):
            with self.subTest(name=name):
                self.assertTrue(private_path(name))
        for name in (".env.example", "config.json", "assets/fillers/README.txt", "tests/test_jev_client.py"):
            with self.subTest(name=name):
                self.assertFalse(private_path(name))

    def test_known_token_and_environment_matches_do_not_store_value(self):
        token = "sk" + "-proj-" + "A" * 64
        opaque = "local-credential-" + "b" * 40
        found = scan_text("source.py", f"first\n{token}\n{opaque}\n".encode(), [opaque])
        self.assertEqual({item.line for item in found}, {2, 3})
        self.assertNotIn(token, repr(found))
        self.assertNotIn(opaque, repr(found))

    def test_unknown_literal_credentials_rejected_but_explicit_dummy_allowed(self):
        literal = 'api_key = "' + "sensitive" * 4 + '"'
        self.assertTrue(scan_text("settings.py", literal.encode()))
        self.assertFalse(scan_text("test.py", b'api_key = "offline-not-a-real-key"'))

    def test_environment_match_overrides_dummy_prefix(self):
        value = "test-" + "credential" * 3
        self.assertTrue(scan_text("settings.txt", value.encode(), [value]))

    def test_real_secret_formats_detected(self):
        for value in ("ghp" + "_" + "a" * 36, "AK" + "IA" + "A" * 16,
                      "-----BEGIN " + "PRIVATE KEY-----"):
            with self.subTest(prefix=value[:3]):
                self.assertTrue(scan_text("settings.txt", value.encode()))

    def test_binary_files_are_not_silently_published(self):
        self.assertTrue(scan_text("unexpected.bin", b"\xff\xfe\x00"))


class GitPublicationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".runtime"
        root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.command("init", "-q")
        (self.root / ".gitignore").write_text(".env\ndata_*/\n.runtime/\n", encoding="utf-8")
        (self.root / "README.md").write_text("public source", encoding="utf-8")

    def command(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_ignored_files_are_not_publication_candidates(self):
        (self.root / ".env").write_text("excluded value", encoding="utf-8")
        (self.root / "data_jev").mkdir()
        (self.root / "data_jev" / "session.json").write_text("private", encoding="utf-8")
        count, findings = inspect_repository(self.root)
        self.assertEqual(count, 2)
        self.assertEqual(findings, [])

    def test_forced_tracked_private_file_is_rejected_even_when_ignored(self):
        (self.root / ".env").write_text("not public", encoding="utf-8")
        self.command("add", "-f", ".env")
        _, findings = inspect_repository(self.root)
        self.assertTrue(any(item.path == ".env" for item in findings))

    def test_staged_secret_still_detected_after_working_copy_is_fixed(self):
        token = "sk" + "-proj-" + "B" * 64
        path = self.root / "settings.txt"
        path.write_text(token, encoding="utf-8")
        self.command("add", "settings.txt")
        path.write_text("clean working file", encoding="utf-8")
        _, findings = inspect_repository(self.root)
        self.assertTrue(any(item.path == "settings.txt" and item.source == "index" for item in findings))
        self.assertNotIn(token, repr(findings))

    def test_cli_never_prints_matching_environment_credential(self):
        value = "unique-test-credential-" + "q" * 25
        (self.root / "settings.txt").write_text(value, encoding="utf-8")
        output = io.StringIO()
        with patch.dict(os.environ, {"JEV_API_KEY":value}), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--root", str(self.root)]), 1)
        self.assertIn("settings.txt:1:", output.getvalue())
        self.assertNotIn(value, output.getvalue())


if __name__ == "__main__":
    unittest.main()
