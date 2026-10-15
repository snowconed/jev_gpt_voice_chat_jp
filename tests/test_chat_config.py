import json
from pathlib import Path
import tempfile
import unittest

from chat_config import ChatSettings, load_settings


class ChatConfigTests(unittest.TestCase):
    def test_shipped_configuration_is_local_and_editable(self):
        settings = load_settings()
        self.assertEqual((settings.model, settings.reasoning_effort, settings.service_tier),
                         ("gpt-6.1-sol", "low", "ultrafast"))
        self.assertEqual(settings.max_output_tokens, 2000)
        self.assertTrue(settings.prompt_file.is_file())
        self.assertIn("日本語", settings.system_prompt)

    def test_relative_prompt_and_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.txt").write_text("短く日本語で答えてください。", encoding="utf-8")
            (root / "settings.json").write_text(json.dumps({"model": "other-model",
                "reasoning_effort": "low", "service_tier": "priority", "prompt_file": "prompt.txt"}), encoding="utf-8")
            settings = load_settings(root / "settings.json")
            self.assertEqual(settings.model, "other-model")
            self.assertEqual(settings.system_prompt, "短く日本語で答えてください。")
            self.assertEqual(settings.prompt_file, (root / "prompt.txt").resolve())

    def test_inline_prompt_does_not_read_default_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text('{"system_prompt":"会話してください。"}', encoding="utf-8-sig")
            settings = load_settings(config)
            self.assertEqual(settings.system_prompt, "会話してください。")
            self.assertIsNone(settings.prompt_file)

    def test_unsafe_or_ambiguous_configuration_is_rejected(self):
        cases = [{"prompt_file": "../outside.txt"}, {"prompt_file": ""},
                 {"system_prompt": "a", "prompt_file": "p.txt"},
                 {"system_prompt": "a", "api_key": "offline-must-not-read"}, [], None]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            for values in cases:
                with self.subTest(values=values):
                    config.write_text(json.dumps(values), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_settings(config)

    def test_types_and_limits(self):
        cases = [{"model": "bad\nmodel"}, {"model": None}, {"reasoning_effort": "extreme"},
                 {"service_tier": "secret"}, {"system_prompt": " "}, {"greeting": ""},
                 {"greeting": "あ" * 2001}, {"max_output_tokens": True},
                 {"max_output_tokens": 63}, {"timeout_seconds": float("nan")},
                 {"timeout_seconds": True}, {"timeout_seconds": 121},
                 {"attempt_timeout_seconds": 0}, {"attempt_timeout_seconds": True},
                 {"attempt_timeout_seconds": float("nan")},
                 {"max_retries": 2}, {"max_retries": True}, {"max_retries": -1}]
        for values in cases:
            with self.subTest(values=list(values)):
                with self.assertRaises(ValueError):
                    ChatSettings(**values)

    def test_missing_or_invalid_file_has_safe_message(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "private-config.json"
            for contents in (None, "{not-json"):
                if contents:
                    config.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError) as error:
                    load_settings(config)
                self.assertNotIn(directory, str(error.exception))


if __name__ == "__main__":
    unittest.main()
