"""Editable chat settings, independent of every experience/dialogue project.

load_settings(path=None) loads config.json beside this module by default.
Relative prompt_file paths are resolved inside the configuration directory.
For an injected configuration, system_prompt may replace prompt_file.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ChatSettings:
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "low"
    service_tier: str = "ultrafast"
    prompt_file: Path | None = None
    system_prompt: str = "日本語で気軽に会話してください。音声で聞きやすい短い返答にしてください。"
    greeting: str = "こんにちは。今日はどんなことを話しましょうか？"
    max_output_tokens: int = 2000
    timeout_seconds: float = 30.0
    attempt_timeout_seconds: float = 8.0
    max_retries: int = 1

    def __post_init__(self):
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}", self.model):
            raise ValueError("modelの指定が不正です。")
        if self.reasoning_effort not in ("none", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("reasoning_effortの指定が不正です。")
        if self.service_tier not in ("default", "auto", "fast", "ultrafast", "priority", "flex", "scale"):
            raise ValueError("service_tierの指定が不正です。")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip() or len(self.system_prompt) > 16000:
            raise ValueError("会話プロンプトは1〜16000文字で指定してください。")
        if not isinstance(self.greeting, str) or not self.greeting.strip() or len(self.greeting) > 2000:
            raise ValueError("挨拶は1〜2000文字で指定してください。")
        if type(self.max_output_tokens) is not int or not 64 <= self.max_output_tokens <= 2000:
            raise ValueError("max_output_tokensは64〜2000の整数で指定してください。")
        if (type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds)
                or not 1 <= self.timeout_seconds <= 120):
            raise ValueError("timeout_secondsは1〜120秒で指定してください。")
        if (type(self.attempt_timeout_seconds) not in (int, float)
                or not math.isfinite(self.attempt_timeout_seconds)
                or not .1 <= self.attempt_timeout_seconds <= 120):
            raise ValueError("attempt_timeout_secondsは0.1〜120秒で指定してください。")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 1:
            raise ValueError("max_retriesは0または1で指定してください。")


def load_settings(path=None):
    location = Path(path).resolve() if path is not None else ROOT / "config.json"
    try:
        if location.stat().st_size > 65536:
            raise ValueError("設定ファイルが大きすぎます。")
        values = json.loads(location.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("設定ファイルを読み込めません。") from None
    allowed = {"model", "reasoning_effort", "service_tier", "prompt_file", "system_prompt",
               "greeting", "max_output_tokens", "timeout_seconds", "attempt_timeout_seconds", "max_retries"}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("設定項目が不正です。")
    if "system_prompt" in values and "prompt_file" in values:
        raise ValueError("system_promptとprompt_fileはいずれか一方を指定してください。")
    if "system_prompt" not in values:
        filename = values.get("prompt_file", "prompts/chat_ja.txt")
        if not isinstance(filename, str) or not filename or len(filename) > 1000:
            raise ValueError("prompt_fileの指定が不正です。")
        target = (location.parent / filename).resolve()
        if not target.is_relative_to(location.parent):
            raise ValueError("prompt_fileは設定ファイルと同じフォルダ内で指定してください。")
        try:
            if target.stat().st_size > 65536:
                raise ValueError("会話プロンプトが大きすぎます。")
            values["system_prompt"] = target.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            raise ValueError("会話プロンプトを読み込めません。") from None
        values["prompt_file"] = target
    return ChatSettings(**values)
