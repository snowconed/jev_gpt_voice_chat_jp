"""Check files that Git would publish; print locations, never matched secrets.

Checks both working files and staged blobs, including tracked ignored files.
This is a local heuristic check, not a scan of previous Git commits or a proof
that arbitrary personal information is absent. No network requests are made.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

MAX_TEXT_BYTES = 2 * 1024 * 1024
PRIVATE_DIRECTORIES = {
    ".venv", "venv", "__pycache__", "node_modules", "models", ".runtime",
    "validation", "data", "logs", "results", "reports", "build", "dist",
}
PRIVATE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".pcm", ".wav", ".mp3",
                    ".flac", ".ogg", ".m4a", ".jsonl", ".log", ".db", ".zip", ".7z"}
TOKEN_PATTERNS = (
    ("API-key-like value", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    ("GitHub-token-like value", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("AWS-access-key-like value", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("personal Windows path", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+")),
)
LITERAL_CREDENTIAL = re.compile(
    r'''(?i)["']?(?:api[_-]?key|openai_api_key|jev_api_key|typesafe_api_key|client_secret|access_token)["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']''')
ENV_NAMES = ("OPENAI_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY", "GITHUB_TOKEN", "GH_TOKEN")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    category: str
    source: str = "working tree"


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    parts = [part.lower() for part in path.parts]
    leaf = parts[-1] if parts else ""
    if any(part in PRIVATE_DIRECTORIES or part.startswith("data_") for part in parts[:-1]):
        return True
    if leaf.startswith(".env") and leaf != ".env.example":
        return True
    if leaf.startswith(("credentials", "secrets", "api_keys", "config.local.")):
        return True
    if path.suffix.lower() in PRIVATE_SUFFIXES or leaf.endswith((".pyc", ".pyo")) or ".sqlite" in leaf or ".7z." in leaf:
        return True
    return parts[:2] == ["assets", "fillers"] and parts != ["assets", "fillers", "readme.txt"]


def scan_text(name: str, data: bytes, environment=(), *, source="working tree") -> list[Finding]:
    findings = []
    if len(data) > MAX_TEXT_BYTES:
        return [Finding(name, 0, "unexpected large file", source)]
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [Finding(name, 0, "unexpected binary/non-UTF-8 file", source)]
    secrets = tuple(value for value in environment if isinstance(value, str) and len(value) >= 8)
    for number, line in enumerate(content.splitlines(), 1):
        if any(value in line for value in secrets):
            findings.append(Finding(name, number, "matches current credential environment", source))
        for category, pattern in TOKEN_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(name, number, category, source))
        for match in LITERAL_CREDENTIAL.finditer(line):
            value = match.group(1).lower()
            # Public test fixtures use explicit, recognizable dummy strings.
            if not value.startswith(("test", "fake", "dummy", "example", "offline", "not-a-", "your-", "replace-")):
                findings.append(Finding(name, number, "literal credential assignment", source))
    return findings


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError("Git inspection failed")
    return result.stdout


def inspect_repository(root: Path, environment=()) -> tuple[int, list[Finding]]:
    root = root.resolve()
    # Refuse an enclosing repository: the caller should check this project.
    top = Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != root:
        raise RuntimeError("Run this in the Jev voice chat repository root")
    names = {name.decode("utf-8") for name in git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0") if name}
    findings = []
    for name in sorted(names):
        if private_path(name):
            findings.append(Finding(name, 0, "private/generated file is a Git candidate"))
            continue
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            findings.append(Finding(name, 0, "symlink or path outside project"))
        elif path.is_file():
            findings.extend(scan_text(name, path.read_bytes(), environment))

    # The index can still contain a secret that was removed only from the
    # working copy. Inspect its actual blobs, never just the current files.
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        info, encoded_name = entry.split(b"\t", 1)
        mode, blob, stage = info.split()
        name = encoded_name.decode("utf-8")
        if mode != b"100644" and mode != b"100755":
            findings.append(Finding(name, 0, "non-regular staged file", "index"))
        elif not private_path(name):
            findings.extend(scan_text(name, git(root, "cat-file", "blob", blob.decode("ascii")), environment, source="index"))
    return len(names), list(dict.fromkeys(findings))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        count, findings = inspect_repository(args.root, [os.environ.get(name, "") for name in ENV_NAMES])
    except (OSError, RuntimeError, ValueError, UnicodeError):
        print("Publication check could not finish. Check Git availability, git init, and the repository path.", file=sys.stderr)
        return 2
    if findings:
        print("Publication check FAILED (matched values are never printed):")
        for item in findings:
            print(f"  {item.path}:{item.line}: {item.category} [{item.source}]")
        return 1
    print(f"Publication check passed: {count} Git candidate files and their staged versions checked.")
    print("No known credential/private-file findings. Previous commits and arbitrary personal text are outside this check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
