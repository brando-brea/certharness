"""API key resolution: ANTHROPIC_API_KEY if set, else a key file outside the repo.

The key file (default ~/.config/certharness/anthropic_api_key, override with
CERTHARNESS_KEY_FILE) lives outside the working tree so it can never be committed,
and must be readable by its owner only. The key is handed straight to the SDK
client; it is never logged, cached, or placed in request params.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

DEFAULT_KEY_FILE = Path("~/.config/certharness/anthropic_api_key")


def key_file() -> Path:
    return Path(os.environ.get("CERTHARNESS_KEY_FILE", DEFAULT_KEY_FILE)).expanduser()


def key_source() -> str | None:
    """Where the key would come from ('env' or the file path), or None if unavailable."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "env"
    path = key_file()
    return str(path) if path.is_file() and path.stat().st_size > 0 else None


def key_file_private(path: Path | None = None) -> bool:
    path = path or key_file()
    return not (path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO))


def client_kwargs() -> dict[str, str]:
    """Extra kwargs for anthropic.Anthropic(): empty when the env var is set (the SDK
    reads it itself), otherwise the key from the key file."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    path = key_file()
    if not path.is_file():
        return {}
    if not key_file_private(path):
        raise PermissionError(f"{path} must be readable by its owner only (chmod 600)")
    return {"api_key": parse_key(path.read_text(encoding="utf-8"))}


def parse_key(text: str) -> str:
    """Accept a bare key or a dotenv-style line (ANTHROPIC_API_KEY=..., optionally quoted)."""
    value = text.strip()
    if "=" in value and not value.startswith("sk-"):
        value = value.split("=", 1)[1].strip()
    return value.strip("'\"")
