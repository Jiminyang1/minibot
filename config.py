"""Centralized configuration for MiniBot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env(package_dir: Path | None = None) -> None:
    """Load .env from *package_dir* (default: this file's directory).

    Uses ``setdefault`` so real environment variables always win.
    """
    env_path = (package_dir or Path(__file__).resolve().parent) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    model: str = "gpt-5.4-mini"
    max_history_turns: int = 40
    compact_token_threshold: int = 40000
    compact_keep_recent: int = 10

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            model=os.environ.get("MINIBOT_MODEL", cls.model),
        )
