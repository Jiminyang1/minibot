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
    max_iterations: int = 20
    max_parallel_tools: int = 4
    max_history_turns: int = 40
    compact_token_threshold: int = 40000
    reserved_completion_tokens: int = 4096
    compact_keep_recent: int = 10
    auto_approve: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0。")
        if self.max_parallel_tools < 0:
            raise ValueError("max_parallel_tools 不能小于 0。")
        if self.compact_token_threshold <= 0:
            raise ValueError("compact_token_threshold 必须大于 0。")
        if self.reserved_completion_tokens <= 0:
            raise ValueError("reserved_completion_tokens 必须大于 0。")
        if self.reserved_completion_tokens >= self.compact_token_threshold:
            raise ValueError(
                "reserved_completion_tokens 必须小于 compact_token_threshold。"
            )

    @classmethod
    def from_env(cls) -> Config:
        def _get_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} 必须是整数。") from exc

        auto_approve = os.environ.get("MINIBOT_AUTO_APPROVE", "").lower() in {"1", "true", "yes"}

        return cls(
            model=os.environ.get("MINIBOT_MODEL", cls.model),
            max_iterations=_get_int("MINIBOT_MAX_ITERATIONS", cls.max_iterations),
            max_parallel_tools=_get_int(
                "MINIBOT_MAX_PARALLEL_TOOLS",
                cls.max_parallel_tools,
            ),
            auto_approve=auto_approve,
            max_history_turns=_get_int("MINIBOT_MAX_HISTORY_TURNS", cls.max_history_turns),
            compact_token_threshold=_get_int(
                "MINIBOT_COMPACT_TOKEN_THRESHOLD",
                cls.compact_token_threshold,
            ),
            reserved_completion_tokens=_get_int(
                "MINIBOT_RESERVED_COMPLETION_TOKENS",
                cls.reserved_completion_tokens,
            ),
            compact_keep_recent=_get_int(
                "MINIBOT_COMPACT_KEEP_RECENT",
                cls.compact_keep_recent,
            ),
        )
