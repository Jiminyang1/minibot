"""Centralized configuration for MiniBot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


ApprovalMode: TypeAlias = Literal["ask", "always"]


def resolve_state_home() -> Path:
    """Return the global state home (sessions, runs, memory, MCP config).

    Assistant memory is centralized: conversations belong to the user, not
    to whichever directory the CLI happened to start in. The workspace only
    scopes the *tools* (fs/exec); it is recorded as session metadata.
    """
    raw = os.environ.get("MINIBOT_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".minibot").resolve()


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
    approval_mode: ApprovalMode = "ask"
    max_iterations: int = 20
    max_parallel_tools: int = 4
    context_window_tokens: int | None = None
    model_max_input_tokens: int | None = None
    model_max_output_tokens: int | None = None
    max_output_tokens: int = 4096
    compact_token_threshold: int | None = None
    compact_keep_recent_tokens: int = 16000
    llm_max_retries: int = 3

    def __post_init__(self) -> None:
        if self.approval_mode not in {"ask", "always"}:
            raise ValueError("approval_mode 必须是 ask 或 always。")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0。")
        if self.max_parallel_tools < 0:
            raise ValueError("max_parallel_tools 不能小于 0。")
        if (
            self.context_window_tokens is not None
            and self.context_window_tokens <= 0
        ):
            raise ValueError("context_window_tokens 必须大于 0。")
        if (
            self.model_max_input_tokens is not None
            and self.model_max_input_tokens <= 0
        ):
            raise ValueError("model_max_input_tokens 必须大于 0。")
        if (
            self.model_max_output_tokens is not None
            and self.model_max_output_tokens <= 0
        ):
            raise ValueError("model_max_output_tokens 必须大于 0。")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens 必须大于 0。")
        if (
            self.model_max_output_tokens is not None
            and self.max_output_tokens > self.model_max_output_tokens
        ):
            raise ValueError("max_output_tokens 不能超过模型最大输出。")
        if (
            self.compact_token_threshold is not None
            and self.compact_token_threshold <= 0
        ):
            raise ValueError("compact_token_threshold 必须大于 0。")
        if self.compact_keep_recent_tokens <= 0:
            raise ValueError("compact_keep_recent_tokens 必须大于 0。")
        if (
            self.compact_token_threshold is not None
            and self.compact_keep_recent_tokens >= self.compact_token_threshold
        ):
            raise ValueError(
                "compact_keep_recent_tokens 必须小于压缩触发阈值。"
            )
        if self.llm_max_retries < 0:
            raise ValueError("llm_max_retries 不能小于 0。")

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

        def _get_optional_int(name: str) -> int | None:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                return None
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} 必须是整数。") from exc

        approval_mode = _get_approval_mode()
        model = os.environ.get("MINIBOT_MODEL", cls.model)
        max_output_tokens = _get_optional_int("MINIBOT_MAX_OUTPUT_TOKENS")
        if max_output_tokens is None:
            max_output_tokens = _get_optional_int(
                "MINIBOT_RESERVED_COMPLETION_TOKENS"
            )

        return cls(
            model=model,
            approval_mode=approval_mode,
            max_iterations=_get_int("MINIBOT_MAX_ITERATIONS", cls.max_iterations),
            max_parallel_tools=_get_int(
                "MINIBOT_MAX_PARALLEL_TOOLS",
                cls.max_parallel_tools,
            ),
            context_window_tokens=_get_optional_int(
                "MINIBOT_CONTEXT_WINDOW_TOKENS"
            ),
            model_max_input_tokens=_get_optional_int(
                "MINIBOT_MODEL_MAX_INPUT_TOKENS"
            ),
            model_max_output_tokens=_get_optional_int(
                "MINIBOT_MODEL_MAX_OUTPUT_TOKENS"
            ),
            max_output_tokens=(
                cls.max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
            compact_token_threshold=_get_optional_int(
                "MINIBOT_COMPACT_TOKEN_THRESHOLD"
            ),
            compact_keep_recent_tokens=_get_int(
                "MINIBOT_COMPACT_KEEP_RECENT_TOKENS",
                cls.compact_keep_recent_tokens,
            ),
            llm_max_retries=_get_int("MINIBOT_LLM_MAX_RETRIES", cls.llm_max_retries),
        )


def _get_approval_mode() -> ApprovalMode:
    raw_mode = os.environ.get("MINIBOT_APPROVAL_MODE")
    if raw_mode is not None and raw_mode.strip():
        return _parse_approval_mode(raw_mode, env_name="MINIBOT_APPROVAL_MODE")
    return "ask"


def _parse_approval_mode(raw: str, *, env_name: str) -> ApprovalMode:
    value = raw.strip().lower()
    if value == "ask":
        return "ask"
    if value == "always":
        return "always"
    raise ValueError(f"{env_name} 必须是 ask 或 always。")
