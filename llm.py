"""LLM client abstraction for MiniBot.

Provides a provider-agnostic interface so the Agent Core
never depends on a specific SDK.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any


# ── normalised response types ────────────────────────────────────


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token usage for one model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Provider-agnostic model response."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    debug: dict[str, Any] | None = None


# ── abstract client ──────────────────────────────────────────────


class LLMClient(abc.ABC):
    """Interface that every LLM provider must implement."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse: ...


# ── OpenAI-compatible implementation ─────────────────────────────


class OpenAIClient(LLMClient):
    """Works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, local, …)."""

    def __init__(self, model: str) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 OPENAI_API_KEY，请在 .env 或环境变量里设置。"
            )

        self.model = model
        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        return LLMResponse(
            content=_extract_message_content(msg.content),
            tool_calls=tool_calls,
            usage=_extract_token_usage(getattr(resp, "usage", None)),
            debug=_build_response_debug(
                raw_content=msg.content,
                finish_reason=getattr(resp.choices[0], "finish_reason", None),
                tool_call_count=len(tool_calls),
            ),
        )


def _extract_message_content(raw_content: Any) -> str | None:
    if raw_content is None:
        return None
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts) or None
    return str(raw_content)


def _build_response_debug(
    *,
    raw_content: Any,
    finish_reason: Any,
    tool_call_count: int,
) -> dict[str, Any]:
    return {
        "raw_content_type": type(raw_content).__name__,
        "raw_content_preview": _preview_raw_content(raw_content),
        "finish_reason": finish_reason if isinstance(finish_reason, str) else repr(finish_reason),
        "tool_call_count": tool_call_count,
    }


def _preview_raw_content(raw_content: Any, limit: int = 240) -> str:
    if raw_content is None:
        return "None"
    if isinstance(raw_content, str):
        compact = " ".join(raw_content.split())
        return compact if len(compact) <= limit else compact[:limit] + "..."
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            item_type = type(item).__name__
            text: str | None = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                candidate = item.get("text")
                if isinstance(candidate, str):
                    text = candidate
            else:
                candidate = getattr(item, "text", None)
                if isinstance(candidate, str):
                    text = candidate

            if text is not None:
                compact = " ".join(text.split())
                if len(compact) > 80:
                    compact = compact[:80] + "..."
                parts.append(f"{item_type}:{compact}")
            else:
                parts.append(item_type)

        preview = " | ".join(parts)
        return preview if len(preview) <= limit else preview[:limit] + "..."
    return repr(raw_content)


def _extract_token_usage(raw_usage: Any) -> TokenUsage | None:
    """Convert provider usage objects to the local TokenUsage shape."""
    if raw_usage is None:
        return None

    input_tokens = _coerce_token_count(
        _usage_value(raw_usage, "prompt_tokens", "input_tokens")
    )
    output_tokens = _coerce_token_count(
        _usage_value(raw_usage, "completion_tokens", "output_tokens")
    )
    total_tokens = _coerce_token_count(_usage_value(raw_usage, "total_tokens"))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _usage_value(raw_usage: Any, *names: str) -> Any:
    for name in names:
        if isinstance(raw_usage, dict) and name in raw_usage:
            return raw_usage[name]
        value = getattr(raw_usage, name, None)
        if value is not None:
            return value
    return None


def _coerce_token_count(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value
