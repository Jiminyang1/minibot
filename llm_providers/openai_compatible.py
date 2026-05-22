"""OpenAI-compatible LLM provider adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from ..llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from ..llm_profile import LLMProfile
from ..tools.definitions import ModelToolDefinition

if TYPE_CHECKING:
    from ..runtime.messages import ModelMessage


class OpenAICompatibleClient(LLMClient):
    """Works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, local, ...)."""

    def __init__(self, profile: LLMProfile) -> None:
        from openai import OpenAI

        if not profile.api_key:
            raise RuntimeError(
                f"缺少 {profile.provider} API key，请在 .env 或环境变量里设置。"
            )

        self.model = profile.model
        self.provider = profile.provider
        self.base_url = profile.base_url or ""
        self.compat = profile.compat
        kwargs: dict[str, Any] = {"api_key": profile.api_key}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        self._client = OpenAI(**kwargs)

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        from ..runtime.messages import model_messages_to_openai

        request_model = model or self.model
        kwargs: dict[str, Any] = {
            "model": request_model,
            "messages": model_messages_to_openai(
                messages,
                include_reasoning_content=self.compat.include_reasoning_content,
            ),
        }
        if tools:
            kwargs["tools"] = model_tool_definitions_to_openai(tools)

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
            reasoning_content=_extract_reasoning_content(msg),
            tool_calls=tool_calls,
            usage=_extract_token_usage(getattr(resp, "usage", None)),
            debug=_build_response_debug(
                raw_content=msg.content,
                finish_reason=getattr(resp.choices[0], "finish_reason", None),
                tool_call_count=len(tool_calls),
            ),
        )


def model_tool_definition_to_openai(
    tool: ModelToolDefinition,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def model_tool_definitions_to_openai(
    tools: list[ModelToolDefinition],
) -> list[dict[str, Any]]:
    return [model_tool_definition_to_openai(tool) for tool in tools]


def _extract_reasoning_content(msg: Any) -> str | None:
    """Capture provider-specific reasoning text (DeepSeek thinking mode, etc.)."""
    raw = getattr(msg, "reasoning_content", None)
    if raw is None and hasattr(msg, "model_extra"):
        extra = getattr(msg, "model_extra", None) or {}
        raw = extra.get("reasoning_content")
    if isinstance(raw, str) and raw.strip():
        return raw
    return None


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
