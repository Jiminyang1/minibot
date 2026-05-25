"""Token budget estimation helpers for runtime request planning."""

from __future__ import annotations

import json
from typing import Any

from ..tools.definitions import ModelToolDefinition
from .messages import ModelMessage, model_messages_to_openai

_ENCODER: Any = None
_ENCODER_UNAVAILABLE = False


def _token_encoder() -> Any:
    global _ENCODER, _ENCODER_UNAVAILABLE
    if _ENCODER is not None or _ENCODER_UNAVAILABLE:
        return _ENCODER
    try:
        import tiktoken

        _ENCODER = tiktoken.encoding_for_model("gpt-4o")
    except (ImportError, KeyError):
        _ENCODER_UNAVAILABLE = True
    return _ENCODER


def estimate_text_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken when available, else a local heuristic."""
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, len(text) // 2)


def estimate_messages_tokens(
    messages: list[ModelMessage],
    *,
    include_reasoning_content: bool = True,
) -> int:
    """Estimate total tokens for provider-agnostic model messages."""
    total = 0
    for msg in model_messages_to_openai(
        messages,
        include_reasoning_content=include_reasoning_content,
    ):
        total += 4
        for value in msg.values():
            if isinstance(value, str):
                total += estimate_text_tokens(value)
            elif isinstance(value, list):
                total += estimate_text_tokens(json.dumps(value, ensure_ascii=False))
    total += 2
    return total


def estimate_request_tokens(
    messages: list[ModelMessage],
    tools: list[ModelToolDefinition] | None = None,
    *,
    include_reasoning_content: bool = True,
) -> int:
    """Estimate tokens for one concrete model request payload."""
    total = estimate_messages_tokens(
        messages,
        include_reasoning_content=include_reasoning_content,
    )
    if tools:
        from ..llm_providers.openai_compatible import (
            model_tool_definitions_to_openai,
        )

        total += estimate_text_tokens(
            json.dumps(
                model_tool_definitions_to_openai(tools),
                ensure_ascii=False,
            )
        )
    return total
