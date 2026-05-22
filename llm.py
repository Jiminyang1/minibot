"""LLM client abstraction for MiniBot.

Provides a provider-agnostic interface so the Agent Core
never depends on a specific SDK.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime.messages import ModelMessage
    from .tools.definitions import ModelToolDefinition


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
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    debug: dict[str, Any] | None = None


# ── abstract client ──────────────────────────────────────────────


class LLMClient(abc.ABC):
    """Interface that every LLM provider must implement."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse: ...
