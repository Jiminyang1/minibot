"""LLM client abstraction for MiniBot.

Provides a provider-agnostic interface so the Agent Core
never depends on a specific SDK.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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


@dataclass(frozen=True)
class LLMStreamEvent:
    """One element of a streamed model response.

    Contract: a stream yields zero or more ``text`` / ``reasoning`` deltas,
    then exactly one ``response`` event carrying the complete ``LLMResponse``
    as its final element. Deltas are advisory; the terminal response is the
    single source of truth (usage and tool calls ride on it).
    """

    kind: Literal["text", "reasoning", "response"]
    text: str = ""
    response: LLMResponse | None = None

    @classmethod
    def text_delta(cls, text: str) -> "LLMStreamEvent":
        return cls(kind="text", text=text)

    @classmethod
    def reasoning_delta(cls, text: str) -> "LLMStreamEvent":
        return cls(kind="reasoning", text=text)

    @classmethod
    def completed(cls, response: LLMResponse) -> "LLMStreamEvent":
        return cls(kind="response", response=response)


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

    def chat_stream(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """Stream a model response.

        The default wraps ``chat`` in a single terminal event, so providers
        (and test fakes) without native streaming keep working unchanged and
        callers never need a streaming/non-streaming branch.
        """
        yield LLMStreamEvent.completed(self.chat(messages, tools, model))
