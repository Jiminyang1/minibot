"""Runtime message types produced by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..session import MessageEvent


@dataclass(frozen=True)
class ModelToolCall:
    """Provider-agnostic model-request tool call."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelMessage:
    """Provider-agnostic message used inside runtime LLM requests."""

    role: str
    content: str
    tool_calls: list[ModelToolCall] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    reasoning_content: str | None = None

    @classmethod
    def create(
        cls,
        *,
        role: str,
        content: str,
        tool_calls: list[ModelToolCall] | list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        reasoning_content: str | None = None,
    ) -> "ModelMessage":
        return cls(
            role=role,
            content=content,
            tool_calls=_coerce_model_tool_calls(tool_calls),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reasoning_content=reasoning_content,
        )


@dataclass(frozen=True)
class AgentMessage:
    """One model-facing message emitted by the agent loop."""

    role: str
    content: str
    tool_calls: list[ModelToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    @classmethod
    def create(
        cls,
        *,
        role: str,
        content: str,
        tool_calls: list[ModelToolCall] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        reasoning_content: str | None = None,
    ) -> "AgentMessage":
        return cls(
            role=role,
            content=content,
            tool_calls=_coerce_model_tool_calls(tool_calls),
            tool_call_id=tool_call_id,
            name=name,
            reasoning_content=reasoning_content,
        )

    def with_content(self, content: str) -> "AgentMessage":
        return AgentMessage(
            role=self.role,
            content=content,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            name=self.name,
            reasoning_content=self.reasoning_content,
        )


def replace_final_assistant_reply(
    messages: list[AgentMessage],
    reply: str,
) -> list[AgentMessage]:
    """Return messages with the final non-tool assistant reply replaced."""

    updated = list(messages)
    for index in range(len(updated) - 1, -1, -1):
        message = updated[index]
        if message.role != "assistant" or message.tool_calls:
            continue
        updated[index] = message.with_content(reply)
        break
    return updated


def session_message_to_model(
    message: "MessageEvent",
    *,
    include_reasoning_content: bool = False,
) -> ModelMessage:
    reasoning_content = (
        message.reasoning_content
        if include_reasoning_content and message.role == "assistant"
        else None
    )
    return ModelMessage.create(
        role=message.role,
        content=message.content,
        tool_calls=_model_tool_calls_from_openai(message.tool_calls),
        tool_call_id=message.tool_call_id,
        tool_name=message.name,
        reasoning_content=reasoning_content,
    )


def agent_message_to_session(message: AgentMessage) -> "MessageEvent":
    from ..session import MessageEvent

    return MessageEvent.create(
        role=message.role,
        content=message.content,
        tool_calls=_openai_tool_calls_from_model(message.tool_calls),
        tool_call_id=message.tool_call_id,
        name=message.name,
        reasoning_content=message.reasoning_content,
    )


def model_message_to_openai(
    message: ModelMessage,
    *,
    include_reasoning_content: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = _openai_tool_calls_from_model(message.tool_calls)
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_name:
        data["name"] = message.tool_name
    if (
        include_reasoning_content
        and message.reasoning_content
        and message.role == "assistant"
    ):
        data["reasoning_content"] = message.reasoning_content
    return data


def model_messages_to_openai(
    messages: list[ModelMessage],
    *,
    include_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    return [
        model_message_to_openai(
            message,
            include_reasoning_content=include_reasoning_content,
        )
        for message in messages
    ]


def format_model_messages_for_summary(messages: list[ModelMessage]) -> str:
    """Flatten internal model messages into a readable summary transcript."""

    lines: list[str] = []
    for message in messages:
        role = message.role.upper()
        content = message.content.strip()
        if content:
            lines.append(f"{role}: {content}")
        for call in message.tool_calls or []:
            lines.append(
                f"ASSISTANT_TOOL_CALL: {call.name}({call.arguments})"
            )
        if message.role == "tool":
            name = message.tool_name or "tool"
            lines.append(f"TOOL_RESULT[{name}]: {content}")
    return "\n".join(lines)


def _model_tool_calls_from_openai(
    tool_calls: list[dict[str, Any]] | None,
) -> list[ModelToolCall] | None:
    if not tool_calls:
        return None
    converted: list[ModelToolCall] = []
    for call in tool_calls:
        function = call.get("function") or {}
        converted.append(
            ModelToolCall(
                id=str(call.get("id", "")),
                name=str(function.get("name", "")),
                arguments=str(function.get("arguments", "{}")),
            )
        )
    return converted


def _openai_tool_calls_from_model(
    tool_calls: list[ModelToolCall] | None,
) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None
    converted = _coerce_model_tool_calls(tool_calls)
    if not converted:
        return None
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }
        for call in converted
    ]


def _coerce_model_tool_calls(
    tool_calls: list[ModelToolCall] | list[dict[str, Any]] | None,
) -> list[ModelToolCall] | None:
    if not tool_calls:
        return None
    converted: list[ModelToolCall] = []
    for call in tool_calls:
        if isinstance(call, ModelToolCall):
            converted.append(call)
            continue
        function = call.get("function") or {}
        converted.append(
            ModelToolCall(
                id=str(call.get("id", "")),
                name=str(function.get("name", "")),
                arguments=str(function.get("arguments", "{}")),
            )
        )
    return converted
