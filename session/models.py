"""Session domain models for MiniBot."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def preview(text: str, limit: int = 30) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


class MessageEvent:
    """One message in a conversation."""

    __slots__ = (
        "id",
        "role",
        "content",
        "created_at",
        "tool_calls",
        "tool_call_id",
        "name",
        "reasoning_content",
    )

    def __init__(
        self,
        *,
        role: str,
        content: str,
        id: str | None = None,
        created_at: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.id = id or ("m_" + uuid.uuid4().hex[:12])
        self.role = role
        self.content = content
        self.created_at = created_at or utc_now()
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name
        self.reasoning_content = reasoning_content

    @classmethod
    def create(
        cls,
        *,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        reasoning_content: str | None = None,
    ) -> "MessageEvent":
        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
            reasoning_content=reasoning_content,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.reasoning_content:
            data["reasoning_content"] = self.reasoning_content
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageEvent":
        return cls(
            id=data.get("id"),
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            created_at=data.get("created_at"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            reasoning_content=data.get("reasoning_content"),
        )

    def to_model_message(
        self,
        *,
        include_reasoning_content: bool = False,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.name:
            message["name"] = self.name
        if (
            include_reasoning_content
            and self.reasoning_content
            and self.role == "assistant"
        ):
            message["reasoning_content"] = self.reasoning_content
        return message


class Session:
    """A single conversation with its messages."""

    __slots__ = ("session_id", "title", "created_at", "updated_at", "messages", "_message_count")

    def __init__(
        self,
        session_id: str,
        title: str = "新会话",
        created_at: str | None = None,
        updated_at: str | None = None,
        messages: list[MessageEvent] | None = None,
        message_count: int | None = None,
    ) -> None:
        now = utc_now()
        self.session_id = session_id
        self.title = title
        self.created_at = created_at or now
        self.updated_at = updated_at or self.created_at
        self.messages: list[MessageEvent] = messages or []
        self._message_count = message_count

    @property
    def message_count(self) -> int:
        return len(self.messages) if self.messages else (self._message_count or 0)

    def add_message(self, message: MessageEvent) -> None:
        self.messages.append(message)
        self.updated_at = message.created_at
        if message.role == "user" and message.content.strip() and self.title == "新会话":
            self.title = preview(message.content)

    def turn_count(self) -> int:
        """Return the number of user-anchored turns in the session."""
        _, turns = self._split_preamble_and_turns()
        return len(turns)

    def history_for_model(
        self,
        max_turns: int = 40,
        *,
        include_reasoning_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Return model history while keeping tool-calling turns intact."""
        preamble, turns = self._split_preamble_and_turns()
        if max_turns > 0:
            turns = turns[-max_turns:]
        messages = [*preamble, *self._flatten_turns(turns)]
        return [
            message.to_model_message(
                include_reasoning_content=include_reasoning_content,
            )
            for message in messages
        ]

    def messages_to_compact(self, keep_recent_turns: int) -> list[MessageEvent]:
        """Return the older messages that should be summarized away."""
        preamble, turns = self._split_preamble_and_turns()
        if keep_recent_turns > 0:
            old_turns = turns[:-keep_recent_turns]
        else:
            old_turns = turns
        if not old_turns:
            return []
        return [*preamble, *self._flatten_turns(old_turns)]

    def compact_with_summary(
        self,
        summary_text: str,
        keep_recent_turns: int,
    ) -> tuple[int, int]:
        """Replace older turns with one summary message and the recent turns."""
        before = len(self.messages)
        _, turns = self._split_preamble_and_turns()
        if keep_recent_turns > 0:
            recent_turns = turns[-keep_recent_turns:]
        else:
            recent_turns = []
        summary_message = MessageEvent(
            role="assistant",
            content="[Summary of earlier conversation]\n" + summary_text.strip(),
        )
        self.messages = [summary_message, *self._flatten_turns(recent_turns)]
        self.updated_at = summary_message.created_at
        return before, len(self.messages)

    def _split_preamble_and_turns(
        self,
    ) -> tuple[list[MessageEvent], list[list[MessageEvent]]]:
        """Split leading non-user context from user-anchored turns."""
        preamble: list[MessageEvent] = []
        turns: list[list[MessageEvent]] = []
        current_turn: list[MessageEvent] | None = None

        for message in self.messages:
            if message.role == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [message]
                continue

            if current_turn is None:
                preamble.append(message)
            else:
                current_turn.append(message)

        if current_turn:
            turns.append(current_turn)

        return preamble, turns

    @staticmethod
    def _flatten_turns(turns: list[list[MessageEvent]]) -> list[MessageEvent]:
        return [message for turn in turns for message in turn]
