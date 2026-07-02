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


class SessionEntry:
    """One append-only event in a session log."""

    __slots__ = (
        "id",
        "type",
        "created_at",
        "message",
        "summary",
        "first_kept_entry_id",
        "tokens_before",
        "details",
    )

    def __init__(
        self,
        *,
        type: str,
        id: str | None = None,
        created_at: str | None = None,
        message: MessageEvent | None = None,
        summary: str | None = None,
        first_kept_entry_id: str | None = None,
        tokens_before: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if type not in {"message", "compaction"}:
            raise ValueError(f"unknown session entry type: {type}")
        self.type = type
        self.id = id or self._new_id(type)
        self.created_at = created_at or utc_now()
        self.message = message
        self.summary = summary
        self.first_kept_entry_id = first_kept_entry_id
        self.tokens_before = tokens_before
        self.details = dict(details) if details else None

    @classmethod
    def from_message(cls, message: MessageEvent) -> "SessionEntry":
        return cls(
            type="message",
            id=message.id,
            created_at=message.created_at,
            message=message,
        )

    @classmethod
    def compaction(
        cls,
        *,
        summary: str,
        first_kept_entry_id: str | None,
        tokens_before: int | None = None,
        details: dict[str, Any] | None = None,
        id: str | None = None,
        created_at: str | None = None,
    ) -> "SessionEntry":
        return cls(
            type="compaction",
            id=id,
            created_at=created_at,
            summary=summary.strip(),
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.type == "message":
            if self.message is None:
                raise ValueError("message entry missing message payload")
            return {"type": "message", **self.message.to_dict()}

        data: dict[str, Any] = {
            "type": "compaction",
            "id": self.id,
            "created_at": self.created_at,
            "summary": self.summary or "",
            "first_kept_entry_id": self.first_kept_entry_id,
        }
        if self.tokens_before is not None:
            data["tokens_before"] = self.tokens_before
        if self.details:
            data["details"] = self.details
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionEntry":
        entry_type = str(data["type"])
        if entry_type == "message":
            message = MessageEvent.from_dict(data)
            return cls.from_message(message)
        if entry_type == "compaction":
            tokens_before = data.get("tokens_before")
            details = data.get("details")
            return cls.compaction(
                id=str(data["id"]),
                created_at=str(data["created_at"]) if data.get("created_at") else None,
                summary=str(data["summary"]),
                first_kept_entry_id=(
                    str(data["first_kept_entry_id"])
                    if data.get("first_kept_entry_id")
                    else None
                ),
                tokens_before=tokens_before if isinstance(tokens_before, int) else None,
                details=details if isinstance(details, dict) else None,
            )
        raise ValueError(f"unknown session entry type: {entry_type}")

    @staticmethod
    def _new_id(type: str) -> str:
        prefix = "c" if type == "compaction" else "e"
        return f"{prefix}_" + uuid.uuid4().hex[:12]


class SessionContextProjector:
    """Project append-only session entries into model-visible messages."""

    @classmethod
    def project_messages(cls, entries: list[SessionEntry]) -> list[MessageEvent]:
        latest_compaction_index = cls._latest_compaction_index(entries)
        if latest_compaction_index is None:
            return cls._filter_incomplete_tool_transactions(
                [entry.message for entry in entries if entry.message is not None]
            )

        compaction = entries[latest_compaction_index]
        projected: list[MessageEvent] = [
            MessageEvent(
                id=f"{compaction.id}_summary",
                role="assistant",
                content="[Summary of earlier conversation]\n"
                + (compaction.summary or "").strip(),
                created_at=compaction.created_at,
            )
        ]
        projected.extend(cls._kept_messages_before(entries, latest_compaction_index))
        projected.extend(
            entry.message
            for entry in entries[latest_compaction_index + 1 :]
            if entry.message is not None
        )
        return cls._filter_incomplete_tool_transactions(projected)

    @staticmethod
    def _latest_compaction_index(entries: list[SessionEntry]) -> int | None:
        for index in range(len(entries) - 1, -1, -1):
            if entries[index].type == "compaction":
                return index
        return None

    @staticmethod
    def _kept_messages_before(
        entries: list[SessionEntry],
        compaction_index: int,
    ) -> list[MessageEvent]:
        compaction = entries[compaction_index]
        first_kept_entry_id = compaction.first_kept_entry_id
        if first_kept_entry_id is None:
            return []

        kept: list[MessageEvent] = []
        collecting = False
        for entry in entries[:compaction_index]:
            if entry.type != "message" or entry.message is None:
                continue
            if entry.id == first_kept_entry_id:
                collecting = True
            if collecting:
                kept.append(entry.message)
        return kept

    @staticmethod
    def _filter_incomplete_tool_transactions(
        messages: list[MessageEvent],
    ) -> list[MessageEvent]:
        """Drop assistant tool-call blocks that do not have a complete tool tail."""
        projected: list[MessageEvent] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "assistant" and message.tool_calls:
                tool_messages: list[MessageEvent] = []
                cursor = index + 1
                while cursor < len(messages) and messages[cursor].role == "tool":
                    tool_messages.append(messages[cursor])
                    cursor += 1

                expected_ids = [str(call.get("id", "")) for call in message.tool_calls]
                actual_ids = [
                    str(tool_message.tool_call_id)
                    for tool_message in tool_messages
                    if tool_message.tool_call_id
                ]
                if expected_ids and len(expected_ids) == len(actual_ids) and set(expected_ids) == set(actual_ids):
                    projected.append(message)
                    projected.extend(tool_messages)
                index = cursor
                continue

            projected.append(message)
            index += 1

        return projected


class Session:
    """A single conversation with its messages."""

    __slots__ = (
        "session_id",
        "title",
        "created_at",
        "updated_at",
        "entries",
        "messages",
        "_message_count",
    )

    def __init__(
        self,
        session_id: str,
        title: str = "新会话",
        created_at: str | None = None,
        updated_at: str | None = None,
        entries: list[SessionEntry] | None = None,
        message_count: int | None = None,
    ) -> None:
        now = utc_now()
        self.session_id = session_id
        self.title = title
        self.created_at = created_at or now
        self.updated_at = updated_at or self.created_at
        self.entries = list(entries or [])
        self.messages = SessionContextProjector.project_messages(self.entries)
        self._message_count = message_count

    @property
    def message_count(self) -> int:
        return len(self.messages) if self.messages else (self._message_count or 0)

    def add_message(self, message: MessageEvent) -> None:
        self.entries.append(SessionEntry.from_message(message))
        self.messages = SessionContextProjector.project_messages(self.entries)
        self.updated_at = message.created_at
        if message.role == "user" and message.content.strip() and self.title == "新会话":
            self.title = preview(message.content)

    def turn_count(self) -> int:
        """Return the number of user-anchored turns in the session."""
        _, turns = self._split_preamble_and_turns()
        return len(turns)

    def compact_with_summary(
        self,
        summary_text: str,
        *,
        first_kept_entry_id: str | None,
        tokens_before: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> SessionEntry:
        """Append a compaction entry, refresh the projection, return the entry."""
        compaction_entry = SessionEntry.compaction(
            summary=summary_text,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
        )
        self.entries.append(compaction_entry)
        self.messages = SessionContextProjector.project_messages(self.entries)
        self.updated_at = compaction_entry.created_at
        return compaction_entry

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
