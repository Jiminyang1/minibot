"""Pure compaction planning and summary metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import TYPE_CHECKING, Any

from .messages import (
    ModelMessage,
    format_model_messages_for_summary,
    session_message_to_model,
)
from .token_budget import estimate_messages_tokens

if TYPE_CHECKING:
    from ..session import MessageEvent


@dataclass(frozen=True)
class SummaryRequest:
    """Messages and prior summary state used to produce a compaction summary."""

    messages: list[ModelMessage]
    previous_summary: str | None = None
    turn_prefix_messages: list[ModelMessage] | None = None


@dataclass(frozen=True)
class CompactionPreparation:
    """Prepared cut point and summary inputs for one compaction."""

    first_kept_message_index: int
    messages_to_summarize: list[MessageEvent]
    turn_prefix_messages: list[MessageEvent]
    is_split_turn: bool
    previous_summary: str | None


def format_summary_request(request: SummaryRequest) -> str:
    parts: list[str] = []
    if request.previous_summary:
        parts.append(
            "<previous_summary>\n"
            + request.previous_summary.strip()
            + "\n</previous_summary>"
        )
    if request.messages:
        parts.append(
            "<conversation>\n"
            + format_model_messages_for_summary(request.messages)
            + "\n</conversation>"
        )
    if request.turn_prefix_messages:
        parts.append(
            "<split_turn_prefix>\n"
            + format_model_messages_for_summary(request.turn_prefix_messages)
            + "\n</split_turn_prefix>"
        )
    parts.append(
        "请基于以上内容生成或更新结构化 checkpoint 摘要。"
        "如果 previous_summary 存在，请将新 conversation 和 split_turn_prefix 合并进去，"
        "不要丢失仍然重要的旧信息。"
    )
    return "\n\n".join(parts)


def build_summary_request(
    messages: list[MessageEvent],
    *,
    previous_summary: str | None = None,
    turn_prefix_messages: list[MessageEvent] | None = None,
    include_reasoning_content: bool = False,
) -> SummaryRequest:
    return SummaryRequest(
        messages=[
            session_message_to_model(
                message,
                include_reasoning_content=include_reasoning_content,
            )
            for message in messages
        ],
        previous_summary=previous_summary,
        turn_prefix_messages=[
            session_message_to_model(
                message,
                include_reasoning_content=include_reasoning_content,
            )
            for message in turn_prefix_messages or []
        ],
    )


def find_cut_point(
    messages: list[MessageEvent],
    keep_recent_tokens: int,
    *,
    include_reasoning_content: bool = False,
) -> int:
    """Return the first kept message index for token-based compaction.

    Kept slices prefer user turn boundaries. If one turn alone exceeds the
    retention target, this may return a non-user split-turn cut point, but never
    a tool result.
    """

    preparation = prepare_compaction(
        messages,
        keep_recent_tokens,
        include_reasoning_content=include_reasoning_content,
        previous_summary=None,
    )
    return preparation.first_kept_message_index if preparation is not None else 0


def prepare_compaction(
    messages: list[MessageEvent],
    keep_recent_tokens: int,
    *,
    include_reasoning_content: bool = False,
    previous_summary: str | None = None,
    start_index: int = 0,
) -> CompactionPreparation | None:
    """Prepare a turn-aware compaction cut point and summary inputs."""

    if keep_recent_tokens <= 0 or start_index >= len(messages):
        return None
    start_index = max(0, start_index)

    threshold_index: int | None = None
    total = 0
    for index in range(len(messages) - 1, start_index - 1, -1):
        total += estimate_messages_tokens(
            [
                session_message_to_model(
                    messages[index],
                    include_reasoning_content=include_reasoning_content,
                )
            ],
            include_reasoning_content=include_reasoning_content,
        )
        if total >= keep_recent_tokens:
            threshold_index = index
            break

    if threshold_index is None:
        return None

    first_kept = _first_valid_cut_at_or_after(messages, threshold_index)
    if first_kept is None:
        return None

    turn_start = _turn_start_index(messages, first_kept, start_index=start_index)
    if (
        messages[first_kept].role != "user"
        and turn_start is not None
        and _turn_tokens(
            messages,
            turn_start,
            include_reasoning_content=include_reasoning_content,
        )
        <= keep_recent_tokens
    ):
        first_kept = turn_start
        turn_start = first_kept

    is_user_boundary = messages[first_kept].role == "user"
    is_split_turn = not is_user_boundary and turn_start is not None
    history_end = turn_start if is_split_turn and turn_start is not None else first_kept
    if history_end <= start_index and not is_split_turn:
        return None

    return CompactionPreparation(
        first_kept_message_index=first_kept,
        messages_to_summarize=messages[start_index:history_end],
        turn_prefix_messages=messages[turn_start:first_kept]
        if is_split_turn and turn_start is not None
        else [],
        is_split_turn=is_split_turn,
        previous_summary=previous_summary,
    )


def drop_projected_summary_message(
    messages: list[MessageEvent],
    *,
    previous_summary: str | None,
) -> list[MessageEvent]:
    """Drop the synthetic projector summary when previous_summary is explicit."""
    if not previous_summary or not messages:
        return messages
    first = messages[0]
    if first.role != "assistant":
        return messages
    if not str(first.id).endswith("_summary"):
        return messages
    if not first.content.startswith("[Summary of earlier conversation]\n"):
        return messages
    return messages[1:]


def summary_projection_offset(
    messages: list[MessageEvent],
    *,
    previous_summary: str | None,
) -> int:
    return len(messages) - len(
        drop_projected_summary_message(messages, previous_summary=previous_summary)
    )


def extract_compaction_details(
    *,
    previous_details: dict[str, Any] | None,
    messages: list[MessageEvent],
) -> dict[str, list[str]]:
    read_files: set[str] = set()
    modified_files: set[str] = set()
    _merge_previous_details(previous_details, read_files, modified_files)

    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            index += 1
            continue

        tool_messages: dict[str, MessageEvent] = {}
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].role == "tool":
            tool_message = messages[cursor]
            if tool_message.tool_call_id:
                tool_messages[str(tool_message.tool_call_id)] = tool_message
            cursor += 1

        for call in message.tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            call_id = str(call.get("id", ""))
            if name in {"write_file", "edit_file"}:
                path = _tool_path_argument(function.get("arguments"))
                if path:
                    modified_files.add(path)
            elif name == "read_file":
                path = _tool_path_argument(function.get("arguments"))
                if path:
                    read_files.add(path)
            elif name == "read_artifact":
                path = _read_artifact_file_path(tool_messages.get(call_id))
                if path:
                    read_files.add(path)

        index = cursor

    read_files.difference_update(modified_files)
    return {
        "read_files": sorted(read_files),
        "modified_files": sorted(modified_files),
    }


def append_file_details_to_summary(
    summary: str,
    details: dict[str, list[str]],
) -> str:
    summary = strip_file_detail_blocks(summary).strip()
    sections: list[str] = []
    read_files = details.get("read_files") or []
    modified_files = details.get("modified_files") or []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append(
            "<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>"
        )
    if not sections:
        return summary
    return summary + "\n\n" + "\n\n".join(sections)


def strip_file_detail_blocks(summary: str) -> str:
    summary = re.sub(
        r"\n*<read-files>\n.*?\n</read-files>",
        "",
        summary,
        flags=re.DOTALL,
    )
    return re.sub(
        r"\n*<modified-files>\n.*?\n</modified-files>",
        "",
        summary,
        flags=re.DOTALL,
    )


def _first_valid_cut_at_or_after(
    messages: list[MessageEvent],
    threshold_index: int,
) -> int | None:
    for index in range(threshold_index, len(messages)):
        if messages[index].role == "tool":
            continue
        if not _starts_with_incomplete_tool_transaction(messages, index):
            return index
    return None


def _starts_with_incomplete_tool_transaction(
    messages: list[MessageEvent],
    index: int,
) -> bool:
    message = messages[index]
    if message.role != "assistant" or not message.tool_calls:
        return False
    expected_ids = [str(call.get("id", "")) for call in message.tool_calls]
    if not expected_ids:
        return True
    cursor = index + 1
    actual_ids: list[str] = []
    while cursor < len(messages) and messages[cursor].role == "tool":
        if messages[cursor].tool_call_id:
            actual_ids.append(str(messages[cursor].tool_call_id))
        cursor += 1
    return len(expected_ids) != len(actual_ids) or set(expected_ids) != set(actual_ids)


def _turn_start_index(
    messages: list[MessageEvent],
    index: int,
    *,
    start_index: int = 0,
) -> int | None:
    for cursor in range(index, start_index - 1, -1):
        if messages[cursor].role == "user":
            return cursor
    return None


def _turn_tokens(
    messages: list[MessageEvent],
    turn_start: int,
    *,
    include_reasoning_content: bool,
) -> int:
    turn_end = len(messages)
    for cursor in range(turn_start + 1, len(messages)):
        if messages[cursor].role == "user":
            turn_end = cursor
            break
    return estimate_messages_tokens(
        [
            session_message_to_model(
                message,
                include_reasoning_content=include_reasoning_content,
            )
            for message in messages[turn_start:turn_end]
        ],
        include_reasoning_content=include_reasoning_content,
    )


def _merge_previous_details(
    previous_details: dict[str, Any] | None,
    read_files: set[str],
    modified_files: set[str],
) -> None:
    if not isinstance(previous_details, dict):
        return
    for value in previous_details.get("read_files", []):
        if isinstance(value, str) and value:
            read_files.add(value)
    for value in previous_details.get("modified_files", []):
        if isinstance(value, str) and value:
            modified_files.add(value)


def _tool_path_argument(arguments: Any) -> str | None:
    parsed: Any
    if isinstance(arguments, dict):
        parsed = arguments
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("path")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _read_artifact_file_path(message: MessageEvent | None) -> str | None:
    if message is None:
        return None
    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("kind") != "file":
        return None
    name = data.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name or None
