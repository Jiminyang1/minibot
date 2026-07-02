"""Session reduction mechanics: summary compaction and oversized-tool drops.

The loop decides *when* to reduce; this module owns *how*. Every reduction
appends a compaction entry to the session and persists it to disk in the same
call — there is no pending state to reconcile later.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import TYPE_CHECKING

from ..llm import LLMClient
from ..prompts import SUMMARY_SYSTEM_PROMPT
from .budget import TokenBudget
from .cancel import RunCancelled
from .compaction import (
    SummaryRequest,
    append_file_details_to_summary,
    build_summary_request,
    drop_projected_summary_message,
    extract_compaction_details,
    format_summary_request,
    prepare_compaction,
    summary_projection_offset,
)
from .context_builder import ContextBuilder
from .messages import ModelMessage

if TYPE_CHECKING:
    from ..session import MessageEvent, Session, SessionEntry, SessionManager
    from ..tools.registry import ToolRegistry


def make_summarizer(llm: LLMClient) -> Callable[[SummaryRequest], str]:
    """Create a summariser closure backed by the given LLM client."""

    def summarize(request: SummaryRequest) -> str:
        if not request.messages and not request.turn_prefix_messages:
            raise ValueError("没有可供摘要的历史消息。")
        formatted = format_summary_request(request)
        resp = llm.chat([
            ModelMessage.create(role="system", content=SUMMARY_SYSTEM_PROMPT),
            ModelMessage.create(role="user", content=formatted),
        ])
        summary = (resp.content or "").strip()
        if not summary:
            raise RuntimeError("模型没有返回有效摘要。")
        return summary

    return summarize


class Compactor:
    """Reduce an over-budget session and persist the result immediately."""

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        context_builder: ContextBuilder,
        budget: TokenBudget,
        tool_registry: ToolRegistry,
        summarizer: Callable[[SummaryRequest], str],
        keep_recent_tokens: int,
        include_reasoning_content: bool = False,
    ) -> None:
        self.session_manager = session_manager
        self.context_builder = context_builder
        self.budget = budget
        self.tool_registry = tool_registry
        self.summarizer = summarizer
        self.keep_recent_tokens = keep_recent_tokens
        self.include_reasoning_content = include_reasoning_content

    def reduce(
        self,
        session: Session,
        *,
        tokens_before: int,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Bring *session* back under budget or raise a user-actionable error."""
        did_compact, message = self._compact_once(
            session,
            tokens_before=tokens_before,
            cancel_event=cancel_event,
        )
        if not did_compact:
            drop_message = self._drop_read_only_tool_tail(
                session,
                tokens_before=tokens_before,
                cancel_event=cancel_event,
            )
            if drop_message is None:
                # Nothing reducible: raise with the request size that tripped
                # the budget so the caller sees an actionable error.
                self.budget.ensure_fits(
                    self.context_builder.build(list(session.messages)),
                    request_tokens=tokens_before,
                )
                return message
            self._verify_within_budget(session)
            return drop_message

        tokens_after = self._current_tokens(session)
        if tokens_after > self.budget.input_budget:
            drop_message = self._drop_read_only_tool_tail(
                session,
                tokens_before=tokens_after,
                cancel_event=cancel_event,
            )
            if drop_message is not None:
                message = f"{message}\n{drop_message}"
        self._verify_within_budget(session)
        return message

    def compact_now(self, session: Session) -> tuple[bool, str]:
        """Manual `/compact`: summarise once, report whether anything changed."""
        return self._compact_once(
            session,
            tokens_before=self._current_tokens(session),
            cancel_event=None,
        )

    def _compact_once(
        self,
        session: Session,
        *,
        tokens_before: int,
        cancel_event: threading.Event | None,
    ) -> tuple[bool, str]:
        projected_messages = list(session.messages)
        previous = self._latest_compaction_entry(session)
        previous_summary = previous.summary if previous and previous.summary else None
        previous_details = previous.details if previous is not None else None
        summary_offset = summary_projection_offset(
            projected_messages,
            previous_summary=previous_summary,
        )
        preparation = prepare_compaction(
            projected_messages,
            self.keep_recent_tokens,
            include_reasoning_content=self.include_reasoning_content,
            previous_summary=previous_summary,
            start_index=summary_offset,
        )
        if preparation is None or preparation.first_kept_message_index <= 0:
            return False, (
                "没有可在安全切点处压缩的旧上下文"
                "（最近上下文可能是单个过大的工具结果块，将转交丢弃处理）。"
            )

        self._check_cancel_event(cancel_event)
        summary = self.summarizer(
            build_summary_request(
                preparation.messages_to_summarize,
                previous_summary=preparation.previous_summary,
                turn_prefix_messages=preparation.turn_prefix_messages,
                include_reasoning_content=self.include_reasoning_content,
            )
        )
        compacted_messages = [
            *preparation.messages_to_summarize,
            *preparation.turn_prefix_messages,
        ]
        details = extract_compaction_details(
            previous_details=previous_details,
            messages=compacted_messages,
        )
        summary = append_file_details_to_summary(summary, details)
        self._check_cancel_event(cancel_event)

        before = len(session.messages)
        first_kept_index = preparation.first_kept_message_index
        self._apply_compaction(
            session,
            summary,
            first_kept_entry_id=projected_messages[first_kept_index].id,
            tokens_before=tokens_before,
            details=details,
        )
        after_tokens = self._current_tokens(session)
        return True, (
            f"已压缩: {before} -> {len(session.messages)} 条消息, "
            f"请求预算 {tokens_before} -> {after_tokens} tokens"
        )

    def _drop_read_only_tool_tail(
        self,
        session: Session,
        *,
        tokens_before: int,
        cancel_event: threading.Event | None,
    ) -> str | None:
        projected_messages = list(session.messages)
        block = self._latest_tool_transaction_block(projected_messages)
        if block is None:
            return None

        start, tool_names = block
        if not self._all_tools_read_only(tool_names):
            raise RuntimeError(
                "最新工具事务块超过上下文预算，且包含非只读工具，已中止继续推理。"
            )

        prefix = projected_messages[:start]
        summary = self._summarize_prefix_before_tool_drop(
            session,
            prefix,
            tool_names=tool_names,
            cancel_event=cancel_event,
        )
        previous = self._latest_compaction_entry(session)
        details = extract_compaction_details(
            previous_details=previous.details if previous is not None else None,
            messages=prefix,
        )
        summary = append_file_details_to_summary(summary, details)
        before = len(session.messages)
        self._apply_compaction(
            session,
            summary,
            first_kept_entry_id=None,
            tokens_before=tokens_before,
            details=details,
        )
        after_tokens = self._current_tokens(session)
        return (
            f"已丢弃过大的只读工具事务块: {', '.join(tool_names)}, "
            f"消息 {before} -> {len(session.messages)}, "
            f"请求预算 {tokens_before} -> {after_tokens} tokens"
        )

    def _apply_compaction(
        self,
        session: Session,
        summary: str,
        *,
        first_kept_entry_id: str | None,
        tokens_before: int,
        details: dict[str, list[str]] | None,
    ) -> SessionEntry:
        """Mutate the in-memory session and persist the entry in one step."""
        entry = session.compact_with_summary(
            summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
        )
        self.session_manager.append_entries(session.session_id, [entry])
        self.session_manager.update_metadata(session)
        return entry

    def _summarize_prefix_before_tool_drop(
        self,
        session: Session,
        prefix: list[MessageEvent],
        *,
        tool_names: list[str],
        cancel_event: threading.Event | None,
    ) -> str:
        note = (
            "[Omitted oversized read-only tool transaction]\n"
            f"tools: {', '.join(tool_names)}\n"
            "reason: tool results exceeded the model input budget before the model "
            "could consume them. The raw tool call and results remain in the session log."
        )
        previous_summary = self._latest_compaction_summary(session)
        prefix = drop_projected_summary_message(
            prefix,
            previous_summary=previous_summary,
        )
        if not prefix:
            if previous_summary:
                return previous_summary.strip() + "\n\n" + note
            return note

        self._check_cancel_event(cancel_event)
        summary = self.summarizer(
            build_summary_request(
                prefix,
                previous_summary=previous_summary,
                include_reasoning_content=self.include_reasoning_content,
            )
        )
        self._check_cancel_event(cancel_event)
        return summary.strip() + "\n\n" + note

    def _latest_tool_transaction_block(
        self,
        messages: list[MessageEvent],
    ) -> tuple[int, list[str]] | None:
        if len(messages) < 2:
            return None

        cursor = len(messages) - 1
        while cursor >= 0 and messages[cursor].role == "tool":
            cursor -= 1
        if cursor == len(messages) - 1:
            return None

        assistant = messages[cursor]
        if assistant.role != "assistant" or not assistant.tool_calls:
            return None

        expected_ids = [str(call.get("id", "")) for call in assistant.tool_calls]
        actual_ids = [
            str(message.tool_call_id)
            for message in messages[cursor + 1 :]
            if message.tool_call_id
        ]
        if not expected_ids or set(expected_ids) != set(actual_ids):
            return None

        names: list[str] = []
        for call in assistant.tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            if name:
                names.append(name)
        return (cursor, names) if names else None

    def _all_tools_read_only(self, tool_names: list[str]) -> bool:
        for name in tool_names:
            tool = self.tool_registry.get(name)
            if tool is None or not tool.read_only:
                return False
        return True

    def _latest_compaction_entry(self, session: Session) -> SessionEntry | None:
        for entry in reversed(session.entries):
            if entry.type == "compaction":
                return entry
        return None

    def _latest_compaction_summary(self, session: Session) -> str | None:
        entry = self._latest_compaction_entry(session)
        if entry is None or not entry.summary:
            return None
        return entry.summary

    def _current_tokens(self, session: Session) -> int:
        return self.budget.estimate(self.context_builder.build(list(session.messages)))

    def _verify_within_budget(self, session: Session) -> None:
        self.budget.ensure_fits(self.context_builder.build(list(session.messages)))

    @staticmethod
    def _check_cancel_event(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("run cancelled by user")
