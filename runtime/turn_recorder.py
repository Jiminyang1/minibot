"""Persistence and run summary recording for handled turns."""

from __future__ import annotations

import json

from ..llm import TokenUsage
from ..run_log import RunLogRecord, RunLogStore, preview_text, utc_now
from ..session import MessageEvent, Session, SessionEntry, SessionManager
from ..tools.registry import ToolRegistry
from .context_manager import WorkingContext


class TurnRecorder:
    """Persist turn messages and append run summary records."""

    def __init__(
        self,
        *,
        manager: SessionManager,
        run_log_store: RunLogStore | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.manager = manager
        self.run_log_store = run_log_store
        self.tool_registry = tool_registry

    def on_message(self, session: Session, message: MessageEvent) -> MessageEvent:
        session.add_message(message)
        self.manager.append_entries(
            session.session_id,
            [SessionEntry.from_message(message)],
        )
        self.manager.update_metadata(session)
        return message

    def persist_pending_compaction(self, session: Session) -> None:
        if not self.flush_pending_compaction(session):
            raise RuntimeError("compaction entry missing")

    def flush_pending_compaction(self, session: Session) -> bool:
        entries = session.pop_pending_compaction_entries()
        if not entries:
            return False
        self.manager.append_entries(session.session_id, entries)
        self.manager.update_metadata(session)
        return True

    def record_run(
        self,
        *,
        run_id: str,
        session: Session,
        turn_index: int,
        timestamp: str,
        status: str,
        model: str,
        user_input: str,
        duration_ms: int,
        prepared: WorkingContext | None,
        messages: list[MessageEvent],
        usage: TokenUsage | None,
        reply: str | None,
        error: Exception | None = None,
    ) -> None:
        if self.run_log_store is None:
            return
        try:
            self.run_log_store.append(
                self.build_run_log(
                    run_id=run_id,
                    session=session,
                    turn_index=turn_index,
                    timestamp=timestamp,
                    status=status,
                    model=model,
                    user_input=user_input,
                    duration_ms=duration_ms,
                    prepared=prepared,
                    messages=messages,
                    usage=usage,
                    reply=reply,
                    error=error,
                )
            )
        except Exception:
            # Run-log persistence is best-effort; never fail a handled turn
            # because observability bookkeeping could not be written.
            return

    def build_run_log(
        self,
        *,
        run_id: str,
        session: Session,
        turn_index: int,
        timestamp: str,
        status: str,
        model: str,
        user_input: str,
        duration_ms: int,
        prepared: WorkingContext | None,
        messages: list[MessageEvent],
        usage: TokenUsage | None,
        reply: str | None,
        error: Exception | None = None,
    ) -> RunLogRecord:
        (
            mcp_tool_call_count,
            mcp_servers_used,
            mcp_transports_used,
            mcp_error_count,
        ) = self._summarize_mcp_usage(messages)
        return RunLogRecord(
            run_id=run_id,
            session_id=session.session_id,
            turn_index=turn_index,
            timestamp=timestamp,
            ended_at=utc_now(),
            status=status,  # type: ignore[arg-type]
            model=model,
            user_input_preview=preview_text(user_input, 120),
            duration_ms=duration_ms,
            did_compact=False if prepared is None else prepared.did_compact,
            compact_message=None if prepared is None else prepared.compact_message,
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
            total_tokens=None if usage is None else usage.total_tokens,
            llm_call_count=_count_messages(messages, role="assistant"),
            tool_call_count=_count_messages(messages, role="tool"),
            tools_used=_tools_used(messages),
            mcp_tool_call_count=mcp_tool_call_count,
            mcp_servers_used=mcp_servers_used,
            mcp_transports_used=mcp_transports_used,
            mcp_error_count=mcp_error_count,
            final_reply_preview=None if reply is None else preview_text(reply, 200),
            error_type=None if error is None else type(error).__name__,
            error_message_preview=(
                None if error is None else preview_text(str(error), 200)
            ),
        )

    def _summarize_mcp_usage(
        self,
        messages: list[MessageEvent],
    ) -> tuple[int, list[str], list[str], int]:
        tool_call_count = 0
        servers_used: set[str] = set()
        transports_used: set[str] = set()
        error_count = 0

        for message in messages:
            if message.role != "tool" or not message.name:
                continue
            tool = None
            if self.tool_registry is not None:
                tool = self.tool_registry.get(message.name)
            if tool is None or getattr(tool, "source", "local") != "mcp":
                continue
            tool_call_count += 1

            server_name = getattr(tool, "server_name", None)
            if isinstance(server_name, str) and server_name:
                servers_used.add(server_name)

            transport_type = getattr(tool, "transport_type", None)
            if isinstance(transport_type, str) and transport_type:
                transports_used.add(transport_type)

            if _tool_message_failed(message):
                error_count += 1

        return (
            tool_call_count,
            sorted(servers_used),
            sorted(transports_used),
            error_count,
        )


def _count_messages(messages: list[MessageEvent], *, role: str) -> int:
    return sum(1 for message in messages if message.role == role)


def _tools_used(messages: list[MessageEvent]) -> list[str]:
    return [
        message.name
        for message in messages
        if message.role == "tool" and message.name
    ]


def _tool_message_failed(message: MessageEvent) -> bool:
    try:
        payload = json.loads(message.content)
    except (TypeError, json.JSONDecodeError):
        return False
    return payload.get("ok") is False
