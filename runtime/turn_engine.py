"""Turn orchestration for MiniBot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import time
from typing import TYPE_CHECKING

from .agent_runner import PartialRunError, RunSpec
from ..run_log import RunLogRecord, make_run_id, preview_text, utc_now
from ..session import MessageEvent, Session, SessionManager

if TYPE_CHECKING:
    from .agent_runner import AgentRunner
    from ..config import Config
    from .context_manager import ContextManager
    from ..run_log import RunLogStore


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one handled user turn."""

    reply: str
    did_compact: bool
    compact_message: str | None = None


class TurnEngine:
    """Coordinate one full user turn: context prep, runner, persistence."""

    def __init__(
        self,
        runner: AgentRunner,
        manager: SessionManager,
        config: Config,
        *,
        context_manager: ContextManager,
        event_handler: Callable[[str], None] | None = None,
        run_log_store: RunLogStore | None = None,
    ) -> None:
        self.runner = runner
        self.manager = manager
        self.config = config
        self.context_manager = context_manager
        self.event_handler = event_handler
        self.run_log_store = run_log_store

    def handle_turn(self, session: Session, user_input: str) -> TurnResult:
        run_id = make_run_id()
        timestamp = utc_now()
        started = time.perf_counter()
        turn_index = session.turn_count() + 1

        prepared = None
        reply: str | None = None
        turn_events: list[MessageEvent] = []
        usage = None

        try:
            self._emit_current_context_usage(session)
            prepared = self.context_manager.prepare_for_turn(
                session=session,
                user_input=user_input,
            )
            if prepared.did_compact:
                self.manager.save(session)
            user_event = MessageEvent.create(role="user", content=user_input)
            session.add_message(user_event)
            self.manager.append_messages(session.session_id, [user_event])
            self.manager.update_metadata(session)

            run_spec = RunSpec(
                session_id=session.session_id,
                model=self.config.model,
                messages=prepared.messages,
                tool_definitions=prepared.tool_definitions,
                max_iterations=self.config.max_iterations,
            )
            outcome = self.runner.run(run_spec)
            reply = outcome.reply
            turn_events = outcome.events
            usage = outcome.usage
            (
                mcp_tool_call_count,
                mcp_servers_used,
                mcp_transports_used,
                mcp_error_count,
            ) = self._summarize_mcp_usage(turn_events)
            for event in turn_events:
                session.add_message(event)
            self.manager.append_messages(session.session_id, turn_events)
            self.manager.update_metadata(session)

            result = TurnResult(
                reply=reply,
                did_compact=prepared.did_compact,
                compact_message=prepared.compact_message,
            )
            self._append_run_log(
                RunLogRecord(
                    run_id=run_id,
                    session_id=session.session_id,
                    turn_index=turn_index,
                    timestamp=timestamp,
                    ended_at=utc_now(),
                    status="success",
                    model=self.config.model,
                    user_input_preview=preview_text(user_input, 120),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    did_compact=prepared.did_compact,
                    compact_message=prepared.compact_message,
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                    total_tokens=None if usage is None else usage.total_tokens,
                    llm_call_count=self._count_messages(turn_events, role="assistant"),
                    tool_call_count=self._count_messages(turn_events, role="tool"),
                    tools_used=self._tools_used(turn_events),
                    mcp_tool_call_count=mcp_tool_call_count,
                    mcp_servers_used=mcp_servers_used,
                    mcp_transports_used=mcp_transports_used,
                    mcp_error_count=mcp_error_count,
                    final_reply_preview=preview_text(reply, 200),
                    error_type=None,
                    error_message_preview=None,
                )
            )
            return result
        except PartialRunError as exc:
            reply = exc.reply
            turn_events = exc.events
            usage = exc.usage
            (
                mcp_tool_call_count,
                mcp_servers_used,
                mcp_transports_used,
                mcp_error_count,
            ) = self._summarize_mcp_usage(turn_events)
            self._persist_turn_events(session, turn_events)
            self._append_run_log(
                RunLogRecord(
                    run_id=run_id,
                    session_id=session.session_id,
                    turn_index=turn_index,
                    timestamp=timestamp,
                    ended_at=utc_now(),
                    status="failed",
                    model=self.config.model,
                    user_input_preview=preview_text(user_input, 120),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    did_compact=False if prepared is None else prepared.did_compact,
                    compact_message=None if prepared is None else prepared.compact_message,
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                    total_tokens=None if usage is None else usage.total_tokens,
                    llm_call_count=self._count_messages(turn_events, role="assistant"),
                    tool_call_count=self._count_messages(turn_events, role="tool"),
                    tools_used=self._tools_used(turn_events),
                    mcp_tool_call_count=mcp_tool_call_count,
                    mcp_servers_used=mcp_servers_used,
                    mcp_transports_used=mcp_transports_used,
                    mcp_error_count=mcp_error_count,
                    final_reply_preview=(
                        None if reply is None else preview_text(reply, 200)
                    ),
                    error_type=type(exc.cause).__name__,
                    error_message_preview=preview_text(str(exc.cause), 200),
                )
            )
            raise exc.cause from exc
        except Exception as exc:
            (
                mcp_tool_call_count,
                mcp_servers_used,
                mcp_transports_used,
                mcp_error_count,
            ) = self._summarize_mcp_usage(turn_events)
            self._append_run_log(
                RunLogRecord(
                    run_id=run_id,
                    session_id=session.session_id,
                    turn_index=turn_index,
                    timestamp=timestamp,
                    ended_at=utc_now(),
                    status="failed",
                    model=self.config.model,
                    user_input_preview=preview_text(user_input, 120),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    did_compact=False if prepared is None else prepared.did_compact,
                    compact_message=None if prepared is None else prepared.compact_message,
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                    total_tokens=None if usage is None else usage.total_tokens,
                    llm_call_count=self._count_messages(turn_events, role="assistant"),
                    tool_call_count=self._count_messages(turn_events, role="tool"),
                    tools_used=self._tools_used(turn_events),
                    mcp_tool_call_count=mcp_tool_call_count,
                    mcp_servers_used=mcp_servers_used,
                    mcp_transports_used=mcp_transports_used,
                    mcp_error_count=mcp_error_count,
                    final_reply_preview=(
                        None if reply is None else preview_text(reply, 200)
                    ),
                    error_type=type(exc).__name__,
                    error_message_preview=preview_text(str(exc), 200),
                )
            )
            raise

    def compact_session(self, session: Session) -> tuple[bool, str]:
        did_compact, message = self.context_manager.compact_session(session=session)
        if did_compact:
            self.manager.save(session)
        return did_compact, message

    def delete_session(self, session_id: str) -> bool:
        """Remove a session directory and everything scoped under it."""
        return self.manager.delete_session(session_id)

    def list_available_skills(self) -> list[tuple[str, str, tuple[str, ...]]]:
        return self.context_manager.list_available_skills()

    def _emit_current_context_usage(self, session: Session) -> None:
        current_tokens = self.context_manager.estimate_visible_tokens(session=session)
        budget = self.context_manager.effective_input_budget
        self._emit(
            "当前上下文占用(不含本次输入): "
            f"{current_tokens}/{budget} tokens"
        )

    def _emit(self, message: str) -> None:
        if self.event_handler:
            self.event_handler(message)

    def _append_run_log(self, record: RunLogRecord) -> None:
        if self.run_log_store is None:
            return
        try:
            self.run_log_store.append(record)
        except Exception as exc:
            self._emit(f"写入 run log 失败: {exc}")

    @staticmethod
    def _count_messages(events: list[MessageEvent], *, role: str) -> int:
        return sum(1 for event in events if event.role == role)

    @staticmethod
    def _tools_used(events: list[MessageEvent]) -> list[str]:
        return [event.name for event in events if event.role == "tool" and event.name]

    def _summarize_mcp_usage(
        self,
        events: list[MessageEvent],
    ) -> tuple[int, list[str], list[str], int]:
        tool_call_count = 0
        servers_used: set[str] = set()
        transports_used: set[str] = set()
        error_count = 0

        for event in events:
            if event.role != "tool" or not event.name:
                continue
            tool = self.runner.tool_registry.get(event.name)
            if tool is None or getattr(tool, "source", "local") != "mcp":
                continue
            tool_call_count += 1

            server_name = getattr(tool, "server_name", None)
            if isinstance(server_name, str) and server_name:
                servers_used.add(server_name)

            transport_type = getattr(tool, "transport_type", None)
            if isinstance(transport_type, str) and transport_type:
                transports_used.add(transport_type)

            if self._tool_event_failed(event):
                error_count += 1

        return (
            tool_call_count,
            sorted(servers_used),
            sorted(transports_used),
            error_count,
        )

    @staticmethod
    def _tool_event_failed(event: MessageEvent) -> bool:
        try:
            payload = json.loads(event.content)
        except (TypeError, json.JSONDecodeError):
            return False
        return payload.get("ok") is False

    def _persist_turn_events(
        self,
        session: Session,
        turn_events: list[MessageEvent],
    ) -> None:
        if not turn_events:
            return
        for event in turn_events:
            session.add_message(event)
        self.manager.append_messages(session.session_id, turn_events)
        self.manager.update_metadata(session)
