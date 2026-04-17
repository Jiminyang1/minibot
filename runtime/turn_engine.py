"""Turn orchestration for MiniBot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .agent_runner import RunSpec
from ..session import MessageEvent, Session, SessionManager

if TYPE_CHECKING:
    from .agent_runner import AgentRunner
    from ..config import Config
    from .context_manager import ContextManager


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
    ) -> None:
        self.runner = runner
        self.manager = manager
        self.config = config
        self.context_manager = context_manager
        self.event_handler = event_handler

    def handle_turn(self, session: Session, user_input: str) -> TurnResult:
        self._emit_current_context_usage(session)
        prepared = self.context_manager.prepare_for_turn(
            session=session,
            user_input=user_input,
        )
        if prepared.matched_skills:
            rendered = ", ".join(
                f"{item.name}({item.mode})"
                for item in prepared.matched_skills
            )
            self._emit(f"命中 skills: {rendered}")
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
        reply, turn_events = self.runner.run(run_spec)
        for event in turn_events:
            session.add_message(event)
        self.manager.append_messages(session.session_id, turn_events)
        self.manager.update_metadata(session)

        return TurnResult(
            reply=reply,
            did_compact=prepared.did_compact,
            compact_message=prepared.compact_message,
        )

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
        self._emit(
            "当前上下文占用(不含本次输入): "
            f"{current_tokens}/{self.config.compact_token_threshold} tokens"
        )

    def _emit(self, message: str) -> None:
        if self.event_handler:
            self.event_handler(message)
