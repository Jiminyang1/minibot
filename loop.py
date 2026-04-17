"""Turn orchestration for MiniBot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .context import build_messages
from .compaction import estimate_visible_context_tokens, maybe_compact
from .prompts import MEMORY_INSTRUCTIONS
from .session import MessageEvent, Session, SessionManager

if TYPE_CHECKING:
    from .agent import AgentRunner, AgentSpec
    from .config import Config
    from .memory import MemoryStore


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one handled user turn."""

    reply: str
    did_compact: bool
    compact_message: str | None = None


class TurnEngine:
    """Coordinate one full user turn: session, compaction, context, runner, persistence."""

    def __init__(
        self,
        spec: AgentSpec,
        runner: AgentRunner,
        manager: SessionManager,
        config: Config,
        *,
        summarizer: Callable[[list[dict[str, object]]], str],
        memory_store: MemoryStore | None = None,
        event_handler: Callable[[str], None] | None = None,
    ) -> None:
        self.spec = spec
        self.runner = runner
        self.manager = manager
        self.config = config
        self.memory_store = memory_store
        self.event_handler = event_handler
        self._summarizer = summarizer

    def _effective_system_prompt(self) -> str:
        """Base system prompt plus memory instructions and current memories.

        Rendered fresh every turn so tool-driven updates to memory show up
        in the very next request.
        """
        if self.memory_store is None:
            return self.spec.system_prompt
        memory_block = self.memory_store.render_for_prompt()
        parts = [self.spec.system_prompt, MEMORY_INSTRUCTIONS]
        if memory_block:
            parts.append(memory_block)
        return "\n\n".join(parts)

    def handle_turn(self, session: Session, user_input: str) -> TurnResult:
        system_prompt = self._effective_system_prompt()
        self._emit_current_context_usage(session, system_prompt)
        did_compact, compact_message = self._maybe_compact(
            session, user_input, system_prompt=system_prompt
        )

        history = session.history_for_model(self.config.max_history_turns)
        session.add_message(MessageEvent.create(role="user", content=user_input))
        self.manager.save(session)

        messages = build_messages(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
        )
        reply, turn_events = self.runner.run(messages)
        for event in turn_events:
            session.add_message(event)
        self.manager.save(session)

        return TurnResult(
            reply=reply,
            did_compact=did_compact,
            compact_message=compact_message if did_compact else None,
        )

    def compact_session(self, session: Session) -> tuple[bool, str]:
        return self._maybe_compact(
            session, None, system_prompt=self._effective_system_prompt()
        )

    def _maybe_compact(
        self,
        session: Session,
        user_input: str | None,
        *,
        system_prompt: str,
    ) -> tuple[bool, str]:
        return maybe_compact(
            session,
            self.manager,
            self._summarizer,
            system_prompt=system_prompt,
            max_history_turns=self.config.max_history_turns,
            user_input=user_input,
            tools=self.spec.tool_definitions,
            token_threshold=self.config.compact_token_threshold,
            reserved_completion_tokens=self.config.reserved_completion_tokens,
            keep_recent=self.config.compact_keep_recent,
        )

    def _emit_current_context_usage(self, session: Session, system_prompt: str) -> None:
        current_tokens = estimate_visible_context_tokens(
            session=session,
            system_prompt=system_prompt,
            max_history_turns=self.config.max_history_turns,
            tools=self.spec.tool_definitions,
        )
        self._emit(
            "当前上下文占用(不含本次输入): "
            f"{current_tokens}/{self.config.compact_token_threshold} tokens"
        )

    def _emit(self, message: str) -> None:
        if self.event_handler:
            self.event_handler(message)
