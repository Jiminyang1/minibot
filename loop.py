"""Turn orchestration for MiniBot.

AgentLoop sits between the CLI and the Agent Core, managing
session state, history windowing, compaction, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .compaction import make_summarizer, maybe_compact
from .session import MessageEvent, Session, SessionManager

if TYPE_CHECKING:
    from .agent import Agent
    from .config import Config
    from .llm import LLMClient


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one handled user turn."""

    reply: str
    did_compact: bool
    compact_message: str | None = None


class AgentLoop:
    """Coordinate session state, compaction, and agent execution."""

    def __init__(
        self,
        agent: Agent,
        llm: LLMClient,
        manager: SessionManager,
        config: Config,
    ) -> None:
        self.agent = agent
        self.manager = manager
        self.config = config
        self._summarizer = make_summarizer(llm)

    def handle_turn(self, session: Session, user_input: str) -> TurnResult:
        did_compact, compact_message = self._maybe_compact(session)

        history = session.history_for_model(self.config.max_history_turns)
        session.add_message(MessageEvent.create(role="user", content=user_input))
        self.manager.save(session)

        reply, turn_events = self.agent.run(history, user_input)
        for event in turn_events:
            session.add_message(event)
        self.manager.save(session)

        return TurnResult(
            reply=reply,
            did_compact=did_compact,
            compact_message=compact_message if did_compact else None,
        )

    def compact_session(self, session: Session) -> tuple[bool, str]:
        return self._maybe_compact(session)

    def _maybe_compact(self, session: Session) -> tuple[bool, str]:
        return maybe_compact(
            session,
            self.manager,
            self._summarizer,
            token_threshold=self.config.compact_token_threshold,
            keep_recent=self.config.compact_keep_recent,
        )
