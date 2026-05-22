"""Turn orchestration for MiniBot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING

from .agent_runner import PartialRunError, RunSpec
from .events import RuntimeEventEmitter, RuntimeEventHandler
from .hooks import HookContext, RuntimeHookManager
from .messages import AgentMessage, replace_final_assistant_reply
from .turn_recorder import TurnRecorder
from ..run_log import make_run_id, utc_now
from ..session import Session, SessionManager

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
        hook_manager: RuntimeHookManager | None = None,
        event_handler: RuntimeEventHandler | None = None,
        run_log_store: RunLogStore | None = None,
        recorder: TurnRecorder | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.runner = runner
        self.manager = manager
        self.config = config
        self.context_manager = context_manager
        self.hook_manager = hook_manager or RuntimeHookManager()
        self.event_handler = event_handler
        self.recorder = recorder or TurnRecorder(
            manager=manager,
            run_log_store=run_log_store,
            tool_registry=runner.tool_registry,
        )
        self.workspace = (workspace or manager.state_dir.parent).resolve()

    def handle_turn(
        self,
        session: Session,
        user_input: str,
        *,
        run_id: str | None = None,
        event_emitter: RuntimeEventEmitter | None = None,
        cancel_event: threading.Event | None = None,
        mode: str = "default",
    ) -> TurnResult:
        emitter = event_emitter or RuntimeEventEmitter(
            run_id=run_id or make_run_id(),
            session_id=session.session_id,
            handler=self.event_handler,
        )
        run_id = emitter.run_id
        timestamp = utc_now()
        started = time.perf_counter()
        turn_index = session.turn_count() + 1

        prepared = None
        reply: str | None = None
        turn_messages: list[AgentMessage] = []
        persisted_turn_messages = False
        usage = None
        hook_context = HookContext(
            run_id=run_id,
            session_id=session.session_id,
            workspace=self.workspace,
            mode=mode,
            emitter=emitter,
            cancel_event=cancel_event,
        )

        try:
            self._emit_current_context_usage(session, emitter)
            prepared = self.context_manager.prepare_for_turn(
                session=session,
                user_input=user_input,
            )
            prepared = self.hook_manager.after_context(hook_context, prepared)
            if prepared.did_compact:
                self.recorder.save_session(session)
                emitter.emit(
                    "context.compacted",
                    {
                        "message": prepared.compact_message,
                    },
                )
            self.recorder.persist_user_message(session, user_input)

            run_spec = RunSpec(
                session_id=session.session_id,
                model=self.config.model,
                messages=prepared.messages,
                tool_definitions=prepared.tool_definitions,
                max_iterations=self.config.max_iterations,
                run_id=run_id,
                event_emitter=emitter,
                cancel_event=cancel_event,
                mode=mode,
                workspace=self.workspace,
            )
            outcome = self.runner.run(run_spec)
            reply = outcome.reply
            turn_messages = outcome.messages
            usage = outcome.usage
            result = TurnResult(
                reply=reply,
                did_compact=prepared.did_compact,
                compact_message=prepared.compact_message,
            )
            result = self.hook_manager.after_turn(hook_context, result)
            if result.reply != reply:
                turn_messages = replace_final_assistant_reply(
                    turn_messages,
                    result.reply,
                )
                reply = result.reply
            self.recorder.persist_agent_messages(session, turn_messages)
            persisted_turn_messages = True
            self.recorder.record_run(
                run_id=run_id,
                session=session,
                turn_index=turn_index,
                timestamp=timestamp,
                status="success",
                model=self.config.model,
                user_input=user_input,
                duration_ms=int((time.perf_counter() - started) * 1000),
                prepared=prepared,
                messages=turn_messages,
                usage=usage,
                reply=result.reply,
            )
            return result
        except PartialRunError as exc:
            reply = exc.reply
            turn_messages = exc.messages
            usage = exc.usage
            self.recorder.persist_agent_messages(session, turn_messages)
            self.hook_manager.on_error(hook_context, exc.cause)
            self.recorder.record_run(
                run_id=run_id,
                session=session,
                turn_index=turn_index,
                timestamp=timestamp,
                status="failed",
                model=self.config.model,
                user_input=user_input,
                duration_ms=int((time.perf_counter() - started) * 1000),
                prepared=prepared,
                messages=turn_messages,
                usage=usage,
                reply=reply,
                error=exc.cause,
            )
            raise exc.cause from exc
        except Exception as exc:
            if turn_messages and not persisted_turn_messages:
                self.recorder.persist_agent_messages(session, turn_messages)
            self.hook_manager.on_error(hook_context, exc)
            self.recorder.record_run(
                run_id=run_id,
                session=session,
                turn_index=turn_index,
                timestamp=timestamp,
                status="failed",
                model=self.config.model,
                user_input=user_input,
                duration_ms=int((time.perf_counter() - started) * 1000),
                prepared=prepared,
                messages=turn_messages,
                usage=usage,
                reply=reply,
                error=exc,
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

    def _emit_current_context_usage(
        self,
        session: Session,
        emitter: RuntimeEventEmitter,
    ) -> None:
        current_tokens = self.context_manager.estimate_visible_tokens(session=session)
        budget = self.context_manager.effective_input_budget
        emitter.emit(
            "context.usage",
            {
                "current_tokens": current_tokens,
                "budget": budget,
            },
        )
