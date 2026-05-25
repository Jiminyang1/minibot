"""Turn orchestration for MiniBot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING

from .agent_loop import PartialRunError, RunSpec
from .context_manager import WorkingContext
from .events import RuntimeEventEmitter, RuntimeEventHandler
from .hooks import HookContext, RuntimeHookManager
from .turn_recorder import TurnRecorder
from ..run_log import make_run_id, utc_now
from ..session import MessageEvent, Session, SessionManager

if TYPE_CHECKING:
    from .agent_loop import AgentLoop
    from ..config import Config
    from .context_manager import ContextWindowManager
    from ..run_log import RunLogStore


@dataclass(frozen=True)
class TurnResult:
    """Outcome of one handled user turn."""

    reply: str
    did_compact: bool
    compact_message: str | None = None


class TurnEngine:
    """Coordinate one full user turn: context prep, loop, persistence."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        manager: SessionManager,
        config: Config,
        *,
        context_manager: ContextWindowManager,
        hook_manager: RuntimeHookManager | None = None,
        event_handler: RuntimeEventHandler | None = None,
        run_log_store: RunLogStore | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.config = config
        self.context_manager = context_manager
        self.hook_manager = hook_manager or RuntimeHookManager()
        self.event_handler = event_handler
        self.recorder = TurnRecorder(
            manager=manager,
            run_log_store=run_log_store,
            tool_registry=agent_loop.tool_registry,
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

        prepared_for_log: WorkingContext | None = None
        reply: str | None = None
        turn_messages: list[MessageEvent] = []
        usage = None
        did_compact = False
        compact_message: str | None = None
        final_result: TurnResult | None = None
        hook_context = HookContext(
            run_id=run_id,
            session_id=session.session_id,
            workspace=self.workspace,
            mode=mode,
            emitter=emitter,
            cancel_event=cancel_event,
        )

        def _prepare_next_turn(
            observed_input_tokens: int | None,
        ) -> WorkingContext:
            nonlocal compact_message, did_compact, prepared_for_log
            self.recorder.flush_pending_compaction(session)
            working_context = self.context_manager.build_context(
                session=session,
                observed_input_tokens=observed_input_tokens,
                cancel_event=cancel_event,
            )
            if working_context.did_compact:
                self.recorder.persist_pending_compaction(session)
            working_context = self.hook_manager.after_context(
                hook_context,
                working_context,
            )
            if working_context.did_compact:
                did_compact = True
                compact_message = working_context.compact_message
            prepared_for_log = WorkingContext(
                messages=working_context.messages,
                tool_definitions=working_context.tool_definitions,
                did_compact=did_compact,
                compact_message=compact_message,
            )
            return working_context

        def _on_message(message: MessageEvent) -> MessageEvent:
            nonlocal final_result, reply
            persisted_message = message
            if message.role == "assistant" and not message.tool_calls:
                result = TurnResult(
                    reply=message.content,
                    did_compact=did_compact,
                    compact_message=compact_message,
                )
                result = self.hook_manager.after_turn(hook_context, result)
                final_result = result
                reply = result.reply
                if result.reply != message.content:
                    persisted_message = _message_with_content(message, result.reply)

            persisted_message = self.recorder.on_message(session, persisted_message)
            turn_messages.append(persisted_message)
            return persisted_message

        try:
            self._emit_current_context_usage(session, emitter)
            run_spec = RunSpec(
                session_id=session.session_id,
                model=self.config.model,
                user_input=user_input,
                prepare_next_turn=_prepare_next_turn,
                on_message=_on_message,
                max_iterations=self.config.max_iterations,
                run_id=run_id,
                event_emitter=emitter,
                cancel_event=cancel_event,
                mode=mode,
                workspace=self.workspace,
            )
            outcome = self.agent_loop.run(run_spec)
            self.recorder.flush_pending_compaction(session)
            reply = outcome.reply
            usage = outcome.usage
            result = final_result or TurnResult(
                reply=reply,
                did_compact=did_compact,
                compact_message=compact_message,
            )
            self.recorder.record_run(
                run_id=run_id,
                session=session,
                turn_index=turn_index,
                timestamp=timestamp,
                status="success",
                model=self.config.model,
                user_input=user_input,
                duration_ms=int((time.perf_counter() - started) * 1000),
                prepared=prepared_for_log,
                messages=turn_messages,
                usage=usage,
                reply=result.reply,
            )
            return result
        except PartialRunError as exc:
            reply = exc.reply
            usage = exc.usage
            self.recorder.flush_pending_compaction(session)
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
                prepared=prepared_for_log,
                messages=turn_messages,
                usage=usage,
                reply=reply,
                error=exc.cause,
            )
            raise exc.cause from exc
        except Exception as exc:
            self.recorder.flush_pending_compaction(session)
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
                prepared=prepared_for_log,
                messages=turn_messages,
                usage=usage,
                reply=reply,
                error=exc,
            )
            raise

    def compact_session(self, session: Session) -> tuple[bool, str]:
        did_compact, message = self.context_manager.compact_session(session=session)
        if did_compact:
            self.recorder.persist_pending_compaction(session)
        return did_compact, message

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


def _message_with_content(message: MessageEvent, content: str) -> MessageEvent:
    return MessageEvent(
        id=message.id,
        role=message.role,
        content=content,
        created_at=message.created_at,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
        name=message.name,
        reasoning_content=message.reasoning_content,
    )
