"""Agent-facing run lifecycle service."""

from __future__ import annotations

import threading

from ..run_log import make_run_id, preview_text
from ..session import SessionManager
from .cancel import RunCancelled
from .events import RuntimeEventEmitter, RuntimeEventHandler
from .turn_engine import TurnEngine, TurnResult


class SessionBusyError(RuntimeError):
    """Raised when a session already has an active run."""


__all__ = [
    "AgentSession",
    "RunCancelled",
    "SessionBusyError",
]


class AgentSession:
    """Run lifecycle facade shared by CLI, server, and future frontends."""

    def __init__(
        self,
        *,
        turn_engine: TurnEngine,
        session_manager: SessionManager,
    ) -> None:
        self.turn_engine = turn_engine
        self.session_manager = session_manager
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    def prompt(
        self,
        session_id: str | None,
        user_input: str,
        *,
        run_id: str | None = None,
        mode: str = "default",
        event_handler: RuntimeEventHandler | None = None,
    ) -> TurnResult:
        session = self.session_manager.resolve_session(session_id)
        session_lock = self._session_lock(session.session_id)
        if not session_lock.acquire(blocking=False):
            raise SessionBusyError(f"会话 {session.session_id} 已有运行中的 turn。")

        run_id = run_id or make_run_id()
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._cancel_events[run_id] = cancel_event

        emitter = RuntimeEventEmitter(
            run_id=run_id,
            session_id=session.session_id,
            handler=event_handler,
        )
        try:
            emitter.emit(
                "run.started",
                {
                    "session_id": session.session_id,
                    "input_preview": preview_text(user_input, 120),
                },
            )
            result = self.turn_engine.handle_turn(
                session,
                user_input,
                run_id=run_id,
                event_emitter=emitter,
                cancel_event=cancel_event,
                mode=mode,
            )
            emitter.emit(
                "run.completed",
                {
                    "session_id": session.session_id,
                    "reply": result.reply,
                    "did_compact": result.did_compact,
                    "compact_message": result.compact_message,
                },
            )
            return result
        except RunCancelled:
            emitter.emit("run.cancelled", {"session_id": session.session_id})
            raise
        except Exception as exc:
            emitter.emit(
                "run.failed",
                {
                    "session_id": session.session_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        finally:
            session_lock.release()
            with self._cancel_lock:
                self._cancel_events.pop(run_id, None)

    def abort(self, run_id: str) -> bool:
        """Signal the run to stop at the next checkpoint."""
        with self._cancel_lock:
            event = self._cancel_events.get(run_id)
        if event is None:
            return False
        event.set()
        return True

    def is_busy(self, session_id: str | None = None) -> bool:
        if session_id is None:
            with self._locks_lock:
                return any(lock.locked() for lock in self._locks.values())

        resolved = session_id.strip()
        if resolved == "current":
            current = self.session_manager.get_current_session_id()
            if current is None:
                return False
            resolved = current
        if not resolved:
            return False

        with self._locks_lock:
            lock = self._locks.get(resolved)
        return bool(lock and lock.locked())

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_lock:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock
