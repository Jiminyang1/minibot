"""Run lifecycle coordination for CLI, SSE, and future SDK adapters."""

from __future__ import annotations

from dataclasses import dataclass
import threading

from ..run_log import make_run_id, preview_text
from ..session import Session, SessionManager
from .agent_runner import ApprovalRequest
from .events import RuntimeEventEmitter, RuntimeEventHandler
from .turn_engine import TurnEngine, TurnResult


class SessionBusyError(RuntimeError):
    """Raised when a session already has an active run."""


class SessionNotFoundError(RuntimeError):
    """Raised when a requested session id does not exist."""


@dataclass
class _PendingApproval:
    event: threading.Event
    approved: bool | None = None


class ApprovalBroker:
    """In-memory rendezvous for web approval requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], _PendingApproval] = {}
        self._early_decisions: dict[tuple[str, str], bool] = {}

    def wait(self, request: ApprovalRequest) -> bool:
        key = (request.run_id, request.approval_id)
        with self._lock:
            if key in self._early_decisions:
                return self._early_decisions.pop(key)
            pending = self._pending.setdefault(
                key,
                _PendingApproval(event=threading.Event()),
            )

        pending.event.wait()
        with self._lock:
            approved = bool(pending.approved)
            self._pending.pop(key, None)
            return approved

    def resolve(self, run_id: str, approval_id: str, approved: bool) -> bool:
        key = (run_id, approval_id)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                self._early_decisions[key] = approved
                return False
            pending.approved = approved
            pending.event.set()
            return True


class RunController:
    """Create one run-scoped event emitter around a TurnEngine call."""

    def __init__(
        self,
        *,
        turn_engine: TurnEngine,
        manager: SessionManager,
        approval_broker: ApprovalBroker | None = None,
    ) -> None:
        self.turn_engine = turn_engine
        self.manager = manager
        self.approval_broker = approval_broker
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def run_turn(
        self,
        *,
        session_id: str | None,
        user_input: str,
        event_handler: RuntimeEventHandler | None,
    ) -> TurnResult:
        session = self._resolve_session(session_id)
        session_lock = self._session_lock(session.session_id)
        if not session_lock.acquire(blocking=False):
            raise SessionBusyError(f"会话 {session.session_id} 已有运行中的 turn。")

        run_id = make_run_id()
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

    def resolve_approval(
        self,
        *,
        run_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        if self.approval_broker is None:
            raise RuntimeError("当前 runtime 未配置 approval broker。")
        return self.approval_broker.resolve(run_id, approval_id, approved)

    def _resolve_session(self, requested: str | None) -> Session:
        session_id = None if requested is None else requested.strip()
        if not session_id:
            session = self.manager.create_session()
            self.manager.set_current_session(session.session_id)
            return session
        if session_id == "current":
            session = self.manager.load_current_session()
            if session is not None:
                return session
            session = self.manager.create_session()
            self.manager.set_current_session(session.session_id)
            return session

        session = self.manager.load(session_id)
        if session is None:
            raise SessionNotFoundError(f"未找到会话: {session_id}")
        return session

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_lock:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock
