"""Session persistence for MiniBot."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Iterator
import uuid

from .models import Session, SessionEntry


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class SessionNotFoundError(RuntimeError):
    """Raised when a requested session id does not exist."""


def validate_session_id(session_id: str) -> str:
    """Return a safe session id or raise ValueError."""
    normalized = session_id.strip()
    if not normalized:
        raise ValueError("session_id 不能为空。")
    if not _SESSION_ID_RE.fullmatch(normalized):
        raise ValueError(f"session_id 无效: {session_id!r}")
    return normalized


class SessionManager:
    """Persist sessions under ``<state_home>/sessions/<session_id>/``."""

    def __init__(
        self,
        state_home: Path | None = None,
        *,
        default_workspace: Path | None = None,
    ) -> None:
        if state_home is None:
            from ..config import resolve_state_home

            state_home = resolve_state_home()
        self.state_dir = state_home.resolve()
        self.default_workspace = (
            None if default_workspace is None else str(default_workspace.resolve())
        )
        self.sessions_dir = self.state_dir / "sessions"
        self.locks_dir = self.state_dir / "locks"
        self.current_session_path = self.state_dir / "current_session"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self._safe_session_dir(validate_session_id(session_id))

    def _meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "meta.json"

    def _messages_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "messages.jsonl"

    def create_session(
        self,
        session_id: str | None = None,
        *,
        title: str | None = None,
    ) -> Session:
        resolved_id = validate_session_id(
            session_id or ("s_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        if session_id is None:
            suffix = 1
            while True:
                with self._locked_session(resolved_id):
                    if not self._exists(resolved_id):
                        session = Session(
                            resolved_id,
                            title=title or "新会话",
                            workspace=self.default_workspace,
                        )
                        self._ensure_session_dir(resolved_id)
                        self._write_meta(session)
                        self._write_entries(
                            self._messages_path(session.session_id),
                            session.entries,
                        )
                        return session
                resolved_id = validate_session_id(
                    f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
                )
                suffix += 1

        with self._locked_session(resolved_id):
            if self._exists(resolved_id):
                raise FileExistsError(f"会话 {resolved_id} 已存在")
            session = Session(
                resolved_id,
                title=title or "新会话",
                workspace=self.default_workspace,
            )
            self._ensure_session_dir(resolved_id)
            self._write_meta(session)
            self._write_entries(
                self._messages_path(session.session_id),
                session.entries,
            )
            return session

    def startup_session(self) -> tuple[Session, bool]:
        """Resume current/latest, or create a new current session."""
        current = self.load_current_session()
        if current is not None:
            return current, True
        latest = self.latest_session(prefer_non_empty=True)
        if latest is not None:
            self.set_current_session(latest.session_id)
            return latest, True
        session = self.create_current_session()
        return session, False

    def resolve_session(self, session_id: str | None) -> Session:
        """Resolve a run target, creating a new current session for empty/current."""
        requested = None if session_id is None else session_id.strip()
        if not requested:
            return self.create_current_session()
        if requested == "current":
            session = self.load_current_session()
            if session is not None:
                return session
            return self.create_current_session()

        session = self.load(requested)
        if session is None:
            raise SessionNotFoundError(f"未找到会话: {requested}")
        return session

    def create_current_session(
        self,
        session_id: str | None = None,
        *,
        title: str | None = None,
    ) -> Session:
        session = self.create_session(session_id=session_id, title=title)
        self.set_current_session(session.session_id)
        return session

    def resume_session(self, session_id: str) -> Session | None:
        session = self.load(session_id)
        if session is not None:
            self.set_current_session(session.session_id)
        return session

    def load(self, session_id: str) -> Session | None:
        try:
            session_id = validate_session_id(session_id)
        except ValueError:
            return None
        if self._meta_path(session_id).exists():
            return self._load_native(session_id)
        return None

    def save(self, session: Session) -> None:
        """Rewrite one whole session snapshot in the native layout."""
        session.session_id = validate_session_id(session.session_id)
        with self._locked_session(session.session_id):
            self._ensure_session_dir(session.session_id)
            self._write_meta(session)
            self._write_entries(self._messages_path(session.session_id), session.entries)

    def append_entries(self, session_id: str, entries: list[SessionEntry]) -> None:
        """Append raw session entries to the native log."""
        if not entries:
            return

        session_id = validate_session_id(session_id)
        with self._locked_session(session_id):
            if not self._meta_path(session_id).exists():
                raise FileNotFoundError(f"未找到会话 {session_id}")
            path = self._messages_path(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def update_metadata(self, session: Session) -> None:
        """Rewrite only the metadata record for a native session."""
        session.session_id = validate_session_id(session.session_id)
        with self._locked_session(session.session_id):
            if not self._meta_path(session.session_id).exists():
                raise FileNotFoundError(f"未找到会话 {session.session_id}")
            self._write_meta(session)

    def get_current_session_id(self) -> str | None:
        if not self.current_session_path.exists():
            return None
        session_id = self.current_session_path.read_text(encoding="utf-8").strip()
        if not session_id:
            return None
        try:
            return validate_session_id(session_id)
        except ValueError:
            return None

    def set_current_session(self, session_id: str) -> None:
        session_id = validate_session_id(session_id)
        self._atomic_write_text(self.current_session_path, session_id + "\n")

    def clear_current_session(self) -> None:
        if self.current_session_path.exists():
            self.current_session_path.unlink()

    def load_current_session(self) -> Session | None:
        session_id = self.get_current_session_id()
        if session_id is None:
            if self.current_session_path.exists():
                self.clear_current_session()
            return None
        session = self.load(session_id)
        if session is None:
            self.clear_current_session()
            return None
        return session

    def delete_session(self, session_id: str) -> bool:
        try:
            session_id = validate_session_id(session_id)
        except ValueError:
            return False

        removed = False
        with self._locked_session(session_id):
            session_dir = self._session_dir(session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir)
                removed = True

        if self.get_current_session_id() == session_id:
            self.clear_current_session()
        if removed:
            _drop_thread_lock(self.locks_dir / f"{session_id}.lock")
        return removed

    def latest_session(self, *, prefer_non_empty: bool = True) -> Session | None:
        metas = self._list_metas()
        if not metas:
            return None
        if not prefer_non_empty:
            return self.load(str(metas[0].get("session_id", "")))
        for meta in metas:
            if meta.get("message_count", 0) > 0:
                return self.load(str(meta["session_id"]))
        return self.load(str(metas[0].get("session_id", "")))

    def list_sessions(self) -> list[Session]:
        return [
            Session(
                session_id=str(m["session_id"]),
                title=str(m.get("title", "新会话")),
                created_at=str(m["created_at"]) if m.get("created_at") else None,
                updated_at=str(m["updated_at"]) if m.get("updated_at") else None,
                message_count=int(m.get("message_count", 0)),
                workspace=str(m["workspace"]) if m.get("workspace") else None,
            )
            for m in self._list_metas()
        ]

    def _exists(self, session_id: str) -> bool:
        return self._session_dir(session_id).exists()

    def _ensure_session_dir(self, session_id: str) -> Path:
        path = self._session_dir(session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_native(self, session_id: str) -> Session:
        meta = self._read_meta(session_id) or {}
        messages_path = self._messages_path(session_id)
        entries: list[SessionEntry] = []
        if messages_path.exists():
            for line in messages_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    entries.append(SessionEntry.from_dict(record))

        return Session(
            session_id=str(meta.get("session_id", session_id)),
            title=str(meta.get("title", "新会话")),
            created_at=str(meta["created_at"]) if meta.get("created_at") else None,
            updated_at=str(meta["updated_at"]) if meta.get("updated_at") else None,
            entries=entries,
            message_count=int(meta.get("message_count", len(entries))),
            workspace=str(meta["workspace"]) if meta.get("workspace") else None,
        )

    def _write_meta(self, session: Session) -> None:
        self._ensure_session_dir(session.session_id)
        self._atomic_write_text(
            self._meta_path(session.session_id),
            json.dumps(
                {
                    "session_id": session.session_id,
                    "title": session.title,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "workspace": session.workspace,
                    "message_count": len(session.messages),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def _write_entries(self, path: Path, entries: list[SessionEntry]) -> None:
        lines = [json.dumps(entry.to_dict(), ensure_ascii=False) for entry in entries]
        self._atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))

    def _read_meta(self, session_id: str) -> dict[str, object] | None:
        path = self._meta_path(session_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None

    def _list_metas(self) -> list[dict[str, object]]:
        metas_by_id: dict[str, dict[str, object]] = {}

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            try:
                validate_session_id(session_dir.name)
            except ValueError:
                continue
            meta = self._read_meta(session_dir.name)
            if meta:
                metas_by_id[session_dir.name] = meta

        return sorted(
            metas_by_id.values(),
            key=lambda m: str(m.get("updated_at", "")),
            reverse=True,
        )

    def _safe_session_dir(self, session_id: str) -> Path:
        path = (self.sessions_dir / session_id).resolve()
        if path != self.sessions_dir and self.sessions_dir not in path.parents:
            raise ValueError(f"session_id 无效: {session_id!r}")
        return path

    @contextmanager
    def _locked_session(self, session_id: str) -> Iterator[None]:
        session_id = validate_session_id(session_id)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.locks_dir / f"{session_id}.lock"
        with _thread_lock_for(lock_path):
            with lock_path.open("a", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)


def _thread_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def _drop_thread_lock(path: Path) -> None:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        _THREAD_LOCKS.pop(key, None)
