"""Session persistence for MiniBot."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from .models import MessageEvent, Session


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class SessionManager:
    """Persist sessions under ``.minibot/sessions/<session_id>/``."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.state_dir = (workspace or Path.cwd()).resolve() / ".minibot"
        self.sessions_dir = self.state_dir / "sessions"
        self.current_session_path = self.state_dir / "current_session"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

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
        resolved_id = session_id or ("s_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        if session_id is None:
            suffix = 1
            while self._exists(resolved_id):
                resolved_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
                suffix += 1
        session = Session(resolved_id, title=title or "新会话")
        self.save(session)
        return session

    def load(self, session_id: str) -> Session | None:
        if self._meta_path(session_id).exists():
            return self._load_native(session_id)
        return None

    def save(self, session: Session) -> None:
        """Rewrite one whole session snapshot in the native layout."""
        self._ensure_session_dir(session.session_id)
        self._write_meta(session)
        self._write_messages(self._messages_path(session.session_id), session.messages)

    def append_messages(self, session_id: str, messages: list[MessageEvent]) -> None:
        """Append new events to the native message log."""
        if not messages:
            return

        if not self._meta_path(session_id).exists():
            raise FileNotFoundError(f"未找到会话 {session_id}")

        path = self._messages_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for message in messages:
                line = json.dumps(
                    {"type": "message", **message.to_dict()},
                    ensure_ascii=False,
                )
                f.write(line + "\n")

    def update_metadata(self, session: Session) -> None:
        """Rewrite only the metadata record for a native session."""
        if not self._meta_path(session.session_id).exists():
            raise FileNotFoundError(f"未找到会话 {session.session_id}")
        self._write_meta(session)

    def get_current_session_id(self) -> str | None:
        if not self.current_session_path.exists():
            return None
        session_id = self.current_session_path.read_text(encoding="utf-8").strip()
        return session_id or None

    def set_current_session(self, session_id: str) -> None:
        self.current_session_path.write_text(session_id + "\n", encoding="utf-8")

    def clear_current_session(self) -> None:
        if self.current_session_path.exists():
            self.current_session_path.unlink()

    def load_current_session(self) -> Session | None:
        session_id = self.get_current_session_id()
        if session_id is None:
            return None
        session = self.load(session_id)
        if session is None:
            self.clear_current_session()
            return None
        return session

    def delete_session(self, session_id: str) -> bool:
        removed = False

        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            removed = True

        if self.get_current_session_id() == session_id:
            self.clear_current_session()
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
                messages=[],
                message_count=int(m.get("message_count", 0)),
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
        messages: list[MessageEvent] = []
        if messages_path.exists():
            for line in messages_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "message":
                    messages.append(MessageEvent.from_dict(record))

        return Session(
            session_id=str(meta.get("session_id", session_id)),
            title=str(meta.get("title", "新会话")),
            created_at=str(meta["created_at"]) if meta.get("created_at") else None,
            updated_at=str(meta["updated_at"]) if meta.get("updated_at") else None,
            messages=messages,
            message_count=int(meta.get("message_count", len(messages))),
        )

    def _write_meta(self, session: Session) -> None:
        self._ensure_session_dir(session.session_id)
        self._meta_path(session.session_id).write_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "title": session.title,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "message_count": len(session.messages),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    def _write_messages(self, path: Path, messages: list[MessageEvent]) -> None:
        lines = [
            json.dumps({"type": "message", **message.to_dict()}, ensure_ascii=False)
            for message in messages
        ]
        path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

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
            meta = self._read_meta(session_dir.name)
            if meta:
                metas_by_id[session_dir.name] = meta

        return sorted(
            metas_by_id.values(),
            key=lambda m: str(m.get("updated_at", "")),
            reverse=True,
        )
