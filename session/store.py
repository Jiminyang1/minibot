"""Session persistence for MiniBot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import MessageEvent, Session


class SessionManager:
    """Save and load sessions as `.jsonl` files under `.minibot/sessions/`."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.state_dir = (workspace or Path.cwd()).resolve() / ".minibot"
        self.sessions_dir = self.state_dir / "sessions"
        self.current_session_path = self.state_dir / "current_session"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def create_session(self, session_id: str | None = None, *, title: str | None = None) -> Session:
        resolved_id = session_id or ("s_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        if session_id is None:
            suffix = 1
            while self._path(resolved_id).exists():
                resolved_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
                suffix += 1
        session = Session(resolved_id, title=title or "新会话")
        self.save(session)
        return session

    def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None

        meta: dict[str, object] = {}
        messages: list[MessageEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "meta":
                meta = record
            elif record.get("type") == "message":
                messages.append(MessageEvent.from_dict(record))

        return Session(
            session_id=str(meta.get("session_id", session_id)),
            title=str(meta.get("title", "新会话")),
            created_at=str(meta["created_at"]) if meta.get("created_at") else None,
            updated_at=str(meta["updated_at"]) if meta.get("updated_at") else None,
            messages=messages,
        )

    def save(self, session: Session) -> None:
        meta_line = json.dumps(
            {
                "type": "meta",
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages),
            },
            ensure_ascii=False,
        )
        message_lines = [
            json.dumps({"type": "message", **message.to_dict()}, ensure_ascii=False)
            for message in session.messages
        ]
        self._path(session.session_id).write_text(
            "\n".join([meta_line, *message_lines]) + "\n",
            encoding="utf-8",
        )

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
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        if self.get_current_session_id() == session_id:
            self.clear_current_session()
        return True

    def _load_meta(self, session_id: str) -> dict[str, object] | None:
        """Read only the first (meta) line of a session file."""
        path = self._path(session_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            return None
        record = json.loads(first_line)
        return record if record.get("type") == "meta" else None

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

    def _list_metas(self) -> list[dict[str, object]]:
        metas: list[dict[str, object]] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            meta = self._load_meta(path.stem)
            if meta:
                metas.append(meta)
        return sorted(metas, key=lambda m: str(m.get("updated_at", "")), reverse=True)

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
