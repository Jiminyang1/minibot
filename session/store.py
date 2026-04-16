"""Session persistence for MiniBot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import MessageEvent, Session


class SessionManager:
    """Save and load sessions as `.jsonl` files under `.minibot/sessions/`."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.sessions_dir = (workspace or Path.cwd()).resolve() / ".minibot" / "sessions"
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

    def list_sessions(self) -> list[Session]:
        sessions: list[Session] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            session = self.load(path.stem)
            if session:
                sessions.append(session)
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)
