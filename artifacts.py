"""Shared artifact reference types.

These types are neutral contracts used by both session persistence and
tool results. Keeping them out of ``tools/`` avoids a reverse
dependency from ``session`` back into the tool layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TypeAlias
import uuid

ArtifactKind: TypeAlias = Literal["text", "json", "file", "diff"]


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` as UTF-8 bytes.

    Central helper so that all hash computations in the project (file reads,
    artifact persistence, write-file optimistic lock) share the exact same
    normalization. Anything that decodes to the same string hashes the same.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """A stable reference to a stored tool artifact."""

    id: str
    kind: ArtifactKind
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactPage:
    """A single character-based artifact page."""

    ref: ArtifactRef
    content: str
    offset: int
    limit: int
    total_chars: int
    next_offset: int | None
    has_more: bool
    file_sha256: str | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ArtifactStore:
    """Persist large tool outputs under ``.minibot/sessions/<session_id>/artifacts``."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.state_dir = (workspace or Path.cwd()).resolve() / ".minibot"
        self.sessions_dir = self.state_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def put_text(
        self,
        session_id: str,
        content: str,
        *,
        kind: ArtifactKind = "text",
        name: str | None = None,
    ) -> ArtifactRef:
        artifact_id = "a_" + uuid.uuid4().hex[:12]
        payload = {
            "id": artifact_id,
            "kind": kind,
            "name": name,
            "created_at": _utc_now(),
            "content": content,
            "file_sha256": sha256_text(content),
        }
        self._artifact_path(session_id, artifact_id).write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ArtifactRef(id=artifact_id, kind=kind, name=name)

    def read_page(
        self,
        session_id: str,
        artifact_id: str,
        *,
        offset: int,
        limit: int,
    ) -> ArtifactPage:
        payload = self._load_artifact(session_id, artifact_id)
        content = str(payload["content"])
        total_chars = len(content)
        start = min(offset, total_chars)
        end = min(start + limit, total_chars)
        next_offset = end if end < total_chars else None
        stored_sha = payload.get("file_sha256")
        return ArtifactPage(
            ref=ArtifactRef(
                id=str(payload["id"]),
                kind=str(payload["kind"]),
                name=payload.get("name"),
            ),
            content=content[start:end],
            offset=start,
            limit=limit,
            total_chars=total_chars,
            next_offset=next_offset,
            has_more=next_offset is not None,
            file_sha256=str(stored_sha) if stored_sha else None,
        )

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _ensure_session_dir(self, session_id: str) -> Path:
        path = self._session_dir(session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifacts_dir(self, session_id: str) -> Path:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空。")
        path = self._ensure_session_dir(session_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path(self, session_id: str, artifact_id: str) -> Path:
        return self._artifacts_dir(session_id) / f"{artifact_id}.json"

    def _load_artifact(self, session_id: str, artifact_id: str) -> dict[str, object]:
        path = self._artifact_path(session_id, artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"未找到 artifact {artifact_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"artifact {artifact_id} 格式无效")
        return payload
