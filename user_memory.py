"""Global user-memory store for MiniBot.

This module manages a single cross-session memory file under the user's
home directory. It stores short, stable facts about the user
(preferences, identity, routines) and exposes only structured data
operations; prompt rendering lives in :mod:`minibot.runtime.context_builder`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MemoryItem:
    id: str
    content: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "content": self.content, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict) -> MemoryItem:
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            created_at=str(data.get("created_at", "")),
        )


class MemoryFileDamaged(OSError):
    """The memory file exists but cannot be parsed.

    Writes stop here: saving over it would replace every stored fact with
    the one being added.
    """


class UserMemoryStore:
    """Load / save / mutate global user memory backed by a JSON file.

    The CLI, the web server and the daemon can all hold a store on the same
    file, so every change is one read-modify-write under a file lock, and the
    file is replaced whole: a crash leaves the old file or the new one.
    """

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from .config import resolve_state_home

            root = resolve_state_home()
        self.state_dir = root.resolve()
        self.memory_path = self.state_dir / "user_memory.json"
        self._lock_path = self.state_dir / "locks" / "user_memory.lock"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[MemoryItem]:
        """Current facts; empty when the file is missing or unreadable.

        Every request reads this, so a damaged file must not fail the turn.
        ``read_error`` says whether "empty" really means empty.
        """
        try:
            items, _ = self._read()
        except OSError:
            return []
        return items

    def read_error(self) -> str | None:
        """Why the file cannot be read, or None when it can."""
        try:
            self._read()
        except OSError as exc:
            return str(exc)
        return None

    def add(self, content: str) -> MemoryItem:
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空。")
        with self._locked():
            items, next_index = self._read()
            item = MemoryItem(
                id=f"mem_{next_index}",
                content=content,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            self._save([*items, item], next_index + 1)
        return item

    def delete(self, memory_id: str) -> bool:
        with self._locked():
            items, next_index = self._read()
            kept = [m for m in items if m.id != memory_id]
            if len(kept) == len(items):
                return False
            self._save(kept, next_index)
        return True

    def clear(self) -> int:
        with self._locked():
            items, next_index = self._read()
            self._save([], next_index)
        return len(items)

    def _read(self) -> tuple[list[MemoryItem], int]:
        """Return the stored facts and the next unused index.

        The index never goes back: a forgotten id must not come back pointing
        at a different, newer fact (the model may still hold a stale
        reference to it). It is read from the file each time, because another
        process may have advanced it.
        """
        try:
            raw = self.memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], 1
        if not raw.strip():
            # What the old truncate-then-write save left behind after a crash.
            # It holds nothing to protect, so it is an empty store.
            return [], 1
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise self._damaged(f"不是合法的 JSON（{exc.msg}，第 {exc.lineno} 行）") from exc
        if not isinstance(data, dict) or not isinstance(data.get("memories", []), list):
            raise self._damaged("结构不是 {\"memories\": [...]}")
        items = [
            MemoryItem.from_dict(entry)
            for entry in data.get("memories", [])
            if isinstance(entry, dict) and "id" in entry and "content" in entry
        ]
        stored = data.get("next_index")
        next_index = self._highest_item_index(items) + 1
        if isinstance(stored, int) and not isinstance(stored, bool):
            next_index = max(next_index, stored)
        return items, next_index

    def _damaged(self, reason: str) -> MemoryFileDamaged:
        return MemoryFileDamaged(
            f"长期记忆文件 {self.memory_path} 无法读取：{reason}。"
            "为避免覆盖已有记忆，已停止写入；请修复或移走该文件。"
        )

    def _save(self, items: list[MemoryItem], next_index: int) -> None:
        payload = {
            "memories": [m.to_dict() for m in items],
            "next_index": next_index,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        temp_path = self.memory_path.with_name(
            f".{self.memory_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.memory_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # flock belongs to the open file, not the process, so two stores in
        # one process exclude each other the same way two processes do.
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _highest_item_index(items: list[MemoryItem]) -> int:
        highest = 0
        for item in items:
            if item.id.startswith("mem_") and item.id[4:].isdigit():
                highest = max(highest, int(item.id[4:]))
        return highest
