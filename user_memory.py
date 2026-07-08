"""Global user-memory store for MiniBot.

This module manages a single cross-session memory file under the user's
home directory. It stores short, stable facts about the user
(preferences, identity, routines) and exposes only structured data
operations; prompt rendering lives in :mod:`minibot.runtime.context_builder`.
"""

from __future__ import annotations

import json
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


class UserMemoryStore:
    """Load / save / mutate global user memory backed by a JSON file."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from .config import resolve_state_home

            root = resolve_state_home()
        self.state_dir = root.resolve()
        self.memory_path = self.state_dir / "user_memory.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[MemoryItem]:
        if not self.memory_path.exists():
            return []
        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_items = data.get("memories", []) if isinstance(data, dict) else []
        result: list[MemoryItem] = []
        for raw in raw_items:
            if isinstance(raw, dict) and "id" in raw and "content" in raw:
                result.append(MemoryItem.from_dict(raw))
        return result

    def add(self, content: str) -> MemoryItem:
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空。")
        items = self.list()
        item = MemoryItem(
            id=f"mem_{self._take_next_index(items)}",
            content=content,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        items.append(item)
        self._save(items)
        return item

    def delete(self, memory_id: str) -> bool:
        items = self.list()
        kept = [m for m in items if m.id != memory_id]
        if len(kept) == len(items):
            return False
        self._save(kept)
        return True

    def clear(self) -> int:
        count = len(self.list())
        self._save([])
        return count

    def _save(self, items: list[MemoryItem]) -> None:
        payload = {
            "memories": [m.to_dict() for m in items],
            "next_index": self._stored_next_index(),
        }
        self.memory_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _take_next_index(self, existing: list[MemoryItem]) -> int:
        """Return a never-reused index: a forgotten id must not come back
        pointing at a different, newer fact (the model may still hold a
        stale reference to it)."""
        index = max(
            self._stored_next_index(),
            self._highest_item_index(existing) + 1,
        )
        self._next_index = index + 1
        return index

    def _stored_next_index(self) -> int:
        cached = getattr(self, "_next_index", None)
        if cached is not None:
            return cached
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            raw = data.get("next_index") if isinstance(data, dict) else None
            if isinstance(raw, int) and raw > 0:
                self._next_index = raw
                return raw
        self._next_index = self._highest_item_index(self.list()) + 1
        return self._next_index

    @staticmethod
    def _highest_item_index(items: list[MemoryItem]) -> int:
        highest = 0
        for item in items:
            if item.id.startswith("mem_") and item.id[4:].isdigit():
                highest = max(highest, int(item.id[4:]))
        return highest
