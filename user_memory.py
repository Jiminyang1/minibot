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
            id=self._next_id(items),
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
        payload = {"memories": [m.to_dict() for m in items]}
        self.memory_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _next_id(existing: list[MemoryItem]) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"m_{stamp}"
        taken = {m.id for m in existing}
        if base not in taken:
            return base
        suffix = 1
        while f"{base}_{suffix}" in taken:
            suffix += 1
        return f"{base}_{suffix}"
