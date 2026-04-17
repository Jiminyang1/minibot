"""Long-term memory tools for MiniBot.

These tools let the model write and revise the cross-session memory store.
Reading is not exposed as a tool: every turn, the current memories are
rendered directly into the system prompt.
"""

from __future__ import annotations

from typing import Any

from ..memory import MemoryStore
from .base import Tool


class RememberTool(Tool):
    """Persist a single stable fact about the user to long-term memory."""

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(workspace=None)
        self._store = store

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "把一条关于用户的稳定事实写入长期记忆，跨会话可见。"
            "适合：姓名、身份、常用环境、偏好、当前正在进行的项目及其高层状态。"
            "不适合：一次性的临时信息、daily 进度流水。"
            "如果是对同一项目/状态的更新，请先用 forget 删除旧条目，再写入新的。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "要记住的事实，一句话，尽量自足可读。",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        }

    def execute(self, *, content: str, **kwargs: Any) -> str:
        try:
            item = self._store.add(content)
        except ValueError as exc:
            return f"写入失败: {exc}"
        except OSError as exc:
            return f"写入失败: {exc}"
        return f"已记住 [{item.id}]: {item.content}"


class ForgetTool(Tool):
    """Delete a single fact from long-term memory by its id."""

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(workspace=None)
        self._store = store

    @property
    def name(self) -> str:
        return "forget"

    @property
    def description(self) -> str:
        return (
            "按 id 删除一条长期记忆。"
            "用于修正过时或错误的事实；id 可以在 system prompt 中每条记忆前面的方括号里找到。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要删除的记忆 id，例如 m_20260417_010203。",
                },
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        }

    def execute(self, *, memory_id: str, **kwargs: Any) -> str:
        memory_id = memory_id.strip()
        if not memory_id:
            return "删除失败: memory_id 不能为空。"
        try:
            removed = self._store.delete(memory_id)
        except OSError as exc:
            return f"删除失败: {exc}"
        if not removed:
            return f"未找到记忆 {memory_id}。"
        return f"已删除记忆 {memory_id}。"
