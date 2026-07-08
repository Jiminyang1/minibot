"""Long-term memory tools for MiniBot.

These tools let the model write and revise the cross-session memory store.
Reading is not exposed as a tool: every turn, the current memories are
rendered into the prepared request context.
"""

from __future__ import annotations

from typing import Any

from ..user_memory import UserMemoryStore
from .base import Tool, ToolExecutionContext
from .result import ToolOutput


class RememberTool(Tool):
    """Persist a single stable fact about the user to global memory."""

    def __init__(self, store: UserMemoryStore) -> None:
        super().__init__(workspace=None)
        self._store = store

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "remember"

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "把一条关于用户的稳定事实写入长期记忆，跨会话可见。"
            "适合：姓名、身份、常用环境、偏好、固定习惯。"
            "不适合：一次性的临时信息、daily 进度流水、项目状态。"
            "如果是对同一事实的更新，请先用 forget 删除旧条目，再写入新的。"
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

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        content: str,
        **kwargs: Any,
    ) -> ToolOutput:
        del context
        try:
            item = self._store.add(content)
        except ValueError as exc:
            return ToolOutput.failure("invalid_args", f"写入失败: {exc}")
        except OSError as exc:
            return ToolOutput.failure("error", f"写入失败: {exc}")
        return ToolOutput.success(
            f"已记住 [{item.id}]。",
            data={"memory_id": item.id, "content": item.content},
        )


class ForgetTool(Tool):
    """Delete a single fact from long-term memory by its id."""

    def __init__(self, store: UserMemoryStore) -> None:
        super().__init__(workspace=None)
        self._store = store

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "forget"

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "按 id 删除一条长期记忆。"
            "用于修正过时或错误的事实；id 可以在 `/memory` 输出或当前上下文里的用户记忆数据块中找到。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要删除的记忆 id，例如 mem_1。",
                },
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        memory_id: str,
        **kwargs: Any,
    ) -> ToolOutput:
        del context
        memory_id = memory_id.strip()
        if not memory_id:
            return ToolOutput.failure(
                "invalid_args",
                "删除失败: memory_id 不能为空。",
            )
        try:
            removed = self._store.delete(memory_id)
        except OSError as exc:
            return ToolOutput.failure("error", f"删除失败: {exc}")
        if not removed:
            return ToolOutput.failure(
                "not_found",
                f"未找到记忆 {memory_id}。",
                data={"memory_id": memory_id},
            )
        return ToolOutput.success(
            f"已删除记忆 {memory_id}。",
            data={"memory_id": memory_id},
        )
