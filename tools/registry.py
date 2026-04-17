"""Tool registry for MiniBot."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import Tool


class ToolRegistry:
    """Manage tool definitions and dispatch execution by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_definition() for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return f"工具执行报错: 工具参数必须是 JSON 对象，收到 {type(args).__name__}"

        tool = self.get(name)
        if tool is None:
            return f"工具执行报错: 未知工具 {name}"

        try:
            return tool.execute(**args)
        except TypeError as exc:
            return f"工具执行报错: 参数错误: {exc}"
        except Exception as exc:
            return f"工具执行报错: {exc}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
