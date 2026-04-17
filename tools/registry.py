"""Tool registry for MiniBot."""

from __future__ import annotations

from collections.abc import Iterable
import inspect
from typing import Any

from .base import Tool, ToolExecutionContext, ToolLayer
from .result import ToolOutput


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

    def list_tools(self, *, layer: ToolLayer | None = None) -> list[Tool]:
        tools = list(self._tools.values())
        if layer is None:
            return tools
        return [tool for tool in tools if tool.layer == layer]

    def get_definitions(self, *, layer: ToolLayer | None = None) -> list[dict[str, Any]]:
        return [tool.to_definition() for tool in self.list_tools(layer=layer)]

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        context: ToolExecutionContext,
    ) -> ToolOutput:
        if not isinstance(args, dict):
            return ToolOutput.failure(
                "invalid_args",
                f"工具 {name} 参数错误: 参数必须是 JSON 对象。",
                data={"tool": name, "received_type": type(args).__name__},
            )

        tool = self.get(name)
        if tool is None:
            return ToolOutput.failure(
                "not_found",
                f"未知工具: {name}",
                data={"tool": name},
            )

        try:
            inspect.signature(tool.execute).bind(context=context, **args)
        except TypeError as exc:
            return ToolOutput.failure(
                "invalid_args",
                f"工具 {name} 参数错误: {exc}",
                data={"tool": name, "args": args},
            )

        try:
            result = tool.execute(context=context, **args)
        except Exception as exc:
            return ToolOutput.failure(
                "error",
                f"工具 {name} 执行失败: {exc}",
                data={"tool": name},
                meta={"exception": repr(exc)},
            )
        if not isinstance(result, ToolOutput):
            return ToolOutput.failure(
                "error",
                f"工具 {name} 返回了无效结果类型。",
                data={"tool": name, "returned_type": type(result).__name__},
            )
        return result
