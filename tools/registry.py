"""Tool registry for MiniBot."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import inspect
from typing import Any

import jsonschema

from .base import Tool, ToolExecutionContext, ToolLayer
from .definitions import ModelToolDefinition
from .result import ToolOutput


@dataclass(frozen=True)
class PreparedToolCall:
    """Validated tool invocation ready for execution."""

    tool: Tool
    args: dict[str, Any]
    context: ToolExecutionContext
    tool_call_id: str | None = None


class ToolRegistry:
    """Manage tool definitions and dispatch execution by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # name -> compiled validator, or None when the tool's schema is
        # malformed/unsupported (validation degrades to signature binding).
        self._validators: dict[str, Any] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._validators.pop(tool.name, None)

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

    def get_definitions(
        self,
        *,
        layer: ToolLayer | None = None,
    ) -> list[ModelToolDefinition]:
        return [tool.to_model_definition() for tool in self.list_tools(layer=layer)]

    def prepare(
        self,
        name: str,
        args: dict[str, Any],
        *,
        context: ToolExecutionContext,
    ) -> PreparedToolCall | ToolOutput:
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

        # Validate against the schema the model was shown — errors surface
        # before execution with a message the model can self-correct from.
        schema_error = self._validate_args(tool, args)
        if schema_error is not None:
            return ToolOutput.failure(
                "invalid_args",
                f"工具 {name} 参数校验失败: {schema_error}",
                data={"tool": name, "args": args},
            )

        try:
            inspect.signature(tool.execute).bind(context=context, **args)
        except TypeError as exc:
            return ToolOutput.failure(
                "invalid_args",
                f"工具 {name} 参数错误: {exc}",
                data={"tool": name, "args": args},
            )

        return PreparedToolCall(tool=tool, args=args, context=context)

    def _validate_args(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """Return a readable schema violation, or None when args are valid."""
        validator = self._validator_for(tool)
        if validator is None:
            return None
        error = jsonschema.exceptions.best_match(validator.iter_errors(args))
        if error is None:
            return None
        return f"{error.json_path}: {error.message}"

    def _validator_for(self, tool: Tool) -> Any:
        if tool.name in self._validators:
            return self._validators[tool.name]
        validator: Any = None
        try:
            schema = tool.parameters
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
        except Exception:
            # Malformed or unsupported schema (some MCP servers ship loose
            # ones): skip schema validation, signature binding still applies.
            validator = None
        self._validators[tool.name] = validator
        return validator

    def invoke(self, prepared: PreparedToolCall) -> ToolOutput:
        try:
            result = prepared.tool.execute(context=prepared.context, **prepared.args)
        except Exception as exc:
            return ToolOutput.failure(
                "error",
                f"工具 {prepared.tool.name} 执行失败: {exc}",
                data={"tool": prepared.tool.name},
                meta={"exception": repr(exc)},
            )
        if not isinstance(result, ToolOutput):
            return ToolOutput.failure(
                "error",
                f"工具 {prepared.tool.name} 返回了无效结果类型。",
                data={"tool": prepared.tool.name, "returned_type": type(result).__name__},
            )
        return result

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        context: ToolExecutionContext,
    ) -> ToolOutput:
        prepared = self.prepare(name, args, context=context)
        if isinstance(prepared, ToolOutput):
            return prepared
        return self.invoke(prepared)
