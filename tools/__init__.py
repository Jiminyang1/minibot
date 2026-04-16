"""Tool registry for MiniBot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import exec_cmd, read_file

_REGISTRY: dict[str, Callable[[dict[str, Any]], str]] = {
    "exec": exec_cmd.execute,
    "read_file": read_file.execute,
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    exec_cmd.DEFINITION,
    read_file.DEFINITION,
]


def execute_tool(name: str, args: dict[str, Any]) -> str:
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"工具执行报错: 未知工具 {name}"
    try:
        return fn(args)
    except Exception as exc:
        return f"工具执行报错: {exc}"
