"""Directory listing tool."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolExecutionContext
from .result import ToolOutput


class ListDirTool(Tool):
    """List files and directories at a given path."""

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "列出目录内容。返回文件名列表，目录名以 / 结尾。"

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录路径，默认为当前目录",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    _MAX_ENTRIES = 200

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        path: str = ".",
        **kwargs: Any,
    ) -> ToolOutput:
        del context
        try:
            p = self._resolve_path(path)
        except PermissionError as exc:
            return ToolOutput.failure("permission_denied", f"[安全拦截] {exc}")
        if not p.exists():
            return ToolOutput.failure(
                "not_found",
                f"路径不存在: {path}",
                data={"path": path},
            )
        if not p.is_dir():
            return ToolOutput.failure(
                "error",
                f"不是目录: {path}",
                data={"path": path},
            )
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        total_entries = len(entries)
        if len(entries) > self._MAX_ENTRIES:
            entries = entries[: self._MAX_ENTRIES]
            truncated = True
        else:
            truncated = False
        names = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        return ToolOutput.success(
            (
                f"已列出 {path}，共 {total_entries} 项。"
                if not truncated
                else f"已列出 {path} 的前 {len(names)} 项（共 {total_entries} 项）。"
            ),
            data={
                "path": path,
                "entries": names,
                "total_entries": total_entries,
            },
            truncated=truncated,
        )
