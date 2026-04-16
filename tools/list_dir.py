"""Directory listing tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool


class ListDirTool(Tool):
    """List files and directories at a given path."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "列出目录内容。返回文件名列表，目录名以 / 结尾。"

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

    def execute(self, *, path: str = ".", **kwargs: Any) -> str:
        try:
            p = self._resolve_path(path)
        except PermissionError as exc:
            return f"[安全拦截] {exc}"
        if not p.exists():
            return f"路径不存在: {path}"
        if not p.is_dir():
            return f"不是目录: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        if len(entries) > self._MAX_ENTRIES:
            entries = entries[: self._MAX_ENTRIES]
            truncated = True
        else:
            truncated = False
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        result = "\n".join(lines)
        if truncated:
            result += f"\n...(已截断，共 {len(list(p.iterdir()))} 项)"
        return result
