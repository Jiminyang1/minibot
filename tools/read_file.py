"""File reading tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool


class ReadFileTool(Tool):
    """Read UTF-8 file contents from disk."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取文件内容"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    _MAX_SIZE = 256 * 1024  # 256 KB

    def execute(self, *, path: str, **kwargs: Any) -> str:
        p = Path(path)
        if not p.exists():
            return f"文件不存在: {path}"
        if p.stat().st_size > self._MAX_SIZE:
            return f"文件过大 ({p.stat().st_size} bytes)，上限 {self._MAX_SIZE} bytes。"
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"文件无法以 UTF-8 解码: {path}"
