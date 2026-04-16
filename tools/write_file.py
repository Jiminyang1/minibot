"""File writing tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool


class WriteFileTool(Tool):
    """Create or overwrite a UTF-8 file on disk."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "写入文件内容。如果文件已存在则覆盖，不存在则创建（含中间目录）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文件内容",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    _MAX_SIZE = 256 * 1024  # 256 KB

    def execute(self, *, path: str, content: str, **kwargs: Any) -> str:
        try:
            p = self._resolve_path(path)
        except PermissionError as exc:
            return f"[安全拦截] {exc}"
        if len(content.encode("utf-8")) > self._MAX_SIZE:
            return f"内容过大 ({len(content.encode('utf-8'))} bytes)，上限 {self._MAX_SIZE} bytes。"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"写入失败: {exc}"
        return f"已写入 {p} ({len(content)} 字符)"
