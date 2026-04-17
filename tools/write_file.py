"""File writing tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Tool, ToolExecutionContext
from .result import ToolResult


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

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        path: str,
        content: str,
        **kwargs: Any,
    ) -> ToolResult:
        del context
        try:
            p = self._resolve_path(path)
        except PermissionError as exc:
            return ToolResult.failure("permission_denied", f"[安全拦截] {exc}")
        if len(content.encode("utf-8")) > self._MAX_SIZE:
            return ToolResult.failure(
                "error",
                f"内容过大 ({len(content.encode('utf-8'))} bytes)，上限 {self._MAX_SIZE} bytes。",
                data={"path": path, "size_bytes": len(content.encode('utf-8'))},
            )
        existed = p.exists()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(
                "error",
                f"写入失败: {exc}",
                data={"path": path},
            )
        return ToolResult.success(
            f"已写入 {p}（{len(content)} 字符）。",
            data={
                "path": path,
                "chars_written": len(content),
                "created": not existed,
            },
        )
