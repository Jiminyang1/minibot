"""File reading tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import sha256_text
from .base import Tool, ToolExecutionContext
from .result import ToolOutput


class ReadFileTool(Tool):
    """Read UTF-8 file contents from disk."""

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取文件内容"

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
                    "description": "文件路径",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    _MAX_SIZE = 256 * 1024  # 256 KB

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        path: str,
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
                f"文件不存在: {path}",
                data={"path": path},
            )
        if p.stat().st_size > self._MAX_SIZE:
            return ToolOutput.failure(
                "error",
                f"文件过大 ({p.stat().st_size} bytes)，上限 {self._MAX_SIZE} bytes。",
                data={"path": path, "size_bytes": p.stat().st_size},
            )
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolOutput.failure(
                "error",
                f"文件无法以 UTF-8 解码: {path}",
                data={"path": path},
            )

        total_chars = len(content)
        return ToolOutput.success(
            f"已读取 {path}（{total_chars} 字符）。",
            data={
                "path": path,
                "total_chars": total_chars,
                "file_sha256": sha256_text(content),
            },
            content=content,
            content_kind="file",
            content_name=path,
        )
