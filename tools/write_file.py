"""File writing tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import sha256_text
from .base import Tool, ToolExecutionContext
from .result import ToolOutput


class WriteFileTool(Tool):
    """Create or overwrite a UTF-8 file on disk."""

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "写入文件内容。如果文件已存在则覆盖（需带 expected_sha256），"
            "不存在则创建（含中间目录）。"
        )

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
                "expected_sha256": {
                    "type": "string",
                    "description": (
                        "覆盖已有文件时必填，应来自最近一次 read_file 或 read_artifact "
                        "返回的 data.file_sha256；不匹配会返回 conflict。新建文件可不传。"
                    ),
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
        expected_sha256: str | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        del context
        try:
            p = self._resolve_path(path)
        except PermissionError as exc:
            return ToolOutput.failure("permission_denied", f"[安全拦截] {exc}")
        if len(content.encode("utf-8")) > self._MAX_SIZE:
            return ToolOutput.failure(
                "error",
                f"内容过大 ({len(content.encode('utf-8'))} bytes)，上限 {self._MAX_SIZE} bytes。",
                data={"path": path, "size_bytes": len(content.encode('utf-8'))},
            )

        existed = p.exists()
        if existed:
            try:
                current_text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolOutput.failure(
                    "conflict",
                    "无法校验现有文件（非 UTF-8），拒绝覆盖。",
                    data={"path": path},
                )
            current_sha = sha256_text(current_text)
            if expected_sha256 is None:
                return ToolOutput.failure(
                    "conflict",
                    "覆盖已有文件需提供 expected_sha256（请先 read_file 获取）。",
                    data={"path": path, "current_sha256": current_sha},
                )
            if expected_sha256 != current_sha:
                return ToolOutput.failure(
                    "conflict",
                    "文件已被修改，sha256 不匹配。请重新 read_file 再写。",
                    data={
                        "path": path,
                        "expected_sha256": expected_sha256,
                        "current_sha256": current_sha,
                    },
                )

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolOutput.failure(
                "error",
                f"写入失败: {exc}",
                data={"path": path},
            )
        return ToolOutput.success(
            f"已写入 {p}（{len(content)} 字符）。",
            data={
                "path": path,
                "chars_written": len(content),
                "created": not existed,
                "file_sha256": sha256_text(content),
            },
        )
