"""String-replacement file editing tool."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolExecutionContext
from .result import ToolResult


class EditFileTool(Tool):
    """Replace a unique string in an existing UTF-8 file."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "在已有文件中做精确的字符串替换。"
            "old_string 必须在文件中唯一匹配（除非 replace_all=true），"
            "否则会报错并要求你提供更长的上下文来唯一定位。"
            "相比 write_file 不需要重传整个文件，推荐作为编辑现有文件的首选。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要编辑的文件路径。",
                },
                "old_string": {
                    "type": "string",
                    "description": "要被替换的原文本，必须与文件中的内容逐字符一致。",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新文本。",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "是否替换所有出现；默认 false，要求唯一匹配。",
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        }

    _MAX_SIZE = 256 * 1024  # 256 KB

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        del context
        try:
            p = self._resolve_path(path)
        except PermissionError as exc:
            return ToolResult.failure("permission_denied", f"[安全拦截] {exc}")
        if not p.exists():
            return ToolResult.failure(
                "not_found",
                f"文件不存在: {path}",
                data={"path": path},
            )
        if not p.is_file():
            return ToolResult.failure(
                "error",
                f"不是文件: {path}",
                data={"path": path},
            )
        if p.stat().st_size > self._MAX_SIZE:
            return ToolResult.failure(
                "error",
                f"文件过大 ({p.stat().st_size} bytes)，上限 {self._MAX_SIZE} bytes。",
                data={"path": path, "size_bytes": p.stat().st_size},
            )
        if not old_string:
            return ToolResult.failure(
                "invalid_args",
                "old_string 不能为空。",
                data={"path": path},
            )
        if old_string == new_string:
            return ToolResult.noop(
                "old_string 与 new_string 相同，无需编辑。",
                data={"path": path},
            )

        try:
            original = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(
                "error",
                f"文件无法以 UTF-8 解码: {path}",
                data={"path": path},
            )

        occurrences = original.count(old_string)
        if occurrences == 0:
            return ToolResult.failure(
                "not_found",
                f"未在文件中找到 old_string（0 处匹配）: {path}",
                data={"path": path},
            )
        if occurrences > 1 and not replace_all:
            return ToolResult.failure(
                "invalid_args",
                (
                    f"old_string 在文件中匹配到 {occurrences} 处；"
                    "请提供更长的上下文以唯一定位，或设置 replace_all=true。"
                ),
                data={"path": path, "occurrences": occurrences},
            )

        if replace_all:
            updated = original.replace(old_string, new_string)
            count = occurrences
        else:
            updated = original.replace(old_string, new_string, 1)
            count = 1

        try:
            p.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(
                "error",
                f"写入失败: {exc}",
                data={"path": path},
            )
        return ToolResult.success(
            f"已编辑 {p}（替换 {count} 处）。",
            data={
                "path": path,
                "replacements": count,
                "replace_all": replace_all,
            },
        )
