"""File content search tool."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import Tool, ToolExecutionContext
from .result import ToolResult


class SearchFilesTool(Tool):
    """Search file contents by regex pattern under a directory."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return (
            "在目录中搜索文件内容。支持正则表达式，"
            "返回匹配的文件名、行号和内容。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索的正则表达式",
                },
                "path": {
                    "type": "string",
                    "description": "搜索的目录路径，默认为当前目录",
                },
                "glob": {
                    "type": "string",
                    "description": "文件名过滤，例如 '*.py'，默认 '*'",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    _MAX_MATCHES = 50
    _MAX_FILE_SIZE = 256 * 1024
    _SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".minibot"}

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        pattern: str,
        path: str = ".",
        glob: str = "*",
        **kwargs: Any,
    ) -> ToolResult:
        try:
            root = self._resolve_path(path)
        except PermissionError as exc:
            return ToolResult.failure("permission_denied", f"[安全拦截] {exc}")
        if not root.is_dir():
            return ToolResult.failure(
                "not_found",
                f"不是目录: {path}",
                data={"path": path},
            )
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult.failure(
                "invalid_args",
                f"正则表达式无效: {exc}",
                data={"pattern": pattern},
            )

        matches: list[str] = []
        for file in self._walk(root, glob):
            if file.stat().st_size > self._MAX_FILE_SIZE:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = file.relative_to(root) if file.is_relative_to(root) else file
                    matches.append(f"{rel}:{lineno}: {line.rstrip()}")

        if not matches:
            return ToolResult.failure(
                "not_found",
                f"未找到匹配: {pattern}",
                data={"pattern": pattern, "path": path, "total_matches": 0, "matches": []},
            )

        total_matches = len(matches)
        preview_matches = matches[: self._MAX_MATCHES]
        if total_matches <= self._MAX_MATCHES:
            return ToolResult.success(
                f"找到 {total_matches} 处匹配。",
                data={
                    "pattern": pattern,
                    "path": path,
                    "glob": glob,
                    "total_matches": total_matches,
                    "matches": preview_matches,
                },
            )

        artifact = self._require_session_manager().put_artifact_text(
            context.session_id,
            "\n".join(matches),
            kind="text",
            name=f"search:{pattern}",
        )
        return ToolResult.success(
            f"找到 {total_matches} 处匹配，已返回前 {self._MAX_MATCHES} 条。",
            data={
                "pattern": pattern,
                "path": path,
                "glob": glob,
                "total_matches": total_matches,
                "matches": preview_matches,
            },
            artifact=artifact,
            truncated=True,
        )

    def _walk(self, root: Path, glob_pattern: str):
        for item in sorted(root.iterdir(), key=lambda e: e.name):
            if item.name in self._SKIP_DIRS:
                continue
            if item.is_dir():
                yield from self._walk(item, glob_pattern)
            elif item.is_file() and item.match(glob_pattern):
                yield item
