"""Hash-guarded line editing tool."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

from ..artifacts import sha256_text
from .base import Tool, ToolExecutionContext
from .result import ToolOutput


@dataclass(frozen=True)
class _Mutation:
    start: int
    end: int
    new_text: str
    label: str


class EditFileTool(Tool):
    """Apply optimistic-lock line edits to an existing UTF-8 file."""

    _MAX_SIZE = 256 * 1024  # 256 KB
    _PREVIEW_CHARS = 200

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "在已有 UTF-8 文件上执行基于 expected_sha256 的行级编辑。"
            "调用前必须先 read_file 或 read_artifact 取得 data.file_sha256。"
            "支持 replace、insert_before、insert_after、append；"
            "拒绝过期快照、错误行范围和重叠编辑。"
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
                "expected_sha256": {
                    "type": "string",
                    "description": (
                        "必须来自最近一次 read_file 或 read_artifact 的 "
                        "data.file_sha256。若文件已变更，将返回 conflict。"
                    ),
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "编辑列表。replace 需要 start_line/end_line/old_text/new_text；"
                        "insert_before 与 insert_after 需要 line/new_text；"
                        "append 只需要 new_text。所有行号均为 1-based。"
                    ),
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "replace",
                                    "insert_before",
                                    "insert_after",
                                    "append",
                                ],
                            },
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "line": {"type": "integer", "minimum": 1},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                        },
                        "required": ["op", "new_text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "expected_sha256", "edits"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        path: str,
        expected_sha256: str,
        edits: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
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
        if not p.is_file():
            return ToolOutput.failure(
                "error",
                f"不是文件: {path}",
                data={"path": path},
            )
        if p.stat().st_size > self._MAX_SIZE:
            return ToolOutput.failure(
                "error",
                f"文件过大 ({p.stat().st_size} bytes)，上限 {self._MAX_SIZE} bytes。",
                data={"path": path, "size_bytes": p.stat().st_size},
            )
        if not expected_sha256:
            return ToolOutput.failure(
                "invalid_args",
                "expected_sha256 不能为空。",
                data={"path": path},
            )
        if not isinstance(edits, list) or not edits:
            return ToolOutput.failure(
                "invalid_args",
                "edits 必须是非空数组。",
                data={"path": path},
            )

        try:
            original = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolOutput.failure(
                "error",
                f"文件无法以 UTF-8 解码: {path}",
                data={"path": path},
            )

        current_sha = sha256_text(original)
        if expected_sha256 != current_sha:
            return ToolOutput.failure(
                "conflict",
                "文件已被修改，sha256 不匹配。请重新 read_file 再 edit。",
                data={
                    "path": path,
                    "expected_sha256": expected_sha256,
                    "current_sha256": current_sha,
                },
            )

        line_spans = self._line_spans(original)
        mutations_or_error = self._build_mutations(
            path=path,
            original=original,
            line_spans=line_spans,
            edits=edits,
        )
        if isinstance(mutations_or_error, ToolOutput):
            return mutations_or_error

        overlap_error = self._validate_non_overlapping(path, mutations_or_error)
        if overlap_error is not None:
            return overlap_error

        updated = original
        for mutation in sorted(
            mutations_or_error,
            key=lambda item: (item.start, item.end, item.label),
            reverse=True,
        ):
            updated = updated[: mutation.start] + mutation.new_text + updated[mutation.end :]

        if updated == original:
            return ToolOutput.noop(
                "编辑结果与原文件相同，无需写入。",
                data={"path": path, "expected_sha256": expected_sha256},
            )

        try:
            self._atomic_write_text(p, updated)
        except OSError as exc:
            return ToolOutput.failure(
                "error",
                f"写入失败: {exc}",
                data={"path": path},
            )

        new_sha = sha256_text(updated)
        return ToolOutput.success(
            f"已编辑 {p}（应用 {len(mutations_or_error)} 处变更）。",
            data={
                "path": path,
                "edits_applied": len(mutations_or_error),
                "previous_sha256": current_sha,
                "file_sha256": new_sha,
            },
        )

    def _build_mutations(
        self,
        *,
        path: str,
        original: str,
        line_spans: list[tuple[int, int]],
        edits: list[dict[str, Any]],
    ) -> list[_Mutation] | ToolOutput:
        mutations: list[_Mutation] = []
        for index, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                return ToolOutput.failure(
                    "invalid_args",
                    f"第 {index} 个 edit 必须是对象。",
                    data={"path": path, "edit_index": index},
                )
            op = edit.get("op")
            if op == "replace":
                mutation = self._build_replace_mutation(
                    path=path,
                    original=original,
                    line_spans=line_spans,
                    edit=edit,
                    index=index,
                )
            elif op == "insert_before":
                mutation = self._build_insert_before_mutation(
                    path=path,
                    original=original,
                    line_spans=line_spans,
                    edit=edit,
                    index=index,
                )
            elif op == "insert_after":
                mutation = self._build_insert_after_mutation(
                    path=path,
                    original=original,
                    line_spans=line_spans,
                    edit=edit,
                    index=index,
                )
            elif op == "append":
                mutation = self._build_append_mutation(
                    path=path,
                    original=original,
                    edit=edit,
                    index=index,
                )
            else:
                return ToolOutput.failure(
                    "invalid_args",
                    f"第 {index} 个 edit 的 op 非法: {op!r}",
                    data={"path": path, "edit_index": index},
                )
            if isinstance(mutation, ToolOutput):
                return mutation
            mutations.append(mutation)
        return mutations

    def _build_replace_mutation(
        self,
        *,
        path: str,
        original: str,
        line_spans: list[tuple[int, int]],
        edit: dict[str, Any],
        index: int,
    ) -> _Mutation | ToolOutput:
        unexpected = set(edit) - {"op", "start_line", "end_line", "old_text", "new_text"}
        if unexpected:
            return self._invalid_edit_keys(path, index, unexpected)

        start_line = self._read_positive_int(edit.get("start_line"))
        end_line = self._read_positive_int(edit.get("end_line"))
        old_text = edit.get("old_text")
        new_text = edit.get("new_text")
        if start_line is None or end_line is None:
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 replace edit 必须提供正整数 start_line 和 end_line。",
                data={"path": path, "edit_index": index},
            )
        if start_line > end_line:
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 replace edit 的 start_line 不能大于 end_line。",
                data={"path": path, "edit_index": index},
            )
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 replace edit 必须提供字符串 old_text/new_text。",
                data={"path": path, "edit_index": index},
            )
        if not line_spans:
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 replace edit 无法作用于空文件。",
                data={"path": path, "edit_index": index},
            )
        if end_line > len(line_spans):
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 replace edit 超出文件行数 {len(line_spans)}。",
                data={"path": path, "edit_index": index, "line_count": len(line_spans)},
            )

        start = line_spans[start_line - 1][0]
        end = line_spans[end_line - 1][1]
        actual = original[start:end]
        if actual != old_text:
            return ToolOutput.failure(
                "invalid_args",
                (
                    f"第 {index} 个 replace edit 的 old_text 与 "
                    f"{start_line}-{end_line} 行内容不一致。"
                ),
                data={
                    "path": path,
                    "edit_index": index,
                    "start_line": start_line,
                    "end_line": end_line,
                    "actual_text_preview": self._preview(actual),
                },
            )

        return _Mutation(
            start=start,
            end=end,
            new_text=new_text,
            label=f"replace#{index}",
        )

    def _build_insert_before_mutation(
        self,
        *,
        path: str,
        original: str,
        line_spans: list[tuple[int, int]],
        edit: dict[str, Any],
        index: int,
    ) -> _Mutation | ToolOutput:
        unexpected = set(edit) - {"op", "line", "new_text"}
        if unexpected:
            return self._invalid_edit_keys(path, index, unexpected)

        line = self._read_positive_int(edit.get("line"))
        new_text = edit.get("new_text")
        if line is None or not isinstance(new_text, str):
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 insert_before edit 必须提供正整数 line 和字符串 new_text。",
                data={"path": path, "edit_index": index},
            )
        if line_spans:
            if line > len(line_spans) + 1:
                return ToolOutput.failure(
                    "invalid_args",
                    f"第 {index} 个 insert_before edit 超出允许行号 {len(line_spans) + 1}。",
                    data={"path": path, "edit_index": index, "line_count": len(line_spans)},
                )
            start = len(original) if line == len(line_spans) + 1 else line_spans[line - 1][0]
        else:
            if line != 1:
                return ToolOutput.failure(
                    "invalid_args",
                    f"空文件只能对第 1 行做 insert_before；收到 line={line}。",
                    data={"path": path, "edit_index": index},
                )
            start = 0

        return _Mutation(
            start=start,
            end=start,
            new_text=new_text,
            label=f"insert_before#{index}",
        )

    def _build_insert_after_mutation(
        self,
        *,
        path: str,
        original: str,
        line_spans: list[tuple[int, int]],
        edit: dict[str, Any],
        index: int,
    ) -> _Mutation | ToolOutput:
        unexpected = set(edit) - {"op", "line", "new_text"}
        if unexpected:
            return self._invalid_edit_keys(path, index, unexpected)

        del original
        line = self._read_positive_int(edit.get("line"))
        new_text = edit.get("new_text")
        if line is None or not isinstance(new_text, str):
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 insert_after edit 必须提供正整数 line 和字符串 new_text。",
                data={"path": path, "edit_index": index},
            )
        if not line_spans:
            return ToolOutput.failure(
                "invalid_args",
                f"空文件不能执行第 {index} 个 insert_after edit，请改用 append 或 insert_before(line=1)。",
                data={"path": path, "edit_index": index},
            )
        if line > len(line_spans):
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 insert_after edit 超出文件行数 {len(line_spans)}。",
                data={"path": path, "edit_index": index, "line_count": len(line_spans)},
            )

        start = line_spans[line - 1][1]
        return _Mutation(
            start=start,
            end=start,
            new_text=new_text,
            label=f"insert_after#{index}",
        )

    def _build_append_mutation(
        self,
        *,
        path: str,
        original: str,
        edit: dict[str, Any],
        index: int,
    ) -> _Mutation | ToolOutput:
        unexpected = set(edit) - {"op", "new_text"}
        if unexpected:
            return self._invalid_edit_keys(path, index, unexpected)

        new_text = edit.get("new_text")
        if not isinstance(new_text, str):
            return ToolOutput.failure(
                "invalid_args",
                f"第 {index} 个 append edit 必须提供字符串 new_text。",
                data={"path": path, "edit_index": index},
            )

        start = len(original)
        return _Mutation(
            start=start,
            end=start,
            new_text=new_text,
            label=f"append#{index}",
        )

    def _validate_non_overlapping(
        self,
        path: str,
        mutations: list[_Mutation],
    ) -> ToolOutput | None:
        previous: _Mutation | None = None
        for mutation in sorted(mutations, key=lambda item: (item.start, item.end, item.label)):
            if previous is None:
                previous = mutation
                continue
            if mutation.start < previous.end or (
                mutation.start == previous.end
                and (mutation.start == mutation.end or previous.start == previous.end)
            ):
                return ToolOutput.failure(
                    "invalid_args",
                    "edits 存在重叠或共享同一插入点，请先合并为单个 edit。",
                    data={
                        "path": path,
                        "previous_edit": previous.label,
                        "conflicting_edit": mutation.label,
                    },
                )
            previous = mutation
        return None

    @staticmethod
    def _line_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for chunk in text.splitlines(keepends=True):
            start = cursor
            cursor += len(chunk)
            spans.append((start, cursor))
        return spans

    @staticmethod
    def _read_positive_int(value: Any) -> int | None:
        return value if isinstance(value, int) and value >= 1 else None

    def _invalid_edit_keys(
        self,
        path: str,
        index: int,
        keys: set[str],
    ) -> ToolOutput:
        return ToolOutput.failure(
            "invalid_args",
            f"第 {index} 个 edit 包含未支持字段: {sorted(keys)}",
            data={"path": path, "edit_index": index},
        )

    def _preview(self, text: str) -> str:
        compact = text if len(text) <= self._PREVIEW_CHARS else text[: self._PREVIEW_CHARS] + "..."
        return compact.replace("\n", "\\n")

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_path = tmp.name
            os.replace(temp_path, path)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
