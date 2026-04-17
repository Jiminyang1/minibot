"""Tool for reading stored artifacts."""

from __future__ import annotations

import json
from typing import Any
from typing import TYPE_CHECKING

from .base import Tool, ToolExecutionContext
from .result import ToolResult

if TYPE_CHECKING:
    from ..session import SessionManager


class ReadArtifactTool(Tool):
    """Read a previously stored artifact by character range."""

    _DEFAULT_LIMIT = 4000
    _MAX_LIMIT = 8000

    def __init__(self, session_manager: SessionManager) -> None:
        super().__init__(workspace=None, session_manager=session_manager)

    @property
    def name(self) -> str:
        return "read_artifact"

    @property
    def description(self) -> str:
        return "读取先前工具返回的大结果 artifact，按字符分页回查。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "artifact id，例如 a_123abc。",
                },
                "offset": {
                    "type": "integer",
                    "description": "起始字符偏移，默认 0。",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "读取的字符数，默认 4000，上限 8000。",
                    "minimum": 1,
                    "maximum": self._MAX_LIMIT,
                },
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        artifact_id: str,
        offset: int = 0,
        limit: int = _DEFAULT_LIMIT,
        **kwargs: Any,
    ) -> ToolResult:
        artifact_id = artifact_id.strip()
        if not artifact_id:
            return ToolResult.failure(
                "invalid_args",
                "读取 artifact 失败: artifact_id 不能为空。",
            )
        if offset < 0:
            return ToolResult.failure(
                "invalid_args",
                "读取 artifact 失败: offset 不能小于 0。",
                data={"artifact_id": artifact_id, "offset": offset},
            )
        if limit <= 0:
            return ToolResult.failure(
                "invalid_args",
                "读取 artifact 失败: limit 必须大于 0。",
                data={"artifact_id": artifact_id, "limit": limit},
            )
        limit = min(limit, self._MAX_LIMIT)

        try:
            page = self._require_session_manager().read_artifact_page(
                context.session_id,
                artifact_id,
                offset=offset,
                limit=limit,
            )
        except FileNotFoundError:
            return ToolResult.failure(
                "not_found",
                f"未找到 artifact {artifact_id}。",
                data={"artifact_id": artifact_id},
            )
        except ValueError as exc:
            return ToolResult.failure(
                "error",
                f"读取 artifact 失败: {exc}",
                data={"artifact_id": artifact_id},
            )
        except OSError as exc:
            return ToolResult.failure(
                "error",
                f"读取 artifact 失败: {exc}",
                data={"artifact_id": artifact_id},
            )

        content = self._fit_content(
            artifact_id=artifact_id,
            kind=page.ref.kind,
            name=page.ref.name,
            content=page.content,
            offset=page.offset,
            requested_limit=page.limit,
            total_chars=page.total_chars,
        )
        if not content:
            return ToolResult.failure(
                "error",
                f"读取 artifact 失败: artifact {artifact_id} 片段过大，无法安全返回。",
                data={"artifact_id": artifact_id},
            )

        returned_chars = len(content)
        returned_end = page.offset + returned_chars
        next_offset = returned_end if returned_end < page.total_chars else None
        has_more = next_offset is not None

        return ToolResult.success(
            (
                f"已读取 artifact {artifact_id} "
                f"({page.offset}-{returned_end}/{page.total_chars} 字符)"
            ),
            data={
                "artifact_id": artifact_id,
                "kind": page.ref.kind,
                "name": page.ref.name,
                "content": content,
                "offset": page.offset,
                "limit": page.limit,
                "returned_chars": returned_chars,
                "next_offset": next_offset,
                "has_more": has_more,
                "total_chars": page.total_chars,
            },
            truncated=has_more,
        )

    def _fit_content(
        self,
        *,
        artifact_id: str,
        kind: str,
        name: str | None,
        content: str,
        offset: int,
        requested_limit: int,
        total_chars: int,
    ) -> str:
        base_data = {
            "artifact_id": artifact_id,
            "kind": kind,
            "name": name,
            "offset": offset,
            "limit": requested_limit,
            "returned_chars": 0,
            "next_offset": None,
            "has_more": False,
            "total_chars": total_chars,
        }
        max_size = ToolResult._MAX_DATA_CHARS

        def serialized_size(text: str) -> int:
            returned_chars = len(text)
            returned_end = offset + returned_chars
            next_offset = returned_end if returned_end < total_chars else None
            payload = {
                **base_data,
                "content": text,
                "returned_chars": returned_chars,
                "next_offset": next_offset,
                "has_more": next_offset is not None,
            }
            return len(json.dumps(payload, ensure_ascii=False))

        if serialized_size(content) <= max_size:
            return content

        lo = 1
        hi = len(content)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = content[:mid]
            if serialized_size(candidate) <= max_size:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best
