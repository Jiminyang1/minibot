"""Materialize raw tool outputs into model-facing results."""

from __future__ import annotations

from ..artifacts import ArtifactStore
from ..tools.base import ToolExecutionContext
from ..tools.result import ToolOutput, ToolResult


class ToolOutputMaterializer:
    """Decide whether tool content is inlined or stored as an artifact."""

    _INLINE_CONTENT_CHARS = 12000
    _PREVIEW_CHARS = 2000

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    def materialize(
        self,
        output: ToolOutput,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if output.content is None:
            return ToolResult(
                ok=output.ok,
                code=output.code,
                summary=output.summary,
                data=dict(output.data),
                truncated=output.truncated,
                meta=dict(output.meta),
            )

        data = dict(output.data)
        if len(output.content) <= self._INLINE_CONTENT_CHARS:
            data["content"] = output.content
            return ToolResult(
                ok=output.ok,
                code=output.code,
                summary=output.summary,
                data=data,
                truncated=output.truncated,
                meta=dict(output.meta),
            )

        artifact = self._artifact_store.put_text(
            context.session_id,
            output.content,
            kind=output.content_kind,
            name=output.content_name,
        )
        data.setdefault("preview", output.content[: self._PREVIEW_CHARS])

        summary = output.summary
        if "artifact" not in summary and "截断" not in summary:
            summary = f"{summary.rstrip('。')}（结果较大，已返回预览并保存为 artifact）。"

        return ToolResult(
            ok=output.ok,
            code=output.code,
            summary=summary,
            data=data,
            artifact=artifact,
            truncated=True,
            meta=dict(output.meta),
        )
