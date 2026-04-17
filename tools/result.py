"""Structured tool results for MiniBot."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, TypeAlias

from ..artifacts import ArtifactRef

ToolCode: TypeAlias = Literal[
    "success",
    "invalid_args",
    "not_found",
    "permission_denied",
    "timeout",
    "denied",
    "error",
    "noop",
]


@dataclass(frozen=True)
class ToolResult:
    """Stable envelope for every tool response.

    ``data`` is for *structured metadata* (paths, counts, short previews).
    Bulk content must go through the artifact store and be referenced via
    ``artifact``. Violators fail fast in ``__post_init__``.
    """

    # Soft cap on serialized ``data`` to prevent bypassing the artifact
    # mechanism. Raise this only if you have a concrete reason.
    _MAX_DATA_CHARS = 8000

    ok: bool
    code: ToolCode
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifact: ArtifactRef | None = None
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data:
            return
        size = len(json.dumps(self.data, ensure_ascii=False))
        if size > self._MAX_DATA_CHARS:
            raise ValueError(
                f"ToolResult.data 序列化长度 {size} 超过上限 "
                f"{self._MAX_DATA_CHARS}；请把大内容存为 artifact，"
                f"data 只保留结构化元信息与短预览。"
            )

    @classmethod
    def success(
        cls,
        summary: str,
        *,
        data: dict[str, Any] | None = None,
        artifact: ArtifactRef | None = None,
        truncated: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            code="success",
            summary=summary,
            data=data or {},
            artifact=artifact,
            truncated=truncated,
            meta=meta or {},
        )

    @classmethod
    def failure(
        cls,
        code: ToolCode,
        summary: str,
        *,
        data: dict[str, Any] | None = None,
        artifact: ArtifactRef | None = None,
        truncated: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=False,
            code=code,
            summary=summary,
            data=data or {},
            artifact=artifact,
            truncated=truncated,
            meta=meta or {},
        )

    @classmethod
    def noop(
        cls,
        summary: str,
        *,
        data: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            code="noop",
            summary=summary,
            data=data or {},
            meta=meta or {},
        )

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "summary": self.summary,
            "data": self.data,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "truncated": self.truncated,
        }

    def to_model_content(self) -> str:
        return json.dumps(
            self.to_model_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
