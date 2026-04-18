"""Structured tool outputs and materialized tool results for MiniBot."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, TypeAlias

from ..artifacts import ArtifactKind, ArtifactRef

ToolCode: TypeAlias = Literal[
    "success",
    "invalid_args",
    "not_found",
    "permission_denied",
    "timeout",
    "denied",
    "conflict",
    "error",
    "noop",
]


@dataclass(frozen=True)
class ToolOutput:
    """Raw semantic result produced by a tool before runtime materialization."""

    _MAX_DATA_CHARS = 8000

    ok: bool
    code: ToolCode
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    content: str | None = None
    content_kind: ArtifactKind = "text"
    content_name: str | None = None
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data:
            return
        size = len(json.dumps(self.data, ensure_ascii=False))
        if size > self._MAX_DATA_CHARS:
            raise ValueError(
                f"ToolOutput.data 序列化长度 {size} 超过上限 "
                f"{self._MAX_DATA_CHARS}；请把大正文放进 content，"
                "data 只保留结构化元信息与短字段。"
            )

    @classmethod
    def success(
        cls,
        summary: str,
        *,
        data: dict[str, Any] | None = None,
        content: str | None = None,
        content_kind: ArtifactKind = "text",
        content_name: str | None = None,
        truncated: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> "ToolOutput":
        return cls(
            ok=True,
            code="success",
            summary=summary,
            data=data or {},
            content=content,
            content_kind=content_kind,
            content_name=content_name,
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
        content: str | None = None,
        content_kind: ArtifactKind = "text",
        content_name: str | None = None,
        truncated: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> "ToolOutput":
        return cls(
            ok=False,
            code=code,
            summary=summary,
            data=data or {},
            content=content,
            content_kind=content_kind,
            content_name=content_name,
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
    ) -> "ToolOutput":
        return cls(
            ok=True,
            code="noop",
            summary=summary,
            data=data or {},
            meta=meta or {},
        )


@dataclass(frozen=True)
class ToolResult:
    """Final materialized envelope returned to the model."""

    ok: bool
    code: ToolCode
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifact: ArtifactRef | None = None
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

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
