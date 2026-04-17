"""Tool abstraction for MiniBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from .result import ToolResult

if TYPE_CHECKING:
    from ..session import SessionManager


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime-only execution context for one tool call."""

    session_id: str


class Tool(ABC):
    """A single agent capability exposed through function calling."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self._workspace = workspace.resolve() if workspace else None
        self._session_manager = session_manager

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name used by the model."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description for the model."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool's parameters."""

    @property
    def requires_approval(self) -> bool:
        """Whether this tool needs user confirmation before execution."""
        return False

    @abstractmethod
    def execute(self, *, context: ToolExecutionContext, **kwargs: Any) -> ToolResult:
        """Run the tool with named parameters."""

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace and verify it stays inside."""
        if self._workspace is None:
            return Path(path).resolve()
        resolved = (self._workspace / path).resolve()
        if not resolved.is_relative_to(self._workspace):
            raise PermissionError(
                f"路径 {path} 超出工作目录 {self._workspace}"
            )
        return resolved

    def _require_session_manager(self) -> SessionManager:
        if self._session_manager is None:
            raise RuntimeError("当前工具未配置 SessionManager。")
        return self._session_manager

    def to_definition(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
