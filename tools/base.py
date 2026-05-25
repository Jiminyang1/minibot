"""Tool abstraction for MiniBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Literal, TypeAlias

from .definitions import ModelToolDefinition
from .result import ToolOutput


ToolLayer: TypeAlias = Literal["kernel", "extension"]
ToolSource: TypeAlias = Literal["local", "mcp"]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime-only execution context for one tool call."""

    session_id: str
    run_id: str | None = None
    cancel_event: threading.Event | None = None


class Tool(ABC):
    """A single agent capability exposed through function calling."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
    ) -> None:
        self._workspace = workspace.resolve() if workspace else None

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
    def layer(self) -> ToolLayer:
        """Architectural layer for the tool."""
        return "extension"

    @property
    def source(self) -> ToolSource:
        """High-level source category for UI/logging."""
        return "local"

    @property
    def display_name(self) -> str:
        """Human-friendly tool label for logs."""
        return self.name

    @property
    def requires_approval(self) -> bool:
        """Whether this tool needs user confirmation before execution."""
        return False

    @property
    def read_only(self) -> bool:
        """Whether this tool is side-effect free and safe to batch."""
        return False

    @property
    def exclusive(self) -> bool:
        """Whether this tool should always run alone."""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """Whether this tool may run with other concurrency-safe tools."""
        return self.read_only and not self.exclusive

    @abstractmethod
    def execute(self, *, context: ToolExecutionContext, **kwargs: Any) -> ToolOutput:
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

    def to_model_definition(self) -> ModelToolDefinition:
        """Return a provider-agnostic runtime tool definition."""
        return ModelToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
