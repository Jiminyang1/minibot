"""Tool abstraction for MiniBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tool(ABC):
    """A single agent capability exposed through function calling."""

    def __init__(self, *, workspace: Path | None = None) -> None:
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
    def requires_approval(self) -> bool:
        """Whether this tool needs user confirmation before execution."""
        return False

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
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
