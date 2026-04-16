"""Tool abstraction for MiniBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """A single agent capability exposed through function calling."""

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

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool with named parameters."""

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
