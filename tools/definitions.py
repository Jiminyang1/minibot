"""Provider-agnostic tool definition types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelToolDefinition:
    """Internal tool definition used by MiniBot runtime and LLM clients."""

    name: str
    description: str
    parameters: dict[str, Any]
