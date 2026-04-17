"""Shared artifact reference types.

These types are neutral contracts used by both session persistence and
tool results. Keeping them out of ``tools/`` avoids a reverse
dependency from ``session`` back into the tool layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

ArtifactKind: TypeAlias = Literal["text", "json", "file", "diff"]


@dataclass(frozen=True)
class ArtifactRef:
    """A stable reference to a stored tool artifact."""

    id: str
    kind: ArtifactKind
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
