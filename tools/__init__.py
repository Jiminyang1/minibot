"""Tool exports for MiniBot."""

from __future__ import annotations

from pathlib import Path

from .base import Tool
from .exec_cmd import ExecTool
from .list_dir import ListDirTool
from .read_file import ReadFileTool
from .registry import ToolRegistry
from .search_files import SearchFilesTool
from .write_file import WriteFileTool


def create_default_registry(workspace: Path | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    ws = {"workspace": workspace} if workspace else {}
    registry.register(ExecTool(**ws))
    registry.register(ReadFileTool(**ws))
    registry.register(WriteFileTool(**ws))
    registry.register(ListDirTool(**ws))
    registry.register(SearchFilesTool(**ws))
    return registry


__all__ = [
    "Tool",
    "ToolRegistry",
    "ExecTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "SearchFilesTool",
    "create_default_registry",
]
