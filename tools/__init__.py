"""Tool exports for MiniBot."""

from __future__ import annotations

from .base import Tool
from .exec_cmd import ExecTool
from .list_dir import ListDirTool
from .read_file import ReadFileTool
from .registry import ToolRegistry
from .search_files import SearchFilesTool
from .write_file import WriteFileTool


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ExecTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirTool())
    registry.register(SearchFilesTool())
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
