"""Tool exports for MiniBot.

Tools are grouped into small factory functions (toolsets) that take
their own dependencies as arguments. The composition root (`__main__`)
decides which toolsets to wire in and hands them to `ToolRegistry`.
"""

from __future__ import annotations

from pathlib import Path

from ..user_memory import UserMemoryStore
from .base import Tool
from .edit_file import EditFileTool
from .exec_cmd import ExecTool
from .list_dir import ListDirTool
from .memory_tools import ForgetTool, RememberTool
from .read_file import ReadFileTool
from .registry import ToolRegistry
from .search_files import SearchFilesTool
from .write_file import WriteFileTool


def filesystem_toolset(workspace: Path) -> list[Tool]:
    """Read, write, edit, browse, and search files under *workspace*."""
    return [
        ReadFileTool(workspace=workspace),
        WriteFileTool(workspace=workspace),
        EditFileTool(workspace=workspace),
        ListDirTool(workspace=workspace),
        SearchFilesTool(workspace=workspace),
    ]


def shell_toolset(workspace: Path) -> list[Tool]:
    """Execute shell commands within *workspace*."""
    return [ExecTool(workspace=workspace)]


def memory_toolset(store: UserMemoryStore) -> list[Tool]:
    """Read and mutate global user memory backed by *store*."""
    return [RememberTool(store), ForgetTool(store)]


__all__ = [
    "Tool",
    "ToolRegistry",
    "ExecTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    "SearchFilesTool",
    "RememberTool",
    "ForgetTool",
    "filesystem_toolset",
    "shell_toolset",
    "memory_toolset",
]
