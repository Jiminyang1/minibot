"""Tool exports for MiniBot.

Tools are grouped into small factory functions (toolsets) that take
their own dependencies as arguments. The composition root (`__main__`)
decides which toolsets to wire in and hands them to `ToolRegistry`.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING

from ..macos import AppleScriptBridge
from ..user_memory import UserMemoryStore
from .base import Tool, ToolExecutionContext
from .edit_file import EditFileTool
from .exec_cmd import ExecTool
from .fetch_url import FetchUrlTool
from .list_dir import ListDirTool
from .memory_tools import ForgetTool, RememberTool
from .macos_apps import (
    CalendarCreateEventTool,
    CalendarListEventsTool,
    NotesAppendTool,
    NotesCreateTool,
    NotesSearchTool,
    RemindersCompleteTool,
    RemindersCreateTool,
    RemindersListTool,
)
from .read_artifact import ReadArtifactTool
from .read_file import ReadFileTool
from .registry import ToolRegistry
from .result import ArtifactRef, ToolResult
from .search_files import SearchFilesTool
from .web_search import WebSearchTool
from .write_file import WriteFileTool

if TYPE_CHECKING:
    from ..session import SessionManager


def filesystem_toolset(workspace: Path, session_manager: SessionManager) -> list[Tool]:
    """Read, write, edit, browse, and search files under *workspace*."""
    return [
        ReadFileTool(workspace=workspace, session_manager=session_manager),
        WriteFileTool(workspace=workspace),
        EditFileTool(workspace=workspace),
        ListDirTool(workspace=workspace),
        SearchFilesTool(workspace=workspace, session_manager=session_manager),
        ReadArtifactTool(session_manager),
    ]


def shell_toolset(workspace: Path, session_manager: SessionManager) -> list[Tool]:
    """Execute shell commands within *workspace*."""
    return [ExecTool(workspace=workspace, session_manager=session_manager)]


def network_toolset(session_manager: SessionManager) -> list[Tool]:
    """Network-facing tools such as search and webpage fetching."""
    return [
        WebSearchTool(workspace=None),
        FetchUrlTool(session_manager),
    ]


def memory_toolset(store: UserMemoryStore) -> list[Tool]:
    """Read and mutate global user memory backed by *store*."""
    return [RememberTool(store), ForgetTool(store)]


def macos_toolset() -> list[Tool]:
    """macOS builtin app tools backed by AppleScript."""
    if sys.platform != "darwin" or shutil.which("osascript") is None:
        return []
    bridge = AppleScriptBridge()
    return [
        CalendarListEventsTool(bridge),
        CalendarCreateEventTool(bridge),
        RemindersListTool(bridge),
        RemindersCreateTool(bridge),
        RemindersCompleteTool(bridge),
        NotesSearchTool(bridge),
        NotesCreateTool(bridge),
        NotesAppendTool(bridge),
    ]


__all__ = [
    "ArtifactRef",
    "Tool",
    "ToolExecutionContext",
    "ToolResult",
    "ToolRegistry",
    "ExecTool",
    "FetchUrlTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    "SearchFilesTool",
    "WebSearchTool",
    "CalendarListEventsTool",
    "CalendarCreateEventTool",
    "RemindersListTool",
    "RemindersCreateTool",
    "RemindersCompleteTool",
    "NotesSearchTool",
    "NotesCreateTool",
    "NotesAppendTool",
    "RememberTool",
    "ForgetTool",
    "filesystem_toolset",
    "shell_toolset",
    "network_toolset",
    "memory_toolset",
    "macos_toolset",
]
