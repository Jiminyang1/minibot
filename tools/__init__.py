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

from ..artifacts import ArtifactRef, ArtifactStore
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
from .read_skill import ReadSkillTool
from .registry import ToolRegistry
from .result import ToolOutput, ToolResult
from .search_files import SearchFilesTool
from .web_search import WebSearchTool
from .write_file import WriteFileTool

if TYPE_CHECKING:
    from ..skills import SkillRegistry


def filesystem_toolset(workspace: Path, artifact_store: ArtifactStore) -> list[Tool]:
    """Read, write, edit, browse, and search files under *workspace*."""
    return [
        ReadFileTool(workspace=workspace),
        WriteFileTool(workspace=workspace),
        EditFileTool(workspace=workspace),
        ListDirTool(workspace=workspace),
        SearchFilesTool(workspace=workspace),
        ReadArtifactTool(artifact_store),
    ]


def shell_toolset(workspace: Path) -> list[Tool]:
    """Execute shell commands within *workspace*."""
    return [ExecTool(workspace=workspace)]


def network_toolset() -> list[Tool]:
    """Network-facing tools such as search and webpage fetching."""
    return [
        WebSearchTool(workspace=None),
        FetchUrlTool(),
    ]


def memory_toolset(store: UserMemoryStore) -> list[Tool]:
    """Read and mutate global user memory backed by *store*."""
    return [RememberTool(store), ForgetTool(store)]


def skill_toolset(skill_registry: SkillRegistry) -> list[Tool]:
    """Expose on-demand skill-body loading via ``read_skill``."""
    return [ReadSkillTool(skill_registry)]


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
    "ArtifactStore",
    "Tool",
    "ToolExecutionContext",
    "ToolOutput",
    "ToolResult",
    "ToolRegistry",
    "ExecTool",
    "FetchUrlTool",
    "ReadFileTool",
    "ReadSkillTool",
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
    "skill_toolset",
]
