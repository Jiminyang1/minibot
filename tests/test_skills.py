from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.artifacts import ArtifactStore
from minibot.runtime.context_manager import ContextManager
from minibot.session import Session
from minibot.skills import SkillRegistry
from minibot.tools import (
    EditFileTool,
    ExecTool,
    FetchUrlTool,
    ForgetTool,
    ListDirTool,
    ReadArtifactTool,
    ReadFileTool,
    RememberTool,
    SearchFilesTool,
    WebSearchTool,
    WriteFileTool,
)
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.read_skill import ReadSkillTool
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput
from minibot.user_memory import UserMemoryStore


class _DummyTool(Tool):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        return ToolOutput.success("ok")


def _write_skill(
    directory: Path,
    *,
    name: str,
    description: str,
    tools: list[str],
    body: str,
) -> None:
    path = directory / f"{name}.md"
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        "tools:",
        *(f"  - {item}" for item in tools),
        "---",
        body,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_context_manager(
    *,
    registry: SkillRegistry,
    tool_registry: ToolRegistry,
) -> ContextManager:
    return ContextManager(
        base_system_prompt="BASE PROMPT",
        memory_store=None,
        skill_registry=registry,
        tool_registry=tool_registry,
        max_history_turns=10,
        compact_token_threshold=40000,
        reserved_completion_tokens=1000,
        compact_keep_recent=4,
        summarizer=lambda messages: "summary",
    )


class SkillCatalogTests(unittest.TestCase):
    def test_catalog_lists_skills_whose_tools_are_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _write_skill(
                directory,
                name="calendar",
                description="calendar skill",
                tools=["calendar_list_events"],
                body="calendar body",
            )
            _write_skill(
                directory,
                name="notes",
                description="notes skill",
                tools=["notes_search"],
                body="notes body",
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            manager = _make_context_manager(registry=registry, tool_registry=tool_registry)

            prompt = manager._build_request(
                session=Session("s_test"),
                user_input="any turn",
            ).messages[0]["content"]

            self.assertIn("## Available Skills", prompt)
            self.assertIn("- calendar: calendar skill | tools: calendar_list_events", prompt)
            self.assertNotIn("- notes:", prompt)
            self.assertIn("read_skill", prompt)

    def test_prompt_never_injects_skill_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _write_skill(
                directory,
                name="calendar",
                description="calendar skill",
                tools=["calendar_list_events"],
                body="CALENDAR-BODY-MARKER",
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            manager = _make_context_manager(registry=registry, tool_registry=tool_registry)

            prompt = manager._build_request(
                session=Session("s_test"),
                user_input="用 calendar 帮我安排 3 点的会议",
            ).messages[0]["content"]

            self.assertNotIn("CALENDAR-BODY-MARKER", prompt)
            self.assertNotIn("## Matched Skills", prompt)
            self.assertNotIn("Skill: calendar", prompt)

    def test_loader_accepts_frontmatter_without_closing_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "calendar.md").write_text(
                "\n".join(
                    [
                        "---",
                        "## name: calendar",
                        "description: calendar skill",
                        "tools:",
                        "  - calendar_list_events",
                        "",
                        "1. 确认时间范围",
                        "2. 创建事件",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            registry = SkillRegistry.from_directory(directory)

        skills = registry.list()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "calendar")
        self.assertEqual(skills[0].tools, ("calendar_list_events",))
        self.assertIn("1. 确认时间范围", skills[0].body)


class ReadSkillToolTests(unittest.TestCase):
    def _make_registry_with(self, tmpdir: Path) -> SkillRegistry:
        _write_skill(
            tmpdir,
            name="calendar",
            description="calendar skill",
            tools=["calendar_list_events"],
            body="步骤 1: 确认时间范围\n步骤 2: 创建事件",
        )
        return SkillRegistry.from_directory(tmpdir)

    def test_returns_full_body_for_known_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._make_registry_with(Path(tmpdir))
            tool = ReadSkillTool(registry)

            result = tool.execute(
                context=ToolExecutionContext(session_id="s_test"),
                name="calendar",
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.code, "success")
            self.assertEqual(result.data["name"], "calendar")
            self.assertEqual(result.data["description"], "calendar skill")
            self.assertEqual(result.data["tools"], ["calendar_list_events"])
            self.assertIn("步骤 1", result.data["body"])
            self.assertIn("步骤 2", result.data["body"])
            self.assertFalse(result.truncated)

    def test_trims_whitespace_in_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._make_registry_with(Path(tmpdir))
            tool = ReadSkillTool(registry)

            result = tool.execute(
                context=ToolExecutionContext(session_id="s_test"),
                name="  calendar  ",
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["name"], "calendar")

    def test_unknown_skill_returns_failure_with_available_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._make_registry_with(Path(tmpdir))
            tool = ReadSkillTool(registry)

            result = tool.execute(
                context=ToolExecutionContext(session_id="s_test"),
                name="unknown",
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "not_found")
            self.assertIn("calendar", result.data["available"])

    def test_empty_name_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = self._make_registry_with(Path(tmpdir))
            tool = ReadSkillTool(registry)

            result = tool.execute(
                context=ToolExecutionContext(session_id="s_test"),
                name="   ",
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "invalid_args")

    def test_large_body_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            giant_body = "x" * (ReadSkillTool._MAX_BODY_CHARS + 500)
            _write_skill(
                directory,
                name="big",
                description="big skill",
                tools=["big_tool"],
                body=giant_body,
            )
            registry = SkillRegistry.from_directory(directory)
            tool = ReadSkillTool(registry)

            result = tool.execute(
                context=ToolExecutionContext(session_id="s_test"),
                name="big",
            )

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertEqual(len(result.data["body"]), ReadSkillTool._MAX_BODY_CHARS)
            self.assertGreater(result.data["total_chars"], ReadSkillTool._MAX_BODY_CHARS)


class ToolLayerTests(unittest.TestCase):
    def test_core_local_tools_are_marked_as_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            artifact_store = ArtifactStore(workspace)
            memory_store = UserMemoryStore(root=workspace / "state")
            skills = SkillRegistry.from_directory(workspace / "skills")

            tools = [
                ReadFileTool(workspace=workspace),
                ReadArtifactTool(artifact_store),
                ReadSkillTool(skills),
                ListDirTool(workspace=workspace),
                SearchFilesTool(workspace=workspace),
                WriteFileTool(workspace=workspace),
                EditFileTool(workspace=workspace),
                ExecTool(workspace=workspace),
                RememberTool(memory_store),
                ForgetTool(memory_store),
            ]

            self.assertTrue(all(tool.layer == "kernel" for tool in tools))

    def test_concurrency_metadata_matches_v1_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            artifact_store = ArtifactStore(workspace)
            memory_store = UserMemoryStore(root=workspace / "state")
            skills = SkillRegistry.from_directory(workspace / "skills")

            concurrency_safe = [
                ReadFileTool(workspace=workspace),
                ReadArtifactTool(artifact_store),
                ReadSkillTool(skills),
                ListDirTool(workspace=workspace),
                SearchFilesTool(workspace=workspace),
            ]
            serial_only = [
                WriteFileTool(workspace=workspace),
                EditFileTool(workspace=workspace),
                ExecTool(workspace=workspace),
                RememberTool(memory_store),
                ForgetTool(memory_store),
                WebSearchTool(workspace=None),
                FetchUrlTool(),
            ]

            self.assertTrue(all(tool.concurrency_safe for tool in concurrency_safe))
            self.assertTrue(all(not tool.concurrency_safe for tool in serial_only))
            self.assertTrue(all(tool.exclusive for tool in serial_only))


if __name__ == "__main__":
    unittest.main()
