from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.artifacts import ArtifactStore
from minibot.runtime.context_builder import ContextBuilder
from minibot.session import Session
from minibot.skills import SkillRegistry
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.edit_file import EditFileTool
from minibot.tools.exec_cmd import ExecTool
from minibot.tools.fetch_url import FetchUrlTool
from minibot.tools.list_dir import ListDirTool
from minibot.tools.memory_tools import ForgetTool, RememberTool
from minibot.tools.read_artifact import ReadArtifactTool
from minibot.tools.read_file import ReadFileTool
from minibot.tools.read_skill import ReadSkillTool
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput
from minibot.tools.search_files import SearchFilesTool
from minibot.tools.web_search import WebSearchTool
from minibot.tools.write_file import WriteFileTool
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


def _make_context_builder(
    *,
    registry: SkillRegistry,
    tool_registry: ToolRegistry,
    memory_store=None,
    now_provider=None,
) -> ContextBuilder:
    return ContextBuilder(
        base_system_prompt="BASE PROMPT",
        memory_store=memory_store,
        skill_registry=registry,
        tool_registry=tool_registry,
        now_provider=now_provider,
    )


class SkillCatalogTests(unittest.TestCase):
    def test_prompt_includes_local_time_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _write_skill(
                directory,
                name="calendar",
                description="calendar skill",
                tools=["calendar_list_events"],
                body="calendar body",
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            builder = _make_context_builder(
                registry=registry,
                tool_registry=tool_registry,
                now_provider=lambda: datetime(
                    2026,
                    4,
                    23,
                    13,
                    2,
                    5,
                    tzinfo=timezone(timedelta(hours=8)),
                ),
            )

            prompt = builder.build(Session("s_test").messages).messages[0].content

            self.assertIn("## Local Time Context", prompt)
            self.assertIn("now_local: 2026-04-23T13:02:05+08:00", prompt)
            self.assertIn("today_local: 2026-04-23", prompt)
            self.assertIn("timezone_local:", prompt)

    def test_memory_context_uses_date_free_ids_after_time_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            memory_store = UserMemoryStore(root=directory / "state")
            item = memory_store.add(
                "用户将于 2026 年 7 月 8 日加入深圳 Amazon AGL 团队。"
            )
            registry = SkillRegistry.from_directory(directory / "skills")
            builder = _make_context_builder(
                registry=registry,
                tool_registry=ToolRegistry(),
                memory_store=memory_store,
                now_provider=lambda: datetime(
                    2026,
                    7,
                    7,
                    13,
                    0,
                    0,
                    tzinfo=timezone(timedelta(hours=8)),
                ),
            )

            prompt = builder.build(Session("s_test").messages).messages[0].content

            self.assertLess(
                prompt.index("## Local Time Context"),
                prompt.index("## User Memory Data"),
            )
            self.assertIn("today_local: 2026-07-07", prompt)
            self.assertIn("id: mem_1", prompt)
            self.assertEqual(item.id, "mem_1")
            self.assertIn(item.id, prompt)

    def test_packaged_skills_include_drawio(self) -> None:
        package_skills_dir = Path(__file__).resolve().parents[1] / "skills"
        registry = SkillRegistry.from_directory(package_skills_dir)

        drawio = registry.get_by_name("drawio")
        self.assertIsNotNone(drawio)
        assert drawio is not None
        self.assertIn("write_file", drawio.tools)
        self.assertIn("exec", drawio.tools)

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
            builder = _make_context_builder(registry=registry, tool_registry=tool_registry)

            prompt = builder.build(Session("s_test").messages).messages[0].content

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
            builder = _make_context_builder(registry=registry, tool_registry=tool_registry)

            prompt = builder.build(Session("s_test").messages).messages[0].content

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

    def test_memory_ids_are_never_reused_after_forget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserMemoryStore(root=Path(tmpdir) / "state")
            store.add("第一条")
            second = store.add("第二条")
            store.delete(second.id)  # forget the highest id

            third = store.add("第三条")

            # A stale reference to mem_2 must not resolve to a newer fact.
            self.assertNotEqual(third.id, second.id)
            self.assertEqual(third.id, "mem_3")
            # The counter survives a reload from disk.
            reloaded = UserMemoryStore(root=Path(tmpdir) / "state")
            reloaded.delete(third.id)
            fourth = reloaded.add("第四条")
            self.assertEqual(fourth.id, "mem_4")

    def test_forget_deletes_date_free_memory_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_store = UserMemoryStore(root=workspace / "state")
            old = memory_store.add("旧事实")
            newest = memory_store.add("新事实")
            tool = ForgetTool(memory_store)

            output = tool.execute(
                context=ToolExecutionContext(session_id="s_test"),
                memory_id=newest.id,
            )

            self.assertTrue(output.ok)
            self.assertEqual(output.data["memory_id"], newest.id)
            self.assertEqual([item.id for item in memory_store.list()], [old.id])

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
