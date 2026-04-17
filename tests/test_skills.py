from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.context_manager import ContextManager, _estimate_tokens
from minibot.session import Session
from minibot.skills import SkillRegistry
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolResult


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

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolResult:
        del context, kwargs
        return ToolResult.success("ok")


class SkillsTests(unittest.TestCase):
    def _write_skill(
        self,
        directory: Path,
        *,
        name: str,
        triggers: list[str],
        tools: list[str],
        summary: str,
        body: str,
    ) -> None:
        path = directory / f"{name}.md"
        lines = [
            "---",
            f"name: {name}",
            f"description: {name} skill",
            "triggers:",
            *(f"  - {item}" for item in triggers),
            "tools:",
            *(f"  - {item}" for item in tools),
            f"summary: {summary}",
            "---",
            body,
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _make_context_manager(
        self,
        *,
        registry: SkillRegistry,
        tool_registry: ToolRegistry,
        max_skill_tokens: int = 800,
        max_full_skill_tokens: int = 500,
        max_summary_skill_tokens: int = 150,
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
            max_skill_tokens=max_skill_tokens,
            max_full_skill_tokens=max_full_skill_tokens,
            max_summary_skill_tokens=max_summary_skill_tokens,
        )

    def test_no_skill_injected_on_unrelated_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_skill(
                directory,
                name="calendar",
                triggers=["calendar", "日历"],
                tools=["calendar_list_events"],
                summary="summary",
                body="body",
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            manager = self._make_context_manager(registry=registry, tool_registry=tool_registry)

            prompt = manager._build_request(
                session=Session("s_test"),
                user_input="帮我解释一下 transformer attention",
            )

            self.assertIn("## Available Skills", prompt.messages[0]["content"])
            self.assertNotIn("## Matched Skills", prompt.messages[0]["content"])
            self.assertEqual(prompt.matched_skills, [])

    def test_only_registered_tool_skills_can_inject(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_skill(
                directory,
                name="calendar",
                triggers=["calendar", "日历"],
                tools=["calendar_list_events"],
                summary="calendar summary",
                body="calendar body",
            )
            self._write_skill(
                directory,
                name="notes",
                triggers=["notes", "笔记"],
                tools=["notes_search"],
                summary="notes summary",
                body="notes body",
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            manager = self._make_context_manager(registry=registry, tool_registry=tool_registry)

            prompt = manager._build_request(
                session=Session("s_test"),
                user_input="帮我看一下 calendar 和 notes",
            ).messages[0]["content"]

            self.assertIn("## Available Skills", prompt)
            self.assertIn("- calendar: calendar skill | tools: calendar_list_events", prompt)
            self.assertNotIn("- notes: notes skill | tools: notes_search", prompt)
            self.assertIn("Skill: calendar", prompt)
            self.assertNotIn("Skill: notes", prompt)

    def test_top_skill_full_body_requires_threshold_or_explicit_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_skill(
                directory,
                name="calendar",
                triggers=["calendar", "日历", "schedule"],
                tools=["calendar_list_events"],
                summary="calendar summary",
                body="calendar detailed body",
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            manager = self._make_context_manager(registry=registry, tool_registry=tool_registry)

            summary_only_prompt = manager._build_request(
                session=Session("s_test"),
                user_input="please schedule",
            ).messages[0]["content"]
            full_prompt = manager._build_request(
                session=Session("s_test"),
                user_input="please use calendar to schedule",
            )

            self.assertIn("Skill: calendar", summary_only_prompt)
            self.assertNotIn("详细流程:\ncalendar detailed body", summary_only_prompt)
            self.assertIn("详细流程:\ncalendar detailed body", full_prompt.messages[0]["content"])
            self.assertEqual(
                [(item.name, item.mode) for item in full_prompt.matched_skills],
                [("calendar", "full")],
            )

    def test_progressive_injection_respects_order_and_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_skill(
                directory,
                name="calendar",
                triggers=["calendar", "schedule"],
                tools=["calendar_list_events"],
                summary="calendar summary " * 20,
                body="calendar body " * 120,
            )
            self._write_skill(
                directory,
                name="notes",
                triggers=["notes"],
                tools=["notes_search"],
                summary="notes summary " * 30,
                body="notes body " * 120,
            )
            registry = SkillRegistry.from_directory(directory)
            tool_registry = ToolRegistry()
            tool_registry.register(_DummyTool("calendar_list_events"))
            tool_registry.register(_DummyTool("notes_search"))
            manager = self._make_context_manager(
                registry=registry,
                tool_registry=tool_registry,
                max_skill_tokens=140,
                max_full_skill_tokens=100,
                max_summary_skill_tokens=40,
            )

            prompt = manager._build_request(
                session=Session("s_test"),
                user_input="calendar schedule notes",
            ).messages[0]["content"]
            skill_block, matched_skills = manager._render_skill_block(
                user_input="calendar schedule notes"
            )

            self.assertIn("Skill: calendar", prompt)
            self.assertIn("Skill: notes", prompt)
            self.assertLess(prompt.index("Skill: calendar"), prompt.index("Skill: notes"))
            self.assertLessEqual(_estimate_tokens(skill_block), 140)
            self.assertEqual(
                [(item.name, item.mode) for item in matched_skills],
                [("calendar", "full"), ("notes", "summary")],
            )


if __name__ == "__main__":
    unittest.main()
