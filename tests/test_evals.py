from __future__ import annotations

from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.evals.runner import evaluate, format_table, load_cases
from minibot.evals.sandbox import macos_tool_specs, run_case
from minibot.evals.scoring import _parse_judgement, resolve_date, score_deterministic
from minibot.llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from minibot.runtime.messages import ModelMessage
from minibot.tools.definitions import ModelToolDefinition

TODAY = date(2026, 10, 1)  # a Thursday


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[list[ModelMessage]] = []

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))
        return self._responses.pop(0)


def _call(call_id: str, tool: str, **args) -> ToolCall:
    return ToolCall(id=call_id, name=tool, arguments=json.dumps(args, ensure_ascii=False))


def _sandbox_env():
    # build_runtime constructs a provider client (never called here) and must
    # not pick up real Langfuse keys from the developer's shell.
    return mock.patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key", "MINIBOT_LANGFUSE": "0"}
    )


class ScoringTests(unittest.TestCase):
    def test_relative_dates(self) -> None:
        self.assertEqual(resolve_date("+1d", TODAY), date(2026, 10, 2))
        # "下周三" from a Thursday is the Wednesday of the next Mon-start week.
        self.assertEqual(resolve_date("nextweek:wed", TODAY), date(2026, 10, 7))
        self.assertEqual(resolve_date("nextweek:wed", date(2026, 10, 5)), date(2026, 10, 14))

    def test_stages_pass_and_fail_independently(self) -> None:
        expected = {
            "skill": "reminders",
            "tools_required": ["reminders_create"],
            "tools_forbidden": ["exec"],
            "args": {"reminders_create": {"due_at": {"date": "+1d", "time": "15:00"}}},
        }
        good = {
            "skills_read": ["reminders"],
            "tool_calls": [
                {"name": "read_skill", "args": {"name": "reminders"}},
                {"name": "reminders_create", "args": {"due_at": "2026-10-02T15:00:00+08:00"}},
            ],
        }
        wrong_time = {
            **good,
            "tool_calls": [
                {"name": "reminders_create", "args": {"due_at": "2026-10-02 03:00"}},
                {"name": "exec", "args": {"command": "date"}},
            ],
        }

        passed = {s.name: s.value for s in score_deterministic(good, expected, today=TODAY)}
        failed = {s.name: s for s in score_deterministic(wrong_time, expected, today=TODAY)}

        self.assertEqual(passed, {"skill_selection": 1.0, "tool_selection": 1.0, "tool_args": 1.0})
        self.assertEqual(failed["skill_selection"].value, 1.0)
        self.assertEqual(failed["tool_selection"].value, 0.0)
        self.assertIn("exec", failed["tool_selection"].comment)
        self.assertEqual(failed["tool_args"].value, 0.0)
        self.assertIn("15:00", failed["tool_args"].comment)

    def test_no_tools_and_no_skill_expectations(self) -> None:
        expected = {"skill": None, "tools_forbidden": ["*"]}
        scores = score_deterministic(
            {"skills_read": [], "tool_calls": [{"name": "web_search", "args": {}}]},
            expected,
            today=TODAY,
        )
        values = {s.name: s.value for s in scores}
        self.assertEqual(values, {"skill_selection": 1.0, "tool_selection": 0.0})

    def test_end_to_end_requires_every_stage_and_a_clean_run(self) -> None:
        expected = {"tools_required": ["read_file"], "answer_rubric": "x"}
        output = {"tool_calls": [{"name": "read_file", "args": {}}], "reply": "ok"}

        scores = evaluate(output, {"prompt": "p"}, expected, judge=None, today=TODAY)
        crashed = evaluate({**output, "error": "RuntimeError: boom"}, {"prompt": "p"}, expected, judge=None, today=TODAY)

        self.assertEqual(scores[-1].name, "end_to_end")
        self.assertEqual(scores[-1].value, 1.0)
        self.assertEqual(crashed[-1].value, 0.0)

    def test_judge_output_parsing(self) -> None:
        self.assertEqual(_parse_judgement('{"score": 0.5, "reason": "部分"}').value, 0.5)
        self.assertEqual(_parse_judgement('结论：```json\n{"score": 1, "reason": "ok"}\n```').value, 1.0)
        self.assertEqual(_parse_judgement("看起来不错").value, 0.0)

    def test_golden_cases_are_well_formed(self) -> None:
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 8)
        tool_names = {f"mcp__macos_system__{spec.remote_name}" for spec in macos_tool_specs()}
        known = tool_names | {
            "read_file", "write_file", "edit_file", "list_dir", "search_files",
            "read_artifact", "exec", "web_search", "fetch_url", "remember", "forget",
            "read_skill", "search_history", "schedule_task", "list_scheduled_tasks",
            "cancel_scheduled_task", "*",
        }
        for case in cases:
            expected = case["expected"]
            mentioned = (
                set(expected.get("tools_required", []))
                | set(expected.get("tools_forbidden", []))
                | set(expected.get("args", {}))
            )
            self.assertLessEqual(mentioned, known, case["id"])
            self.assertTrue(case["input"]["prompt"], case["id"])


class SandboxTests(unittest.TestCase):
    def test_real_turn_runs_in_sandbox_and_scores(self) -> None:
        due = f"{TODAY + timedelta(days=1):%Y-%m-%d}T15:00:00"
        llm = _ScriptedLLM(
            [
                LLMResponse(content="", tool_calls=[_call("c1", "read_skill", name="reminders")]),
                LLMResponse(
                    content="",
                    tool_calls=[
                        _call("c2", "mcp__macos_system__reminders_create", title="交周报", due_at=due)
                    ],
                    usage=TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110),
                ),
                LLMResponse(content="已创建提醒：明天 15:00 交周报。"),
            ]
        )
        case = {"prompt": "明天下午三点提醒我交周报。"}

        with _sandbox_env():
            output = run_case(case, config=Config(model="deepseek-v4-pro"), llm=llm)

        self.assertIsNone(output["error"])
        self.assertEqual(output["reply"], "已创建提醒：明天 15:00 交周报。")
        self.assertEqual(output["skills_read"], ["reminders"])
        create = output["tool_calls"][1]
        self.assertEqual(create["name"], "mcp__macos_system__reminders_create")
        self.assertTrue(create["ok"])
        self.assertTrue(create["approved"])
        # The model saw the production reminders tool schema and skill catalog.
        system_prompt = llm.requests[0][0].content
        self.assertIn("reminders", system_prompt)

        expected = load_cases(["reminders-create-relative-time"])[0]["expected"]
        scores = {s.name: s.value for s in evaluate(output, case, expected, judge=None, today=TODAY)}
        # The happy path no longer requires the skill; the stage is skipped.
        self.assertNotIn("skill_selection", scores)
        self.assertEqual(scores["tool_args"], 1.0)
        self.assertEqual(scores["end_to_end"], 1.0)

    def test_side_effects_never_leave_the_sandbox(self) -> None:
        llm = _ScriptedLLM(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        _call("c1", "exec", command="touch escaped.txt"),
                        _call("c2", "mcp__macos_system__calendar_create_event",
                              title="x", start_at="2026-10-02T10:00", end_at="2026-10-02T11:00"),
                    ],
                ),
                LLMResponse(content="好的。"),
            ]
        )
        case = {"prompt": "do it", "files": {"keep.txt": "keep\n"}, "approval": "deny"}

        with _sandbox_env():
            output = run_case(case, config=Config(model="deepseek-v4-pro"), llm=llm)

        # Denied by the case's approval policy, and exec is a stand-in anyway.
        self.assertEqual([c["approved"] for c in output["tool_calls"]], [False, False])
        self.assertEqual(output["files"], {"keep.txt": "keep\n"})
        self.assertFalse((Path.cwd() / "escaped.txt").exists())

    def test_scripted_failure_then_success_drives_recovery(self) -> None:
        create = "mcp__macos_system__reminders_create"
        llm = _ScriptedLLM(
            [
                LLMResponse(content="", tool_calls=[_call("c1", create, title="打电话")]),
                LLMResponse(content="", tool_calls=[_call("c2", "exec", command="open -a Reminders")]),
                LLMResponse(content="", tool_calls=[_call("c3", create, title="打电话")]),
                LLMResponse(content="已创建。"),
            ]
        )
        case = {
            "prompt": "提醒我打电话",
            "tool_results": {
                create: {"sequence": [{"error": "Application isn’t running. (-600)"}, None]}
            },
        }

        with _sandbox_env():
            output = run_case(case, config=Config(model="deepseek-v4-pro"), llm=llm)

        self.assertEqual(
            [(c["name"], c["ok"]) for c in output["tool_calls"]],
            [(create, False), ("exec", True), (create, True)],
        )
        # The model saw the real-shaped MCP error text, and nothing hinting
        # at a test harness.
        failed_result = llm.requests[1][-1].content
        self.assertIn("-600", failed_result)
        exec_result = llm.requests[2][-1].content
        self.assertNotIn("沙箱", exec_result)

    def test_seeded_memory_and_files_are_visible_to_the_turn(self) -> None:
        llm = _ScriptedLLM([LLMResponse(content="ok")])
        case = {"prompt": "hi", "memory": ["用户在星海科技工作。"], "files": {"a/b.txt": "x"}}

        with _sandbox_env():
            output = run_case(case, config=Config(model="deepseek-v4-pro"), llm=llm)

        system_prompt = llm.requests[0][0].content
        self.assertIn("id: mem_1; fact: 用户在星海科技工作。", system_prompt)
        self.assertEqual(output["files"], {"a/b.txt": "x"})

    def test_report_marks_skipped_and_failed_stages(self) -> None:
        rows = [
            {"case": "a", "scores": {"tool_selection": (1.0, ""), "end_to_end": (1.0, "")}},
            {"case": "b", "scores": {"tool_selection": (0.0, "缺少 ['x']"), "end_to_end": (0.0, "")}},
        ]
        report = format_table(rows)
        self.assertIn("50%", report)
        self.assertIn("b · tool_selection: 缺少 ['x']", report)


if __name__ == "__main__":
    unittest.main()
