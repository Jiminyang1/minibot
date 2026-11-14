"""Per-stage golden assertions, failure localization and the pytest gate.

Fake runs and judges only: nothing here calls a model or the network.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.evals import __main__ as cli
from minibot.evals.conftest import golden_requested, pytest_ignore_collect
from minibot.evals.golden import GoldenSession
from minibot.evals.report import failure_report, format_failure_report, save_failure_report
from minibot.evals.runner import STAGES, applicable_stages, dataset_cases, run_experiment
from minibot.evals.sandbox import run_case
from minibot.llm import LLMClient, LLMResponse, ToolCall
from minibot.runtime.messages import ModelMessage
from minibot.tools.definitions import ModelToolDefinition

TODAY = date(2026, 10, 1)
CASE = {
    "id": "file-read-answer",
    "input": {"prompt": "settings.py 里默认模型是什么？"},
    "expected": {
        "skill": None,
        "tools_required": ["read_file"],
        "args": {"read_file": {"path": {"contains": "settings.py"}}},
        "answer_rubric": "回答默认模型是 orca-7b。",
    },
}


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


class _FakeLangfuse:
    def __init__(self) -> None:
        self.spans: list[str] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, *, name, input=None):
        self.spans.append(name)
        yield SimpleNamespace(trace_id=f"trace-{len(self.spans)}", update=lambda **_: None)

    def get_trace_url(self, *, trace_id):
        return f"https://lf.example/project/p/traces/{trace_id}"

    def flush(self) -> None:
        self.flushed = True


def _fake_run(path: str):
    calls = []

    def run(case_input, *, config, llm=None):
        calls.append(case_input["prompt"])
        return {
            "reply": "默认模型是 orca-7b。",
            "error": None,
            "skills_read": [],
            "tool_calls": [{"name": "read_file", "args": {"path": path}, "ok": True}],
        }

    return run, calls


class GoldenSessionTests(unittest.TestCase):
    def test_runs_each_case_once_and_judges_only_when_asked(self) -> None:
        run, calls = _fake_run("settings.py")
        judge = _ScriptedLLM([LLMResponse(content='{"score": 1, "reason": "答对了"}')])
        session = GoldenSession(Config(model="m"), judge=judge, run=run, today=TODAY)

        tool_args = session.check(CASE, "tool_args")
        selection = session.check(CASE, "tool_selection")

        self.assertTrue(tool_args.passed and selection.passed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(judge.requests, [])  # deterministic stages need no judge

        self.assertTrue(session.check(CASE, "answer").passed)
        self.assertTrue(session.check(CASE, "end_to_end").passed)
        self.assertEqual(len(judge.requests), 1)  # cached between answer and e2e
        self.assertEqual(len(calls), 1)

    def test_unspecified_stages_are_skipped_not_passed(self) -> None:
        run, calls = _fake_run("settings.py")
        session = GoldenSession(Config(model="m"), judge=None, run=run, today=TODAY)

        self.assertIsNone(session.check(CASE, "outcome"))
        self.assertEqual(calls, [])  # skipping never costs a model turn
        self.assertEqual(
            applicable_stages(CASE["expected"]),
            ["skill_selection", "tool_selection", "tool_args", "answer", "end_to_end"],
        )
        self.assertEqual(applicable_stages({}), ["end_to_end"])

    def test_failure_message_names_stage_comment_and_trace(self) -> None:
        run, _ = _fake_run("other.py")
        langfuse = _FakeLangfuse()
        session = GoldenSession(Config(model="m"), judge=None, langfuse=langfuse, run=run, today=TODAY)

        result = session.check(CASE, "tool_args")
        e2e = session.check(CASE, "end_to_end")
        session.close()

        self.assertFalse(result.passed)
        self.assertIn("file-read-answer · tool_args = 0", result.message)
        self.assertIn("path 应包含 'settings.py'", result.message)
        self.assertIn("trace: https://lf.example/project/p/traces/trace-1", result.message)
        self.assertFalse(e2e.passed)
        self.assertEqual(langfuse.spans, ["golden-pytest:file-read-answer"])
        self.assertTrue(langfuse.flushed)

        local = GoldenSession(Config(model="m"), judge=None, run=run, today=TODAY)
        self.assertIn("trace: 无", local.check(CASE, "tool_args").message)


class FailureReportTests(unittest.TestCase):
    ROWS = [
        {"case": "ok", "scores": {"tool_selection": (1.0, ""), "end_to_end": (1.0, "")}, "output": {}},
        {
            "case": "wrong-tool",
            "scores": {
                "skill_selection": (1.0, "未读取 skill"),
                "tool_selection": (0.0, "缺少 ['read_file']"),
                "answer": (0.0, "瞎猜"),
                "end_to_end": (0.0, "未通过"),
            },
            "output": {},
            "trace_id": "t2",
            "trace_url": "https://lf/t2",
        },
        {
            "case": "half-answer",
            "scores": {"answer": (0.5, "部分"), "outcome": (0.0, "a.py: 文件不存在"), "end_to_end": (0.0, "")},
            "output": {},
        },
        {
            "case": "crashed",
            "scores": {"end_to_end": (0.0, "运行失败: boom")},
            "output": {"error": "RuntimeError: boom"},
        },
    ]

    def test_first_failing_stage_in_pipeline_order(self) -> None:
        report = failure_report(self.ROWS, run_name="r1", mode="langfuse")

        self.assertEqual((report["total"], report["passed"]), (4, 1))
        first = {f["case"]: (f["stage"], f["comment"]) for f in report["failures"]}
        self.assertEqual(first["wrong-tool"], ("tool_selection", "缺少 ['read_file']"))
        # ½ passes the answer stage, so the outcome is what broke.
        self.assertEqual(first["half-answer"], ("outcome", "a.py: 文件不存在"))
        self.assertEqual(first["crashed"], ("run", "RuntimeError: boom"))
        counts = report["stage_failures"]
        self.assertEqual(counts["tool_selection"], {"failed": 1, "first": 1})
        self.assertEqual(counts["answer"], {"failed": 1, "first": 0})
        self.assertEqual(counts["run"], {"failed": 1, "first": 1})

        text = format_failure_report(report)
        self.assertIn("wrong-tool → tool_selection: 缺少 ['read_file']", text)
        self.assertIn("trace: https://lf/t2", text)

    def test_local_mode_says_there_is_no_trace_and_saves_json(self) -> None:
        report = failure_report(self.ROWS, run_name="r2", mode="local")
        self.assertIn("trace: 无（本地模式", format_failure_report(report))
        self.assertNotIn("run", failure_report(self.ROWS[:2], run_name="r", mode="local")["stage_failures"])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_failure_report(report, out_dir=Path(tmp))
            self.assertEqual(path.name, "r2.failures.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["failures"][0]["case"], "wrong-tool")

    def test_run_command_prints_and_saves_the_report(self) -> None:
        printed: list[str] = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp}),
            mock.patch.object(cli, "load_env"),
            mock.patch.object(cli, "run_local", return_value=self.ROWS),
            mock.patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))),
        ):
            self.assertEqual(cli.main(["run", "--local", "--name", "r3"]), 0)
            saved = json.loads((Path(tmp) / "evals" / "r3.failures.json").read_text(encoding="utf-8"))
        output = "\n".join(printed)
        self.assertIn("crashed → run: RuntimeError: boom", output)
        self.assertIn("各阶段失败数", output)
        self.assertEqual(saved["mode"], "local")
        self.assertEqual(len(saved["failures"]), 3)

    def test_all_passing(self) -> None:
        report = failure_report(self.ROWS[:1], run_name="r", mode="local")
        self.assertIn("全部端到端通过", format_failure_report(report))


class ExperimentTraceTests(unittest.TestCase):
    def test_experiment_rows_carry_each_items_trace(self) -> None:
        item = SimpleNamespace(id="minibot-golden--a", status="ACTIVE", metadata={"case_id": "a"},
                               input={"prompt": "p"}, expected_output={})
        archived = SimpleNamespace(id="minibot-golden--old", status="ARCHIVED", metadata={},
                                   input={}, expected_output={})
        item_result = SimpleNamespace(item=item, output={"reply": "x"}, trace_id="tr-1",
                                      evaluations=[SimpleNamespace(name="end_to_end", value=1.0, comment="")])
        langfuse = mock.MagicMock()
        langfuse.get_dataset.return_value = SimpleNamespace(items=[item, archived])
        langfuse.run_experiment.return_value = SimpleNamespace(item_results=[item_result])
        langfuse.get_trace_url.return_value = "https://lf/tr-1"

        with mock.patch("minibot.evals.runner.build_judge"):
            rows = run_experiment(langfuse, config=Config(model="m"), case_ids=None, run_name="r")

        self.assertEqual((rows[0]["trace_id"], rows[0]["trace_url"]), ("tr-1", "https://lf/tr-1"))
        self.assertEqual(
            dataset_cases(langfuse),
            [{"id": "a", "input": {"prompt": "p"}, "expected": {}, "dataset_item_id": "minibot-golden--a"}],
        )
        with self.assertRaises(ValueError):
            dataset_cases(langfuse, ["missing"])


class PytestGateTests(unittest.TestCase):
    @staticmethod
    def _config(markexpr: str):
        return SimpleNamespace(getoption=lambda name, default=None: markexpr if name == "markexpr" else default)

    def test_golden_tests_are_collected_only_when_selected(self) -> None:
        self.assertFalse(golden_requested(self._config("")))
        self.assertFalse(golden_requested(self._config("not golden")))
        self.assertTrue(golden_requested(self._config("golden")))
        self.assertTrue(golden_requested(self._config("golden and not slow")))
        golden_file = Path("evals/test_golden.py")
        self.assertTrue(pytest_ignore_collect(golden_file, self._config("")))
        self.assertIsNone(pytest_ignore_collect(golden_file, self._config("golden")))
        self.assertIsNone(pytest_ignore_collect(Path("evals/other.py"), self._config("")))

    def test_cli_passes_args_through_with_golden_marker(self) -> None:
        with mock.patch("pytest.main", return_value=0) as main:
            self.assertEqual(cli.main(["pytest", "-k", "tool_args", "--golden-source", "local"]), 0)
        args = main.call_args.args[0]
        self.assertTrue(args[0].endswith("evals"))
        self.assertEqual(args[1:], ["-m", "golden", "-k", "tool_args", "--golden-source", "local"])
        self.assertEqual(len(STAGES), 6)


class SandboxExecReplayTests(unittest.TestCase):
    def test_exec_sequence_replays_a_failure_then_success(self) -> None:
        llm = _ScriptedLLM(
            [
                LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="exec", arguments='{"command": "make"}')]),
                LLMResponse(content="", tool_calls=[ToolCall(id="c2", name="exec", arguments='{"command": "make"}')]),
                LLMResponse(content="好了。"),
            ]
        )
        case = {
            "prompt": "build",
            "tool_results": {
                "exec": {"sequence": [{"stdout": "", "stderr": "boom", "exit_code": 2}, {"stdout": "ok"}]}
            },
        }
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "MINIBOT_LANGFUSE": "0"}):
            output = run_case(case, config=Config(model="deepseek-v4-pro"), llm=llm)

        self.assertEqual([c["ok"] for c in output["tool_calls"]], [False, True])
        self.assertIn("退出码 2", llm.requests[1][-1].content)
        self.assertIn('"stdout":"ok"', llm.requests[2][-1].content)


if __name__ == "__main__":
    unittest.main()
