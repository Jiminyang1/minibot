"""Baselines: saved runs, ``run --baseline``, ``compare``, and trace links."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.evals import __main__ as cli
from minibot.evals import runner
from minibot.evals.report import format_regression, regression_report
from minibot.evals.runner import load_run, save_run
from minibot.evals.sandbox import model_facing_parts, parts_touched, run_case, summarize_events
from minibot.llm import LLMClient, LLMResponse
from minibot.prompts import SYSTEM_PROMPT
from minibot.runtime.events import RuntimeEvent
from minibot.runtime.messages import ModelMessage
from minibot.tools.definitions import ModelToolDefinition


def _row(
    case: str,
    passed: bool,
    *,
    calls: float = 2,
    tokens: float = 1000,
    trace=None,
    skills=(),
    tools=(),
    compactions: int = 0,
) -> dict:
    scores = {
        "tool_selection": (1.0 if passed else 0.0, "" if passed else "缺少 ['read_file']"),
        "end_to_end": (1.0 if passed else 0.0, ""),
        "model_calls": (calls, ""),
        "tokens": (tokens, ""),
        "latency_s": (3.0, ""),
    }
    return {
        "case": case,
        "output": {
            "reply": "x",
            "error": None,
            "skills_read": list(skills),
            "tool_calls": [{"name": name} for name in tools],
            "compactions": compactions,
        },
        "scores": scores,
        "trace_id": trace,
        "trace_url": f"https://lf/{trace}" if trace else None,
    }


def _run(name: str, rows: list[dict], **metadata) -> dict:
    # Same case definitions and scorer on both sides unless a test says otherwise.
    defaults = {
        "model": "m",
        "git": "abc",
        "scorer": "s1",
        "case_fingerprints": {row["case"]: f"fp-{row['case']}" for row in rows},
        "components": dict(_PARTS),
    }
    return {"run": name, "metadata": {**defaults, **metadata}, "results": rows}


_PARTS = {
    "prompt:system": "p1",
    "prompt:memory": "m1",
    "prompt:summary": "s1",
    "skill:calendar": "c1",
    "skill:mail": "ma1",
    "tool:exec": "e1",
    "tool:read_file": "r1",
}


def _printed(calls: list[str]):
    return mock.patch("builtins.print", side_effect=lambda *a, **k: calls.append(" ".join(map(str, a))))


class RegressionReportTests(unittest.TestCase):
    def test_worse_cases_bring_their_first_failing_stage_and_trace(self) -> None:
        base = _run("base", [_row("a", True), _row("a", True), _row("b", False), _row("b", False)])
        new = _run(
            "new",
            [
                _row("a", True, calls=3, tokens=1500),
                _row("a", False, calls=3, tokens=1500, trace="t-9"),
                _row("b", True),
                _row("b", True),
            ],
        )

        report = regression_report(base, new)

        changes = {flip["case"]: (flip["change"], flip["off"], flip["on"]) for flip in report["flips"]}
        self.assertEqual(changes, {"a": ("broken", (2, 2), (1, 2)), "b": ("fixed", (0, 2), (2, 2))})
        self.assertEqual([(e["case"], e["stage"], e["trace_id"]) for e in report["evidence"]],
                         [("a", "tool_selection", "t-9")])
        self.assertEqual(report["warnings"], [])

        text = format_regression(report)
        self.assertIn("回归对比: 基线 base → 本次 new", text)
        self.assertIn("每个用例 2 次", text)
        self.assertIn("变坏: a 2/2 → 1/2", text)
        self.assertIn("→ tool_selection: 缺少 ['read_file']", text)
        self.assertIn("trace: https://lf/t-9", text)
        self.assertIn("变好: b 0/2 → 2/2", text)
        # Worse cases come first, and cost moves are shown as a change.
        self.assertLess(text.index("变坏"), text.index("变好"))
        self.assertIn("model_calls      2 → 2.5（+25%）", text)
        self.assertIn("tokens           1000 → 1250（+25%）", text)

    def test_only_shared_cases_are_compared(self) -> None:
        base = _run("base", [_row("a", True), _row("gone", False)])
        new = _run("new", [_row("a", True), _row("added", False)])

        report = regression_report(base, new)

        self.assertEqual(report["cases"], 1)
        # "added" fails, but it has nothing to compare against.
        self.assertEqual(report["stages"]["end_to_end"], {"off": 1.0, "on": 1.0})
        self.assertEqual((report["only_base"], report["only_new"]), (["gone"], ["added"]))
        # Single runs, but nothing changed: no caution to give.
        self.assertEqual(report["warnings"], [])
        text = format_regression(report)
        self.assertIn("没有用例的端到端结果发生变化", text)
        self.assertIn("只在基线里（没有比较）: gone", text)
        self.assertIn("只在本次里（没有比较）: added", text)

    def test_warns_about_single_runs_and_a_different_model(self) -> None:
        base = _run("base", [_row("a", True)], model="m1")
        new = _run("new", [_row("a", False)], model="m2")

        text = format_regression(regression_report(base, new))

        self.assertIn("模型不同（m1 → m2）", text)
        self.assertIn("1 个变化的用例在某一边只跑了 1 次", text)
        self.assertIn("变坏: a 1/1 → 0/1", text)

    def test_a_case_redefined_in_between_is_listed_not_scored(self) -> None:
        base = _run("base", [_row("a", True), _row("b", True)])
        new = _run(
            "new",
            [_row("a", False), _row("b", True)],
            case_fingerprints={"a": "fp-a-edited", "b": "fp-b"},
        )

        report = regression_report(base, new)

        # "a" fails now, but its expectations changed: not a regression.
        self.assertEqual(report["flips"], [])
        self.assertEqual((report["cases"], report["redefined"]), (1, ["a"]))
        self.assertIn("两次之间改过定义（没有比较）: a", format_regression(report))

    def test_warns_about_a_scorer_change_and_old_format_runs(self) -> None:
        text = format_regression(
            regression_report(_run("b", [_row("a", True)]), _run("n", [_row("a", True)], scorer="s2"))
        )
        self.assertIn("打分代码不同（s1 → s2）", text)

        old = {"run": "old", "metadata": {"model": "m"}, "results": [_row("a", True)]}
        report = regression_report(old, _run("n", [_row("a", False)]))
        # Without fingerprints the case is still compared, with a caution.
        self.assertEqual([flip["case"] for flip in report["flips"]], ["a"])
        self.assertIn("基线没有记录用例定义（旧格式的运行）", format_regression(report))

    def test_no_shared_cases(self) -> None:
        text = format_regression(regression_report(_run("b", [_row("a", True)]), _run("n", [_row("z", True)])))
        self.assertIn("没有共同的用例", text)


class ComponentChangeTests(unittest.TestCase):
    def test_changed_parts_and_where_each_worse_case_should_look_first(self) -> None:
        moved = {
            **_PARTS,
            "prompt:summary": "s2",
            "skill:calendar": "c2",
            "tool:read_file": "r2",
            "tool:grep": "g1",
        }
        del moved["skill:mail"]
        # "a" read the calendar skill before; now it skips it and fails.
        # "b" only ever ran exec, which did not change.
        base = _run(
            "base",
            [_row("a", True, skills=["calendar"], tools=["read_skill"]), _row("b", True, tools=["exec"])],
        )
        new = _run("new", [_row("a", False), _row("b", False, tools=["exec"])], components=moved)

        report = regression_report(base, new)

        self.assertEqual(
            report["components"],
            {
                "changed": ["prompt:summary", "skill:calendar", "tool:read_file"],
                "added": ["tool:grep"],
                "removed": ["skill:mail"],
            },
        )
        # The summary prompt moved too, but neither case was compacted.
        self.assertEqual(report["suspects"], {"a": ["skill:calendar"], "b": []})
        text = format_regression(report)
        self.assertIn(
            "改过的部件: prompt:summary、skill:calendar、tool:read_file、tool:grep（新增）、skill:mail（去掉）",
            text,
        )
        self.assertIn("先查: skill:calendar（这个用例用到、又改过的部件）", text)
        self.assertIn("它用到的部件都没改", text)

    def test_the_system_prompt_goes_with_every_case_the_summary_prompt_after_compaction(self) -> None:
        moved = {**_PARTS, "prompt:system": "p2", "prompt:summary": "s2"}
        base = _run("base", [_row("a", True), _row("b", True)])
        new = _run("new", [_row("a", False, compactions=1), _row("b", False)], components=moved)

        report = regression_report(base, new)

        self.assertEqual(
            report["suspects"], {"a": ["prompt:summary", "prompt:system"], "b": ["prompt:system"]}
        )

    def test_nothing_recorded_moved_but_the_code_did(self) -> None:
        base = _run("base", [_row("a", True)])
        new = _run("new", [_row("a", False)], git="def")

        report = regression_report(base, new)

        self.assertEqual(report["suspects"], {"a": []})
        text = format_regression(report)
        self.assertIn("改过的部件: 无（prompt、skill、工具定义都和基线一样）；git 不同（abc → def）", text)
        self.assertNotIn("先查", text)
        self.assertNotIn("它用到的部件都没改", text)

    def test_runs_saved_without_parts_get_a_caution(self) -> None:
        report = regression_report(
            _run("base", [_row("a", True)], components=None), _run("new", [_row("a", False)])
        )

        self.assertIsNone(report["components"])
        self.assertEqual(report["suspects"], {})
        text = format_regression(report)
        self.assertIn("基线没有记录部件版本", text)
        self.assertNotIn("改过的部件", text)


class ModelFacingPartsTests(unittest.TestCase):
    def test_prompts_skills_and_tools_are_read_without_a_model_key(self) -> None:
        with mock.patch.dict(os.environ, {"MINIBOT_LANGFUSE": "0"}):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("DEEPSEEK_API_KEY", None)
            parts = model_facing_parts(Config(model=_MODEL))
            versions = runner.component_versions(Config(model=_MODEL))
            again = runner.component_versions(Config(model=_MODEL))

        self.assertLessEqual(
            {
                "prompt:system",
                "prompt:memory",
                "prompt:summary",
                "skill:calendar",
                "tool:read_file",
                # The macOS stand-ins carry the real server's definitions.
                "tool:mcp__macos_system__calendar_create_event",
            },
            set(parts),
        )
        self.assertEqual(parts["prompt:system"], SYSTEM_PROMPT)
        # Nothing from the temp sandbox (paths, times) leaks into a hash.
        self.assertEqual(versions, again)
        self.assertEqual(set(versions), set(parts))

    def test_editing_the_system_prompt_moves_only_its_hash(self) -> None:
        with mock.patch.dict(os.environ, {"MINIBOT_LANGFUSE": "0"}):
            before = runner.component_versions(Config(model=_MODEL))
            with mock.patch("minibot.bootstrap.SYSTEM_PROMPT", SYSTEM_PROMPT + "多一句。"):
                after = runner.component_versions(Config(model=_MODEL))

        self.assertEqual([name for name in before if before[name] != after[name]], ["prompt:system"])

    def test_a_run_touches_the_always_sent_prompts_and_what_it_used(self) -> None:
        output = {"skills_read": ["mail"], "tool_calls": [{"name": "read_skill"}, {"name": "exec"}]}

        self.assertEqual(
            parts_touched(output),
            {"prompt:system", "prompt:memory", "skill:mail", "tool:read_skill", "tool:exec"},
        )
        self.assertIn("prompt:summary", parts_touched({**output, "compactions": 1}))

    def test_compactions_are_counted(self) -> None:
        events = [
            RuntimeEvent(
                id=f"r:{seq}", run_id="r", session_id="s", seq=seq, type=type_, created_at="t", payload={}
            )
            for seq, type_ in enumerate(["model.request.completed", "compaction.request.completed"])
        ]

        summary = summarize_events(events)

        self.assertEqual((summary["model_calls"], summary["compactions"]), (2, 1))


class JudgeModelTests(unittest.TestCase):
    _ENV = {"DEEPSEEK_API_KEY": "test-key", "OPENAI_BASE_URL": "", "MINIBOT_LLM_PROVIDER": ""}

    def test_judge_is_the_tested_model_unless_pinned(self) -> None:
        config = Config(model="deepseek-v4-pro")
        with mock.patch.dict(os.environ, {"MINIBOT_EVAL_JUDGE_MODEL": ""}):
            self.assertEqual(runner.judge_model(config), "deepseek-v4-pro")
        with mock.patch.dict(os.environ, {"MINIBOT_EVAL_JUDGE_MODEL": " deepseek-v4-flash "}):
            self.assertEqual(runner.judge_model(config), "deepseek-v4-flash")

    def test_a_pinned_judge_ignores_the_tested_models_limit_overrides(self) -> None:
        # A context window that would be wrong for the judge's own catalog entry.
        config = Config(model="deepseek-v4-pro", context_window_tokens=500_000)
        with mock.patch.dict(os.environ, {**self._ENV, "MINIBOT_EVAL_JUDGE_MODEL": "deepseek-v4-flash"}):
            judge = runner.build_judge(config)
        self.assertEqual(judge.model, "deepseek-v4-flash")
        self.assertEqual(judge.profile.capabilities.context_window_tokens, 1_048_576)

        with mock.patch.dict(os.environ, {**self._ENV, "MINIBOT_EVAL_JUDGE_MODEL": ""}):
            same = runner.build_judge(config)
        self.assertEqual((same.model, same.profile.capabilities.context_window_tokens), ("deepseek-v4-pro", 500_000))

    def test_an_unknown_judge_fails_with_its_own_message(self) -> None:
        with (
            mock.patch.dict(os.environ, {**self._ENV, "MINIBOT_EVAL_JUDGE_MODEL": "no-such-model"}),
            self.assertRaisesRegex(ValueError, "评审模型 'no-such-model' 用不了"),
        ):
            runner.build_judge(Config(model="deepseek-v4-pro"))

    def test_saved_runs_record_the_judge(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp, "MINIBOT_EVAL_JUDGE_MODEL": "deepseek-v4-flash"}),
        ):
            save_run("r", [_row("a", True)], config=Config(model="deepseek-v4-pro"))
            metadata = load_run("r")["metadata"]
        self.assertEqual((metadata["model"], metadata["judge_model"]), ("deepseek-v4-pro", "deepseek-v4-flash"))

    def test_comparing_models_with_and_without_a_pinned_judge(self) -> None:
        pinned = regression_report(
            _run("old", [_row("a", True)], model="m1", judge_model="j"),
            _run("new", [_row("a", True)], model="m2", judge_model="j"),
        )
        self.assertIn("被测模型不同（m1 → m2）", "\n".join(pinned["warnings"]))
        self.assertNotIn("评审模型不同", "\n".join(pinned["warnings"]))

        # A run saved before the setting existed was graded by its tested model.
        unpinned = format_regression(
            regression_report(
                {"run": "old", "metadata": {"model": "m1"}, "results": [_row("a", True)]},
                _run("new", [_row("a", True)], model="m2", judge_model="m2"),
            )
        )
        self.assertIn("评审模型不同（m1 → m2），回答阶段的分数不能直接比", unpinned)
        self.assertIn("评审 m1", unpinned)


class SavedRunTests(unittest.TestCase):
    def test_save_then_load_by_name_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp}):
            path = save_run("r1", [_row("a", True)], config=Config(model="m"), extra={"repeat": 1})

            by_name, by_path = load_run("r1"), load_run(str(path))

        self.assertEqual(path, Path(tmp).resolve() / "evals" / "r1.json")
        self.assertEqual(by_name, by_path)
        self.assertEqual(by_name["run"], "r1")
        self.assertEqual((by_name["metadata"]["model"], by_name["metadata"]["repeat"]), ("m", 1))
        self.assertEqual(by_name["results"][0]["case"], "a")

    def test_fingerprint_follows_input_and_expected_only(self) -> None:
        case = {"id": "a", "category": "x", "input": {"prompt": "p"}, "expected": {"skill": None, "tools_required": ["t"]}}
        reordered = {"expected": {"tools_required": ["t"], "skill": None}, "input": {"prompt": "p"}, "id": "a", "category": "y"}
        edited = {**case, "expected": {"skill": None, "tools_required": ["u"]}}

        same, again, changed = (runner.case_fingerprints([c])["a"] for c in (case, reordered, edited))

        self.assertEqual(same, again)  # key order and other fields do not count
        self.assertNotEqual(same, changed)

    def test_missing_or_foreign_files_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp}):
            with self.assertRaisesRegex(ValueError, "找不到运行结果 nope"):
                load_run("nope")
            other = Path(tmp) / "other.json"
            other.write_text('{"failures": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是 evals 保存的运行结果"):
                load_run(str(other))
            broken = Path(tmp) / "broken.json"
            broken.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "读不了运行结果"):
                load_run(str(broken))


class BaselineCommandTests(unittest.TestCase):
    def test_run_saves_pooled_repeats_and_compares_with_the_baseline(self) -> None:
        outcomes = iter([True, False])

        def run_local(*, config, case_ids, run_name):
            del config, case_ids, run_name
            return [_row("a", next(outcomes), trace="t-new")]

        printed: list[str] = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp}),
            mock.patch.object(cli, "load_env"),
            mock.patch.object(cli, "run_local", side_effect=run_local),
            _printed(printed),
        ):
            save_run("main", [_row("a", True), _row("a", True)], config=Config(model="m"))
            code = cli.main(["run", "--local", "--name", "br", "--repeat", "2", "--baseline", "main"])
            saved = load_run("br")

        self.assertEqual(code, 0)
        self.assertEqual([(r["case"], r["trial"]) for r in saved["results"]], [("a", 1), ("a", 2)])
        self.assertEqual((saved["metadata"]["repeat"], saved["metadata"]["mode"]), (2, "local"))
        # Every case definition and the scorer are fingerprinted with the run.
        self.assertEqual(
            set(saved["metadata"]["case_fingerprints"]), {c["id"] for c in runner.load_cases()}
        )
        self.assertEqual(saved["metadata"]["scorer"], runner.scorer_version())
        # So are the prompts, skills and tool definitions the cases were shown.
        self.assertLessEqual({"prompt:system", "skill:calendar"}, set(saved["metadata"]["components"]))
        output = "\n".join(printed)
        self.assertIn("可以用作 --baseline br", output)
        self.assertIn("回归对比: 基线 main → 本次 br", output)
        self.assertIn("变坏: a 2/2 → 1/2", output)
        self.assertIn("trace: https://lf/t-new", output)

    def test_a_missing_baseline_stops_before_the_run_spends_anything(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp}),
            mock.patch.object(cli, "load_env"),
            mock.patch.object(cli, "run_local") as run_local,
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            cli.main(["run", "--local", "--baseline", "nope"])
        run_local.assert_not_called()

    def test_baseline_needs_a_single_lessons_mode(self) -> None:
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.main(["run", "--local", "--lessons", "compare", "--baseline", "main"])

    def test_compare_two_saved_runs(self) -> None:
        printed: list[str] = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {"MINIBOT_HOME": tmp}),
            _printed(printed),
        ):
            save_run("old", [_row("a", False)], config=Config(model="m"))
            save_run("new", [_row("a", True)], config=Config(model="m"))
            self.assertEqual(cli.main(["compare", "old", "new"]), 0)
            self.assertEqual(cli.main(["compare", "old", "missing"]), 1)

        output = "\n".join(printed)
        self.assertIn("变好: a 0/1 → 1/1", output)
        self.assertIn("找不到运行结果 missing", output)


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        del messages, tools, model
        return self._responses.pop(0)


class _TraceRecorder:
    """Stands in for ``LangfuseFold``: a fixed trace id per run."""

    def __init__(self) -> None:
        self.runs: list[str] = []

    def __call__(self, event) -> None:
        if event.type == "run.started":
            self.runs.append(event.run_id)

    def trace_id_of(self, run_id: str) -> str | None:
        return f"trace-of-{run_id}" if run_id in self.runs else None

    def shutdown(self) -> None:
        pass


# build_runtime needs a model it knows the context window of.
_MODEL = "deepseek-v4-pro"


class EvalTraceLinkTests(unittest.TestCase):
    def test_sandbox_output_names_the_run_and_its_trace(self) -> None:
        recorder = _TraceRecorder()
        env = {"OPENAI_API_KEY": "test-key", "MINIBOT_LANGFUSE": "0"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("minibot.bootstrap.build_langfuse_fold", return_value=recorder),
        ):
            output = run_case(
                {"prompt": "你好"}, config=Config(model=_MODEL), llm=_ScriptedLLM([LLMResponse(content="嗨")])
            )

        self.assertEqual(recorder.runs, [output["run_id"]])
        self.assertEqual(output["trace_id"], f"trace-of-{output['run_id']}")

    def test_sandbox_output_without_tracing(self) -> None:
        env = {"OPENAI_API_KEY": "test-key", "MINIBOT_LANGFUSE": "0"}
        with mock.patch.dict(os.environ, env):
            output = run_case(
                {"prompt": "你好"}, config=Config(model=_MODEL), llm=_ScriptedLLM([LLMResponse(content="嗨")])
            )
        self.assertTrue(output["run_id"])
        self.assertIsNone(output["trace_id"])

    def test_local_rows_link_the_trace_when_langfuse_is_configured(self) -> None:
        client = mock.MagicMock()
        client.get_trace_url.return_value = "https://lf/trace/t-1"
        output = {"reply": "x", "error": None, "tool_calls": [], "trace_id": "t-1"}
        with (
            mock.patch.object(runner, "run_case", return_value=output),
            mock.patch.object(runner, "load_cases", return_value=[{"id": "a", "input": {}, "expected": {}}]),
            mock.patch.object(runner, "build_judge"),
            mock.patch.object(runner, "langfuse_configured", return_value=True),
            mock.patch("langfuse.get_client", return_value=client),
        ):
            rows = runner.run_local(config=Config(model="m"), case_ids=None, run_name="r", log=lambda _: None)

        self.assertEqual((rows[0]["trace_id"], rows[0]["trace_url"]), ("t-1", "https://lf/trace/t-1"))

        with (
            mock.patch.object(runner, "run_case", return_value=output),
            mock.patch.object(runner, "load_cases", return_value=[{"id": "a", "input": {}, "expected": {}}]),
            mock.patch.object(runner, "build_judge"),
            mock.patch.object(runner, "langfuse_configured", return_value=False),
        ):
            rows = runner.run_local(config=Config(model="m"), case_ids=None, run_name="r", log=lambda _: None)

        # The id is still kept; without keys there is no URL to build.
        self.assertEqual((rows[0]["trace_id"], rows[0]["trace_url"]), ("t-1", None))


if __name__ == "__main__":
    unittest.main()
