"""Evals with lessons on and off, and the comparison between the two runs."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.evals import __main__ as cli
from minibot.evals.report import (
    compare_runs,
    format_comparison,
    format_stability,
    stability_report,
)
from minibot.evals.runner import with_lessons
from minibot.evals.sandbox import summarize_events
from minibot.runtime.events import RuntimeEvent


def _row(
    case: str,
    passed: bool,
    *,
    tools: float = 1.0,
    calls: float = 2.0,
    lessons: list[str] | None = None,
) -> dict:
    return {
        "case": case,
        "output": {"lessons_used": lessons or []},
        "scores": {
            "tool_selection": (tools, ""),
            "end_to_end": (1.0 if passed else 0.0, ""),
            "model_calls": (calls, ""),
        },
    }


def _event(type_: str, payload: dict, seq: int) -> RuntimeEvent:
    return RuntimeEvent(
        id=f"r:{seq}",
        run_id="r",
        session_id="s",
        seq=seq,
        type=type_,
        created_at="t",
        payload=payload,
    )


class WithLessonsTests(unittest.TestCase):
    def test_off_disables_and_on_points_at_the_real_store(self) -> None:
        config = Config(lessons_mode="auto")

        self.assertEqual(with_lessons(config, "off").lessons_mode, "off")
        with patch("minibot.evals.runner.resolve_state_home", return_value=Path("/home/x/.minibot")):
            on = with_lessons(config, "on")
        self.assertEqual(on.lessons_mode, "on")
        # Cases run in a temp state home, so the store path must be explicit.
        self.assertEqual(on.lessons_path, Path("/home/x/.minibot/lessons.json"))

    def test_an_explicit_store_path_is_kept(self) -> None:
        config = Config(lessons_path=Path("/tmp/l.json"))

        self.assertEqual(with_lessons(config, "on").lessons_path, Path("/tmp/l.json"))


class LessonsUsedTests(unittest.TestCase):
    def test_summary_lists_each_lesson_once_in_order(self) -> None:
        events = [
            _event("model.request.started", {"iteration": 1, "lessons": ["les_2", "les_1"]}, 1),
            _event("model.request.started", {"iteration": 2, "lessons": ["les_2"]}, 2),
            _event("model.request.started", {"iteration": 3}, 3),
        ]

        self.assertEqual(summarize_events(events)["lessons_used"], ["les_2", "les_1"])


class CompareRunsTests(unittest.TestCase):
    def test_flips_name_the_lessons_and_rates_cover_both_runs(self) -> None:
        off = [_row("a", False, tools=0.0), _row("b", True), _row("c", True)]
        on = [
            _row("a", True, lessons=["les_1"]),
            _row("b", False, lessons=["les_3"]),
            _row("c", True),
        ]

        comparison = compare_runs(off, on)

        self.assertEqual(
            comparison["flips"],
            [
                {"case": "a", "change": "fixed", "off": (0, 1), "on": (1, 1), "lessons": ["les_1"]},
                {"case": "b", "change": "broken", "off": (1, 1), "on": (0, 1), "lessons": ["les_3"]},
            ],
        )
        self.assertAlmostEqual(comparison["stages"]["tool_selection"]["off"], 2 / 3)
        self.assertAlmostEqual(comparison["stages"]["tool_selection"]["on"], 1.0)
        self.assertIsNone(comparison["stages"]["answer"]["off"])
        self.assertEqual(comparison["cases_with_lessons"], 2)

        text = format_comparison(comparison)
        self.assertIn("修好: a（用到的 lesson: les_1）", text)
        self.assertIn("变坏: b（用到的 lesson: les_3）", text)
        self.assertNotIn("answer", text)

    def test_cost_is_compared_even_when_results_do_not_change(self) -> None:
        off = [_row("a", True, calls=10), _row("b", True, calls=2)]
        on = [_row("a", True, calls=4), _row("b", True, calls=2)]

        comparison = compare_runs(off, on)

        self.assertEqual(comparison["metrics"]["model_calls"], {"off": 6.0, "on": 3.0})
        self.assertIsNone(comparison["metrics"]["tokens"]["off"])
        self.assertIn("model_calls      6 → 3（平均）", format_comparison(comparison))

    def test_repeated_runs_compare_pass_counts_per_case(self) -> None:
        # Three runs of each case per mode, pooled.
        off = [
            *[_row("a", ok) for ok in (False, False, True)],
            *[_row("b", ok) for ok in (True, True, True)],
            *[_row("c", ok) for ok in (True, False, True)],
        ]
        on = [
            _row("a", True, lessons=["les_1"]),
            _row("a", True, lessons=["les_1", "les_2"]),
            _row("a", True),
            *[_row("b", ok) for ok in (True, False, True)],
            *[_row("c", ok) for ok in (False, True, True)],
        ]

        comparison = compare_runs(off, on)

        # c passed 2 of 3 both times: which run failed does not matter.
        self.assertEqual(
            comparison["flips"],
            [
                {
                    "case": "a",
                    "change": "fixed",
                    "off": (1, 3),
                    "on": (3, 3),
                    "lessons": ["les_1", "les_2"],
                },
                {"case": "b", "change": "broken", "off": (3, 3), "on": (2, 3), "lessons": []},
            ],
        )
        self.assertAlmostEqual(comparison["stages"]["end_to_end"]["off"], 6 / 9)
        text = format_comparison(comparison)
        self.assertIn("变好: a 1/3 → 3/3（用到的 lesson: les_1, les_2）", text)
        self.assertIn("变坏: b 3/3 → 2/3（用到的 lesson: 无）", text)
        self.assertIn("2/9 次用例运行在 on 模式下收到了 lesson", text)

    def test_no_change_is_said_plainly(self) -> None:
        rows = [_row("a", True)]

        self.assertIn("没有用例的端到端结果发生变化", format_comparison(compare_runs(rows, rows)))


if __name__ == "__main__":
    unittest.main()


class StabilityTests(unittest.TestCase):
    TRIALS = [
        [_row("steady", True), _row("flaky", True), _row("broken", False)],
        [_row("steady", True), _row("flaky", False), _row("broken", False)],
        [_row("steady", True), _row("flaky", True), _row("broken", False)],
    ]

    def test_each_case_is_pass_flaky_or_fail_across_trials(self) -> None:
        report = stability_report(self.TRIALS)

        self.assertEqual(report["trials"], 3)
        self.assertEqual(
            {case["case"]: (case["passes"], case["runs"], case["verdict"]) for case in report["cases"]},
            {"steady": (3, 3, "pass"), "flaky": (2, 3, "flaky"), "broken": (0, 3, "fail")},
        )
        self.assertEqual(report["verdicts"], {"pass": 1, "flaky": 1, "fail": 1})
        self.assertAlmostEqual(report["pass_rate"]["min"], 1 / 3)
        self.assertAlmostEqual(report["pass_rate"]["max"], 2 / 3)
        self.assertAlmostEqual(report["pass_rate"]["mean"], 5 / 9)

        text = format_stability(report)
        self.assertIn("平均 56%（各次 33% – 67%）", text)
        self.assertIn("时过时不过: flaky 2/3", text)
        self.assertIn("每次都不过: broken 0/3", text)
        self.assertNotIn("steady", text)

    def test_repeat_runs_each_mode_n_times_and_pools_the_comparison(self) -> None:
        names: list[str] = []
        # off: "a" passes 1 of 2. on: "a" passes 2 of 2.
        outcomes = {"off": iter([False, True]), "on": iter([True, True])}

        def run_local(*, config, case_ids, run_name):
            del case_ids
            names.append(run_name)
            return [_row("a", next(outcomes[config.lessons_mode]))]

        printed: list[str] = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"MINIBOT_HOME": tmp}),
            patch.object(cli, "load_env"),
            patch.object(cli, "run_local", side_effect=run_local),
            patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(map(str, a)))),
        ):
            code = cli.main(
                ["run", "--local", "--name", "r", "--lessons", "compare", "--repeat", "2"]
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            names, ["r-lessons-off-t1", "r-lessons-off-t2", "r-lessons-on-t1", "r-lessons-on-t2"]
        )
        output = "\n".join(printed)
        self.assertIn("时过时不过: a 1/2", output)
        self.assertIn("变好: a 1/2 → 2/2", output)

    def test_repeat_must_be_positive(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.main(["run", "--local", "--repeat", "0"])
