"""Evals with lessons on and off, and the comparison between the two runs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.evals.report import compare_runs, format_comparison
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
                {"case": "a", "change": "fixed", "lessons": ["les_1"]},
                {"case": "b", "change": "broken", "lessons": ["les_3"]},
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

    def test_no_change_is_said_plainly(self) -> None:
        rows = [_row("a", True)]

        self.assertIn("没有用例的端到端结果发生变化", format_comparison(compare_runs(rows, rows)))


if __name__ == "__main__":
    unittest.main()
