"""Failure localization: where in the pipeline did each failing case break?

A turn unfolds in a fixed order — pick a skill, pick tools, fill arguments,
answer, leave the world in some state — so the first stage that failed is
where to start reading the trace. Later failures are often consequences
(wrong tool → wrong args → wrong answer). The report lists, per failing
case, that first stage, the scorer's comment and the trace URL, then how
often each stage failed across the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import resolve_state_home
from .scoring import stage_passed

PIPELINE = ("skill_selection", "tool_selection", "tool_args", "answer", "outcome")


def failure_report(rows: list[dict[str, Any]], *, run_name: str, mode: str) -> dict[str, Any]:
    """Reduce scored rows (see ``runner``) to the first failure per case.

    *mode* is ``"langfuse"`` or ``"local"``; local runs have no trace.
    """
    failures: list[dict[str, Any]] = []
    counts = {stage: {"failed": 0, "first": 0} for stage in (*PIPELINE, "run")}
    passed = 0
    for row in rows:
        scores = row.get("scores") or {}
        e2e = scores.get("end_to_end")
        if e2e is not None and stage_passed("end_to_end", e2e[0]):
            passed += 1
            continue
        failed = [
            stage
            for stage in PIPELINE
            if stage in scores and not stage_passed(stage, scores[stage][0])
        ]
        error = (row.get("output") or {}).get("error")
        for stage in failed:
            counts[stage]["failed"] += 1
        if failed:
            first = failed[0]
            comment = scores[first][1]
        else:
            # Every scored stage passed, so the run itself broke (or nothing
            # but end_to_end was scored).
            first = "run"
            comment = error or (e2e[1] if e2e else "没有 end_to_end 分数")
            counts["run"]["failed"] += 1
        counts[first]["first"] += 1
        failures.append(
            {
                "case": row.get("case"),
                "stage": first,
                "comment": comment,
                "failed_stages": failed,
                "error": error,
                "trace_id": row.get("trace_id"),
                "trace_url": row.get("trace_url"),
            }
        )
    return {
        "run": run_name,
        "mode": mode,
        "total": len(rows),
        "passed": passed,
        "failures": failures,
        "stage_failures": {k: v for k, v in counts.items() if k != "run" or v["failed"]},
    }


def format_failure_report(report: dict[str, Any]) -> str:
    failures = report["failures"]
    if not failures:
        return f"失败定位: {report['total']} 个用例全部端到端通过。"
    no_trace = "无（本地模式不关联 Langfuse trace）" if report["mode"] == "local" else "无"
    lines = [f"失败定位（{len(failures)}/{report['total']} 个用例未通过，按首个失败阶段）:"]
    for failure in failures:
        lines.append(f"  {failure['case']} → {failure['stage']}: {failure['comment']}")
        if failure["error"] and failure["stage"] != "run":
            lines.append(f"      运行出错: {failure['error']}")
        lines.append(f"      trace: {failure['trace_url'] or no_trace}")
    lines.append("")
    lines.append("各阶段失败数（失败 / 其中为首个失败）:")
    for stage, count in report["stage_failures"].items():
        lines.append(f"  {stage:<16} {count['failed']} / {count['first']}")
    return "\n".join(lines)


def save_failure_report(report: dict[str, Any], *, out_dir: Path | None = None) -> Path:
    """Write ``<run>.failures.json`` next to the local results."""
    out_dir = out_dir or resolve_state_home() / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['run']}.failures.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


__all__ = ["PIPELINE", "failure_report", "format_failure_report", "save_failure_report"]


# ── lessons on vs off ─────────────────────────────────────────────


def compare_runs(
    off_rows: list[dict[str, Any]],
    on_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-stage pass rates and per-case end-to-end flips between two runs.

    The question it answers: did the approved lessons fix the cases they were
    drawn for without breaking others? A flip names the lessons the "on" run
    actually received, so a regression points at a specific lesson.
    """
    off = {row["case"]: row for row in off_rows}
    on = {row["case"]: row for row in on_rows}
    stages: dict[str, dict[str, float | None]] = {}
    for stage in (*PIPELINE, "end_to_end"):
        stages[stage] = {
            "off": _pass_rate(off_rows, stage),
            "on": _pass_rate(on_rows, stage),
        }
    flips = []
    for case in sorted(off.keys() & on.keys()):
        before = _passed(off[case], "end_to_end")
        after = _passed(on[case], "end_to_end")
        if before is None or after is None or before == after:
            continue
        flips.append(
            {
                "case": case,
                "change": "fixed" if after else "broken",
                "lessons": list((on[case].get("output") or {}).get("lessons_used") or []),
            }
        )
    # Cost too: a lesson can leave correctness unchanged and still save
    # model calls (fewer wrong attempts before the right one).
    metrics = {
        metric: {"off": _mean(off_rows, metric), "on": _mean(on_rows, metric)}
        for metric in _METRICS
    }
    used = sum(1 for row in on_rows if (row.get("output") or {}).get("lessons_used"))
    return {
        "stages": stages,
        "metrics": metrics,
        "flips": flips,
        "cases_with_lessons": used,
        "total": len(on_rows),
    }


def format_comparison(comparison: dict[str, Any]) -> str:
    lines = ["lesson 对比（off → on）:"]
    for stage, rates in comparison["stages"].items():
        if rates["off"] is None and rates["on"] is None:
            continue
        lines.append(f"  {stage:<16} {_rate(rates['off'])} → {_rate(rates['on'])}")
    for metric, means in comparison.get("metrics", {}).items():
        if means["off"] is None or means["on"] is None:
            continue
        lines.append(f"  {metric:<16} {means['off']:g} → {means['on']:g}（平均）")
    lines.append(
        f"  {comparison['cases_with_lessons']}/{comparison['total']} 个用例在 on 模式下收到了 lesson。"
    )
    if not comparison["flips"]:
        lines.append("  没有用例的端到端结果发生变化。")
    for flip in comparison["flips"]:
        mark = "修好" if flip["change"] == "fixed" else "变坏"
        lessons = ", ".join(flip["lessons"]) or "无"
        lines.append(f"  {mark}: {flip['case']}（用到的 lesson: {lessons}）")
    return "\n".join(lines)


_METRICS = ("model_calls", "tokens", "latency_s")


def _mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [row["scores"][metric][0] for row in rows if metric in row["scores"]]
    return round(sum(values) / len(values), 1) if values else None


def _passed(row: dict[str, Any], stage: str) -> bool | None:
    score = row["scores"].get(stage)
    return None if score is None else stage_passed(stage, score[0])


def _pass_rate(rows: list[dict[str, Any]], stage: str) -> float | None:
    results = [p for p in (_passed(row, stage) for row in rows) if p is not None]
    return sum(results) / len(results) if results else None


def _rate(value: float | None) -> str:
    return "  —" if value is None else f"{value:4.0%}"
