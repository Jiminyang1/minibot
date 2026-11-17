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
    no_trace = "无（本地模式下只有配置了 Langfuse 才有 trace）" if report["mode"] == "local" else "无"
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


__all__ = [
    "PIPELINE",
    "failure_report",
    "format_failure_report",
    "format_regression",
    "regression_report",
    "save_failure_report",
]


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
    off = _by_case(off_rows)
    on = _by_case(on_rows)
    stages: dict[str, dict[str, float | None]] = {}
    for stage in (*PIPELINE, "end_to_end"):
        stages[stage] = {
            "off": _pass_rate(off_rows, stage),
            "on": _pass_rate(on_rows, stage),
        }
    flips = []
    for case in sorted(off.keys() & on.keys()):
        before = _pass_count(off[case])
        after = _pass_count(on[case])
        if not before[1] or not after[1]:
            continue
        # With repeats a case moves by a share of its runs, not all or nothing.
        delta = after[0] / after[1] - before[0] / before[1]
        if delta == 0:
            continue
        lessons: list[str] = []
        for row in on[case]:
            for lesson in (row.get("output") or {}).get("lessons_used") or []:
                if lesson not in lessons:
                    lessons.append(lesson)
        flips.append(
            {
                "case": case,
                "change": "fixed" if delta > 0 else "broken",
                "off": before,
                "on": after,
                "lessons": lessons,
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
        f"  {comparison['cases_with_lessons']}/{comparison['total']} 次用例运行在 on 模式下收到了 lesson。"
    )
    if not comparison["flips"]:
        lines.append("  没有用例的端到端结果发生变化。")
    for flip in comparison["flips"]:
        lessons = ", ".join(flip["lessons"]) or "无"
        if flip["off"][1] == 1 and flip["on"][1] == 1:
            mark = "修好" if flip["change"] == "fixed" else "变坏"
            moved = ""
        else:
            mark = "变好" if flip["change"] == "fixed" else "变坏"
            moved = f" {_count(flip['off'])} → {_count(flip['on'])}"
        lines.append(f"  {mark}: {flip['case']}{moved}（用到的 lesson: {lessons}）")
    return "\n".join(lines)


# ── this run against a saved baseline ────────────────────────────


def regression_report(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """What changed between two saved runs (``runner.load_run``) of the cases.

    Only cases present in both runs, with the same definition, are
    compared: adding a case does not move the rates, and a case whose input
    or expectations were edited in between is listed, not scored as a
    change. Runs saved before fingerprints existed cannot be checked; the
    report says so. A case changed when its share of passing runs moved
    (with repeats, 3/3 → 1/3); each case that got worse brings the failing
    runs of the new side, first failing stage and trace, so the next step is
    to open that trace. Cost moves too: a prompt that grew is a regression
    even when every case still passes.
    """
    base_rows = base.get("results") or []
    new_rows = new.get("results") or []
    base_cases = _by_case(base_rows)
    new_cases = _by_case(new_rows)
    shared = base_cases.keys() & new_cases.keys()
    base_prints = _fingerprints(base)
    new_prints = _fingerprints(new)
    redefined = sorted(
        case
        for case in shared
        if case in base_prints and case in new_prints and base_prints[case] != new_prints[case]
    )
    common = shared - set(redefined)
    comparison = compare_runs(
        [row for row in base_rows if row["case"] in common],
        [row for row in new_rows if row["case"] in common],
    )
    broken = {flip["case"] for flip in comparison["flips"] if flip["change"] == "broken"}
    evidence = failure_report(
        [row for row in new_rows if row["case"] in broken],
        run_name=str(new.get("run")),
        mode=str((new.get("metadata") or {}).get("mode") or "local"),
    )["failures"]
    return {
        "base": _run_summary(base, base_cases),
        "new": _run_summary(new, new_cases),
        "cases": len(common),
        "stages": comparison["stages"],
        "metrics": comparison["metrics"],
        "flips": comparison["flips"],
        "evidence": evidence,
        "unchanged": len(common) - len(comparison["flips"]),
        "only_base": sorted(base_cases.keys() - new_cases.keys()),
        "only_new": sorted(new_cases.keys() - base_cases.keys()),
        "redefined": redefined,
        "warnings": _regression_warnings(
            base, new, base_cases, new_cases, {flip["case"] for flip in comparison["flips"]}
        ),
    }


def _fingerprints(run: dict[str, Any]) -> dict[str, str]:
    prints = (run.get("metadata") or {}).get("case_fingerprints")
    return prints if isinstance(prints, dict) else {}


def format_regression(report: dict[str, Any]) -> str:
    base, new = report["base"], report["new"]
    lines = [f"回归对比: 基线 {base['run']} → 本次 {new['run']}"]
    for label, side in (("基线", base), ("本次", new)):
        lines.append(
            f"  {label}: 模型 {side['model'] or '?'} · git {side['git'] or '?'} · "
            f"每个用例 {side['runs_per_case']} 次"
        )
    for warning in report["warnings"]:
        lines.append(f"  注意: {warning}")
    if not report["cases"]:
        lines.append("  两次运行没有共同的用例，没有可比的东西。")
        return "\n".join(lines)
    lines.append(f"  各阶段通过率（{report['cases']} 个共同用例）:")
    for stage, rates in report["stages"].items():
        if rates["off"] is None and rates["on"] is None:
            continue
        lines.append(f"    {stage:<16} {_rate(rates['off'])} → {_rate(rates['on'])}")
    costs = [
        (metric, means)
        for metric, means in report["metrics"].items()
        if means["off"] is not None and means["on"] is not None
    ]
    if costs:
        lines.append("  成本（每次运行的平均值）:")
        for metric, means in costs:
            lines.append(
                f"    {metric:<16} {means['off']:g} → {means['on']:g}{_change(means['off'], means['on'])}"
            )
    evidence: dict[str, list[dict[str, Any]]] = {}
    for failure in report["evidence"]:
        evidence.setdefault(failure["case"], []).append(failure)
    for flip in sorted(report["flips"], key=lambda f: (f["change"] != "broken", f["case"])):
        mark = "变坏" if flip["change"] == "broken" else "变好"
        lines.append(f"  {mark}: {flip['case']} {_count(flip['off'])} → {_count(flip['on'])}")
        for failure in evidence.get(flip["case"], []):
            lines.append(f"      → {failure['stage']}: {failure['comment']}")
            if failure["trace_url"] or failure["trace_id"]:
                lines.append(f"        trace: {failure['trace_url'] or failure['trace_id']}")
    if not report["flips"]:
        lines.append("  没有用例的端到端结果发生变化。")
    else:
        lines.append(f"  其余 {report['unchanged']} 个用例结果没变。")
    if report["only_base"]:
        lines.append(f"  只在基线里（没有比较）: {', '.join(report['only_base'])}")
    if report["only_new"]:
        lines.append(f"  只在本次里（没有比较）: {', '.join(report['only_new'])}")
    if report["redefined"]:
        lines.append(f"  两次之间改过定义（没有比较）: {', '.join(report['redefined'])}")
    return "\n".join(lines)


def _run_summary(run: dict[str, Any], cases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metadata = run.get("metadata") or {}
    counts = {len(rows) for rows in cases.values()}
    return {
        "run": run.get("run"),
        "model": metadata.get("model"),
        "git": metadata.get("git"),
        "lessons": metadata.get("lessons"),
        "runs_per_case": "/".join(str(c) for c in sorted(counts)) or "0",
        "min_runs": min(counts) if counts else 0,
    }


def _regression_warnings(
    base: dict[str, Any],
    new: dict[str, Any],
    base_cases: dict[str, list[dict[str, Any]]],
    new_cases: dict[str, list[dict[str, Any]]],
    changed: set[str],
) -> list[str]:
    warnings = []
    base_meta, new_meta = base.get("metadata") or {}, new.get("metadata") or {}
    for key, label in (("model", "模型"), ("lessons", "lesson 模式")):
        if base_meta.get(key) and new_meta.get(key) and base_meta[key] != new_meta[key]:
            warnings.append(
                f"两边的{label}不同（{base_meta[key]} → {new_meta[key]}），变化不全是代码造成的。"
            )
    old_format = [
        label
        for label, run in (("基线", base), ("本次", new))
        if not _fingerprints(run)
    ]
    if old_format:
        warnings.append(
            f"{'、'.join(old_format)}没有记录用例定义（旧格式的运行），"
            "两次之间改过的用例分不出来。"
        )
    base_scorer, new_scorer = base_meta.get("scorer"), new_meta.get("scorer")
    if base_scorer and new_scorer and base_scorer != new_scorer:
        warnings.append(
            f"打分代码不同（{base_scorer} → {new_scorer}），变化可能来自打分方式，不是 agent。"
        )
    # Only a change needs the caution: one run cannot tell it from variance.
    single = [
        case for case in sorted(changed) if len(base_cases[case]) == 1 or len(new_cases[case]) == 1
    ]
    if single:
        warnings.append(
            f"有 {len(single)} 个变化的用例在某一边只跑了 1 次，这个变化可能只是模型的随机性；"
            "用 --repeat 3 或更多再跑一次再下结论。"
        )
    return warnings


def _change(before: float, after: float) -> str:
    if before == 0:
        return ""
    percent = (after - before) / before * 100
    return f"（{percent:+.0f}%）" if round(percent) else ""


# ── the same cases, run several times ─────────────────────────────


def stability_report(trials: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Which end-to-end results hold when the same cases run again?

    One run of ten cases cannot tell a real change from the model's own
    variance: one flip moves the pass rate by ten points. Repeats give each
    case a pass count, so a case is ``pass`` (every run), ``fail`` (no run)
    or ``flaky`` (some runs), and the pass rate gets a range.
    """
    grouped = _by_case([row for rows in trials for row in rows])
    cases = []
    for case, rows in grouped.items():
        passes, runs = _pass_count(rows)
        if not runs:
            continue
        verdict = "pass" if passes == runs else "fail" if passes == 0 else "flaky"
        cases.append({"case": case, "passes": passes, "runs": runs, "verdict": verdict})
    rates = [
        rate
        for rate in (_pass_rate(rows, "end_to_end") for rows in trials)
        if rate is not None
    ]
    return {
        "trials": len(trials),
        "cases": cases,
        "pass_rate": (
            None
            if not rates
            else {"mean": sum(rates) / len(rates), "min": min(rates), "max": max(rates)}
        ),
        "verdicts": {
            verdict: sum(1 for case in cases if case["verdict"] == verdict)
            for verdict in ("pass", "flaky", "fail")
        },
    }


def format_stability(report: dict[str, Any]) -> str:
    lines = [f"重复 {report['trials']} 次的稳定性:"]
    rate = report["pass_rate"]
    if rate is not None:
        lines.append(
            f"  端到端通过率 平均 {rate['mean']:.0%}（各次 {rate['min']:.0%} – {rate['max']:.0%}）"
        )
    verdicts = report["verdicts"]
    lines.append(
        f"  每次都过 {verdicts['pass']} · 时过时不过 {verdicts['flaky']} · 每次都不过 {verdicts['fail']}"
    )
    for case in report["cases"]:
        if case["verdict"] == "pass":
            continue
        label = "时过时不过" if case["verdict"] == "flaky" else "每次都不过"
        lines.append(f"  {label}: {case['case']} {case['passes']}/{case['runs']}")
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


def _by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    return grouped


def _pass_count(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(end-to-end passes, scored runs) for one case's rows."""
    results = [p for p in (_passed(row, "end_to_end") for row in rows) if p is not None]
    return sum(results), len(results)


def _count(count: tuple[int, int]) -> str:
    return f"{count[0]}/{count[1]}"


def _rate(value: float | None) -> str:
    return "  —" if value is None else f"{value:4.0%}"
