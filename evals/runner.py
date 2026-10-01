"""Golden evals: cases in git, mirrored to a Langfuse dataset, scored per stage.

``cases.json`` is the source of truth (reviewed like code). ``sync`` upserts
it into the ``minibot-golden`` dataset; ``run`` executes every case as one
real turn in the sandbox and records the stage scores. With Langfuse
configured, a run is a Langfuse experiment: each case's minibot trace nests
under its dataset item and carries the scores, so runs can be compared side
by side after a prompt, skill or model change. Without it (or ``--local``)
the same scores are printed and saved under ``$MINIBOT_HOME/evals/``.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from ..config import Config, resolve_state_home
from ..llm import LLMClient
from ..llm_factory import build_llm_client_from_profile
from ..llm_profile import build_llm_profile
from .sandbox import run_case
from .scoring import StageScore, end_to_end, judge_answer, score_deterministic

DATASET = "minibot-golden"
_CASES_PATH = Path(__file__).resolve().parent / "cases.json"
_STAGES = ("skill_selection", "tool_selection", "tool_args", "answer", "outcome", "end_to_end")


def load_cases(case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("cases.json 里有重复的用例 id。")
    if case_ids:
        unknown = set(case_ids) - set(ids)
        if unknown:
            raise ValueError(f"未知用例: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in case_ids]
    return cases


def evaluate(
    output: dict[str, Any],
    case_input: dict[str, Any],
    expected: dict[str, Any],
    *,
    judge: LLMClient | None,
    today: date | None = None,
) -> list[StageScore]:
    """All stage scores for one run, end_to_end last."""
    scores = score_deterministic(output, expected, today=today or date.today())
    rubric = expected.get("answer_rubric")
    if rubric and judge is not None:
        scores.append(
            judge_answer(
                judge,
                prompt=str(case_input.get("prompt")),
                reply=output.get("reply"),
                rubric=rubric,
                tool_calls=output.get("tool_calls") or [],
            )
        )
    scores.append(end_to_end(scores, error=output.get("error")))
    return scores


def metric_scores(output: dict[str, Any]) -> list[StageScore]:
    """Cost/latency alongside correctness, so regressions in either show up."""
    return [
        StageScore("model_calls", float(output.get("model_calls") or 0), ""),
        StageScore(
            "tokens",
            float((output.get("input_tokens") or 0) + (output.get("output_tokens") or 0)),
            "",
        ),
        StageScore("latency_s", float(output.get("elapsed_s") or 0), ""),
    ]


def build_judge(config: Config) -> LLMClient:
    profile = build_llm_profile(
        model=config.model,
        context_window_tokens=config.context_window_tokens,
        model_max_input_tokens=config.model_max_input_tokens,
        model_max_output_tokens=config.model_max_output_tokens,
        request_max_output_tokens=config.max_output_tokens,
    )
    return build_llm_client_from_profile(profile)


# ── Langfuse dataset + experiment ────────────────────────────────


def sync_dataset(langfuse: Any, cases: list[dict[str, Any]]) -> int:
    """Upsert every case; archive dataset items no longer in cases.json."""
    try:
        dataset = langfuse.get_dataset(DATASET)
    except Exception:
        langfuse.create_dataset(
            name=DATASET,
            description="MiniBot golden cases (source of truth: evals/cases.json)",
        )
        dataset = None
    wanted = {_item_id(case["id"]) for case in cases}
    for case in cases:
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            id=_item_id(case["id"]),
            input=case["input"],
            expected_output=case["expected"],
            metadata={
                "case_id": case["id"],
                "category": case.get("category"),
                "description": case.get("description"),
            },
        )
    if dataset is not None:
        for item in dataset.items:
            if item.id not in wanted and str(item.status).upper() != "ARCHIVED":
                langfuse.create_dataset_item(
                    dataset_name=DATASET, id=item.id, status="ARCHIVED"
                )
    return len(cases)


def run_experiment(
    langfuse: Any,
    *,
    config: Config,
    case_ids: list[str] | None,
    run_name: str,
) -> list[dict[str, Any]]:
    from langfuse import Evaluation

    judge = build_judge(config)
    wanted = {_item_id(case_id) for case_id in case_ids} if case_ids else None
    items = [
        item
        for item in langfuse.get_dataset(DATASET).items
        if str(item.status).upper() != "ARCHIVED" and (wanted is None or item.id in wanted)
    ]

    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        return run_case(item.input, config=config)

    def stages(*, input: Any, output: Any, expected_output: Any, **kwargs: Any) -> list[Any]:
        scores = evaluate(output, input, expected_output or {}, judge=judge)
        return [
            Evaluation(name=score.name, value=score.value, comment=score.comment)
            for score in [*scores, *metric_scores(output)]
        ]

    result = langfuse.run_experiment(
        name=DATASET,
        run_name=run_name,
        description="minibot golden eval",
        data=items,
        task=task,
        evaluators=[stages],
        max_concurrency=1,
        metadata=_run_metadata(config),
    )
    langfuse.flush()
    rows = []
    for item_result in result.item_results:
        meta = getattr(item_result.item, "metadata", None) or {}
        rows.append(
            {
                "case": meta.get("case_id") or getattr(item_result.item, "id", "?"),
                "output": item_result.output,
                "scores": {e.name: (e.value, e.comment) for e in item_result.evaluations},
            }
        )
    return rows


# ── local run ─────────────────────────────────────────────────────


def run_local(
    *,
    config: Config,
    case_ids: list[str] | None,
    run_name: str,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    judge = build_judge(config)
    rows = []
    for case in load_cases(case_ids):
        log(f"… {case['id']}")
        output = run_case(case["input"], config=config)
        scores = evaluate(output, case["input"], case["expected"], judge=judge)
        rows.append(
            {
                "case": case["id"],
                "output": output,
                "scores": {
                    s.name: (s.value, s.comment) for s in [*scores, *metric_scores(output)]
                },
            }
        )
    out_dir = resolve_state_home() / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.json"
    path.write_text(
        json.dumps(
            {"run": run_name, "metadata": _run_metadata(config), "results": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"结果已保存: {path}")
    return rows


# ── reporting ─────────────────────────────────────────────────────


def format_table(rows: list[dict[str, Any]]) -> str:
    header = ["case", "skill", "tools", "args", "answer", "outcome", "e2e", "calls", "tokens", "s"]
    lines = [header]
    totals: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    for row in rows:
        scores = row["scores"]
        cells = [row["case"]]
        for stage in _STAGES:
            if stage in scores:
                value = scores[stage][0]
                totals[stage].append(value)
                cells.append(_mark(value, stage))
            else:
                cells.append("—")
        for metric in ("model_calls", "tokens", "latency_s"):
            value = scores.get(metric, (None, ""))[0]
            cells.append("" if value is None else f"{value:g}")
        lines.append(cells)
    avg = ["平均"] + [
        f"{sum(v) / len(v):.0%}" if v else "—" for v in totals.values()
    ] + ["", "", ""]
    lines.append(avg)
    widths = [max(_width(line[i]) for line in lines) for i in range(len(header))]
    rendered = [
        "  ".join(_pad(cell, widths[i]) for i, cell in enumerate(line)) for line in lines
    ]
    failures = [
        f"  {row['case']} · {stage}: {comment}"
        for row in rows
        for stage, (value, comment) in row["scores"].items()
        if stage in _STAGES
        and stage != "end_to_end"
        and value < (0.5 if stage == "answer" else 1.0)
    ]
    report = "\n".join(rendered)
    if failures:
        report += "\n\n未通过的阶段:\n" + "\n".join(failures)
    return report


def default_run_name(config: Config) -> str:
    return f"{config.model}-{datetime.now():%Y%m%d-%H%M}-{_git_revision()}"


def _run_metadata(config: Config) -> dict[str, str]:
    return {"model": config.model, "git": _git_revision()}


def _git_revision() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _item_id(case_id: str) -> str:
    return f"{DATASET}--{case_id}"


def _mark(value: float, stage: str) -> str:
    if stage == "tool_args" and 0 < value < 1:
        return f"{value:.0%}"
    if stage == "answer":
        return {1.0: "✓", 0.5: "½"}.get(value, "✗")
    return "✓" if value >= 1 else "✗"


def _width(text: str) -> int:
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _width(text))


def langfuse_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )


__all__ = [
    "DATASET",
    "default_run_name",
    "evaluate",
    "format_table",
    "langfuse_configured",
    "load_cases",
    "run_experiment",
    "run_local",
    "sync_dataset",
]
