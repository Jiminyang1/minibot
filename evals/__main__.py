"""``python -m minibot.evals`` — golden evals against a real model.

    sync                 upsert evals/cases.json into the Langfuse dataset
    run [--case ID ...]  run cases in the sandbox and score every stage
        [--name NAME]    experiment run name (default: model-time-gitsha)
        [--local]        skip Langfuse; save results under $MINIBOT_HOME/evals
    pytest [ARGS ...]    the same cases as pytest, one test per (case, stage);
                         runs ``pytest evals -m golden ARGS`` (see test_golden.py)
    traces [--limit N] [--failed]
                         list recent MiniBot traces to pick one to promote
    promote TRACE_ID [--id CASE_ID] [--yes] [--cases PATH]
                         draft a case from a real run's trace, confirm, append
                         it to cases.json and sync the dataset

``run`` ends with a failure-localization report (first failing stage per
case, with its trace) saved as ``<run>.failures.json`` next to the results.
Each run calls the configured model for real (cost: a few cents per run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from ..config import Config, load_env
from .report import (
    compare_runs,
    failure_report,
    format_comparison,
    format_failure_report,
    save_failure_report,
)
from .runner import (
    CASES_PATH,
    build_judge,
    default_run_name,
    format_table,
    langfuse_configured,
    load_cases,
    run_experiment,
    run_local,
    sync_dataset,
    trace_url,
    with_lessons,
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["pytest"]:
        # Not argparse: everything after "pytest" belongs to pytest.
        return run_pytest(argv[1:])

    parser = argparse.ArgumentParser(prog="python -m minibot.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="把 cases.json 同步到 Langfuse dataset")
    run = sub.add_parser("run", help="运行评测并打分")
    run.add_argument("--case", action="append", dest="cases", help="只跑指定用例，可重复")
    run.add_argument("--name", help="本次运行名")
    run.add_argument("--local", action="store_true", help="不使用 Langfuse")
    run.add_argument(
        "--lessons",
        choices=("off", "on", "compare"),
        default="off",
        help="是否注入已批准的 lesson；compare = 各跑一次并对比（默认 off）",
    )
    sub.add_parser("pytest", help="用 pytest 逐阶段断言（参数原样传给 pytest）")
    traces = sub.add_parser("traces", help="列出最近的 MiniBot trace")
    traces.add_argument("--limit", type=int, default=20, help="最多列出几条（默认 20）")
    traces.add_argument("--failed", action="store_true", help="只看运行失败或有工具失败的")
    promote = sub.add_parser("promote", help="把一次真实运行的 trace 变成用例")
    promote.add_argument("trace_id")
    promote.add_argument("--id", dest="case_id", help="用例 id（默认 promoted-<trace 前 8 位>）")
    promote.add_argument("--yes", action="store_true", help="不确认，直接写入")
    promote.add_argument("--cases", type=Path, default=CASES_PATH, help="写入哪个 cases.json")
    args = parser.parse_args(argv)

    load_env()
    config = Config.from_env()

    if args.command == "sync":
        if not langfuse_configured():
            parser.error("未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY。")
        from langfuse import get_client

        count = sync_dataset(get_client(), load_cases())
        print(f"已同步 {count} 个用例到 Langfuse dataset。")
        return 0

    if args.command in {"traces", "promote"}:
        if not langfuse_configured():
            parser.error("未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY。")
        from langfuse import get_client

        if args.command == "traces":
            return _traces(get_client(), limit=args.limit, failed=args.failed)
        return _promote(get_client(), config, args)

    base_name = args.name or default_run_name(config)
    lesson_modes = ("off", "on") if args.lessons == "compare" else (args.lessons,)
    local = args.local or not langfuse_configured()
    results = {}
    for lessons in lesson_modes:
        run_name = base_name if args.lessons == "off" else f"{base_name}-lessons-{lessons}"
        results[lessons] = _run_once(
            with_lessons(config, lessons), run_name=run_name, cases=args.cases, local=local
        )
    if args.lessons == "compare":
        print()
        print(format_comparison(compare_runs(results["off"], results["on"])))
    return 0


def _run_once(
    config: Config,
    *,
    run_name: str,
    cases: list[str] | None,
    local: bool,
) -> list[dict[str, Any]]:
    print(f"评测 {run_name}（模型 {config.model}，lesson {config.lessons_mode}）")
    if local:
        mode = "local"
        rows = run_local(config=config, case_ids=cases, run_name=run_name)
    else:
        from langfuse import get_client

        mode = "langfuse"
        langfuse = get_client()
        sync_dataset(langfuse, load_cases())
        rows = run_experiment(langfuse, config=config, case_ids=cases, run_name=run_name)
        print("Langfuse: Datasets → minibot-golden → Runs 里可对比各次运行。")
    print()
    print(format_table(rows))
    report = failure_report(rows, run_name=run_name, mode=mode)
    print()
    print(format_failure_report(report))
    print(f"失败定位已保存: {save_failure_report(report)}")
    return rows


def run_pytest(args: list[str]) -> int:
    import pytest

    # Our -m comes first so an explicit -m in *args* wins.
    return int(pytest.main([str(Path(__file__).resolve().parent), "-m", "golden", *args]))


def _traces(langfuse: Any, *, limit: int, failed: bool) -> int:
    from .promote import format_traces, list_traces

    print(format_traces(list_traces(langfuse, limit=limit, failed=failed)))
    print("\n用 python -m minibot.evals promote <trace_id> 把某次运行变成用例。")
    return 0


def _promote(langfuse: Any, config: Config, args: argparse.Namespace) -> int:
    from .promote import append_case, draft_case, draft_rubric, fetch_observations

    observations = fetch_observations(langfuse, args.trace_id)
    if not observations:
        print(f"找不到 trace {args.trace_id}（刚结束的运行可能要等几分钟才能查到）。")
        return 1
    try:
        draft = draft_case(observations, trace_id=args.trace_id, case_id=args.case_id)
    except ValueError as exc:
        print(exc)
        return 1
    case = draft.case
    print("正在让评测员模型起草 answer_rubric …")
    case["expected"]["answer_rubric"] = draft_rubric(
        build_judge(config), prompt=draft.prompt, reply=draft.reply, tool_calls=draft.calls
    )
    url = trace_url(langfuse, args.trace_id)
    if url:
        case["source"]["trace_url"] = url

    print(f"\n用户输入: {draft.prompt}")
    print(f"原回复: {draft.reply or '（无）'}")
    print("\n用例草稿:")
    print(json.dumps(case, ensure_ascii=False, indent=2))
    for warning in draft.warnings:
        print(f"注意: {warning}")
    if not args.yes:
        answer = input(f"\n写入 {args.cases} 并同步 dataset？[y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消，没有写入。")
            return 1
    try:
        append_case(args.cases, case)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"已追加用例 {case['id']} 到 {args.cases}")
    count = sync_dataset(langfuse, load_cases(path=args.cases))
    langfuse.flush()
    print(f"已同步 {count} 个用例到 Langfuse dataset。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
