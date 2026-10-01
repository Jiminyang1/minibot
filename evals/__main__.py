"""``python -m minibot.evals`` — golden evals against a real model.

    sync                 upsert evals/cases.json into the Langfuse dataset
    run [--case ID ...]  run cases in the sandbox and score every stage
        [--name NAME]    experiment run name (default: model-time-gitsha)
        [--local]        skip Langfuse; save results under $MINIBOT_HOME/evals

Each run calls the configured model for real (cost: a few cents per run).
"""

from __future__ import annotations

import argparse

from ..config import Config, load_env
from .runner import (
    default_run_name,
    format_table,
    langfuse_configured,
    load_cases,
    run_experiment,
    run_local,
    sync_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minibot.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="把 cases.json 同步到 Langfuse dataset")
    run = sub.add_parser("run", help="运行评测并打分")
    run.add_argument("--case", action="append", dest="cases", help="只跑指定用例，可重复")
    run.add_argument("--name", help="本次运行名")
    run.add_argument("--local", action="store_true", help="不使用 Langfuse")
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

    run_name = args.name or default_run_name(config)
    print(f"评测 {run_name}（模型 {config.model}）")
    if args.local or not langfuse_configured():
        rows = run_local(config=config, case_ids=args.cases, run_name=run_name)
    else:
        from langfuse import get_client

        langfuse = get_client()
        sync_dataset(langfuse, load_cases())
        rows = run_experiment(
            langfuse, config=config, case_ids=args.cases, run_name=run_name
        )
        print("Langfuse: Datasets → minibot-golden → Runs 里可对比各次运行。")
    print()
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
