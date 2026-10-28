"""``python -m minibot.lessons`` — the lessons loop from the command line.

    mine [--dry-run]      mine failures, cluster them, draft lessons (LLM)
         [--threshold T]  clustering cosine threshold
         [--min-size N]   smallest cluster worth a draft
         [--limit N]      most clusters drafted per run
    review                approve / edit / reject drafts interactively
    list [--status S]     list lessons
    show <id>             one lesson with all its evidence
    retract <id>          approved → rejected (stop injecting it)

The store is ``MINIBOT_LESSONS_PATH`` (default ``$MINIBOT_HOME/lessons.json``).
``mine`` reads sessions and runs.jsonl under ``$MINIBOT_HOME``; without
``--dry-run`` it calls the configured chat model once per cluster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..config import Config, load_env, resolve_state_home
from .cluster import DEFAULT_CLUSTER_THRESHOLD
from .pipeline import DEFAULT_MAX_CLUSTERS, DEFAULT_MIN_CLUSTER_SIZE
from .review import LessonStateError, format_lesson, format_list_row, retract, run_review
from .store import LESSON_STATUSES, LessonStore

_ACTION_LABELS = {
    "drafted": "新草稿",
    "attached": "并入已有经验",
    "suppressed": "与已拒绝经验重复，跳过",
    "no_rule": "无可复用规则",
    "skipped": "失败，下次重试",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minibot.lessons")
    sub = parser.add_subparsers(dest="command", required=True)
    mine = sub.add_parser("mine", help="挖掘失败记录并起草经验")
    mine.add_argument("--dry-run", action="store_true", help="只展示事件和聚类，不调用模型、不写入")
    mine.add_argument("--threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD, help="聚类余弦阈值")
    mine.add_argument("--min-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE, help="起草所需的最小簇大小")
    mine.add_argument("--limit", type=int, default=DEFAULT_MAX_CLUSTERS, help="本次最多起草的簇数")
    sub.add_parser("review", help="逐条审核草稿")
    listing = sub.add_parser("list", help="列出经验")
    listing.add_argument("--status", choices=LESSON_STATUSES, help="只看某个状态")
    show = sub.add_parser("show", help="查看一条经验")
    show.add_argument("lesson_id")
    retract_cmd = sub.add_parser("retract", help="撤回一条已批准的经验")
    retract_cmd.add_argument("lesson_id")
    args = parser.parse_args(argv)

    load_env()
    config = Config.from_env()
    state_home = resolve_state_home()
    store = LessonStore(config.resolve_lessons_path(state_home))

    if args.command == "mine":
        return _mine(args, config, store, state_home)
    if args.command == "review":
        try:
            counts = run_review(store)
        except (EOFError, KeyboardInterrupt):
            print()
            print("已退出审核；已做的决定都已保存。")
            return 0
        print(
            f"审核结束：批准 {counts['approved']}，拒绝 {counts['rejected']}，"
            f"跳过 {counts['skipped']}。"
        )
        return 0
    if args.command == "list":
        lessons = store.list(status=args.status)
        if not lessons:
            print("没有经验。")
        for lesson in lessons:
            print(format_list_row(lesson))
        return 0
    if args.command == "show":
        lesson = store.get(args.lesson_id)
        if lesson is None:
            print(f"没有这条经验: {args.lesson_id}", file=sys.stderr)
            return 1
        print(format_lesson(lesson, evidence_limit=None))
        return 0
    if args.command == "retract":
        try:
            lesson = retract(store, args.lesson_id)
        except LessonStateError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"已撤回 {lesson.id}，之后不会再注入。")
        return 0
    return 2


def _mine(
    args: argparse.Namespace, config: Config, store: LessonStore, state_home: Path
) -> int:
    from .embed import FastEmbedEmbedder, fastembed_available
    from .pipeline import plan_mining, run_mining

    if not fastembed_available():
        print("挖掘需要 fastembed：请运行 uv sync --extra lessons。", file=sys.stderr)
        return 1
    embedder = FastEmbedEmbedder()
    plan = plan_mining(
        state_home,
        store,
        embedder,
        threshold=args.threshold,
        min_cluster_size=args.min_size,
        max_clusters=args.limit,
    )
    print(
        f"事件 {plan.total_incidents} 条，新增 {len(plan.new_incidents)} 条；"
        f"待起草 {len(plan.clusters)} 簇，暂缓 {len(plan.deferred)} 簇。"
    )
    if args.dry_run:
        for index, cluster in enumerate(plan.clusters, start=1):
            print()
            print(f"簇 {index}（{len(cluster)} 条）")
            for incident in cluster[:5]:
                print(f"  - [{incident.kind}] {incident.session_id}: {incident.problem}")
                print(f"      请求: {incident.request}")
                if incident.correction:
                    print(f"      纠正: {incident.correction}")
            if len(cluster) > 5:
                print(f"  …… 另有 {len(cluster) - 5} 条")
        print()
        print("dry run：未调用模型，未写入经验库。")
        return 0
    if not plan.clusters:
        return 0

    from ..llm_factory import build_llm_client_from_profile
    from ..llm_profile import build_llm_profile

    llm = build_llm_client_from_profile(
        build_llm_profile(
            model=config.model,
            context_window_tokens=config.context_window_tokens,
            model_max_input_tokens=config.model_max_input_tokens,
            model_max_output_tokens=config.model_max_output_tokens,
            request_max_output_tokens=config.max_output_tokens,
        )
    )
    report = run_mining(plan, store=store, llm=llm, embedder=embedder)
    for outcome in report.outcomes:
        label = _ACTION_LABELS[outcome.action]
        target = f" {outcome.lesson_id}" if outcome.lesson_id else ""
        detail = f"（{outcome.detail}）" if outcome.detail else ""
        print(f"  {len(outcome.incident_ids):>3} 条 → {label}{target}{detail}")
    print(
        f"新草稿 {report.count('drafted')} 条，并入 {report.count('attached')} 条。"
        "运行 python -m minibot.lessons review 审核。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
