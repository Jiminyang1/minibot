"""Human review of drafted lessons.

State machine (anything else is refused)::

    draft ──approve──▶ approved ──retract──▶ rejected
      └────reject────────────────────────────▲

The transitions are plain functions over ``LessonStore``; ``run_review`` is
a thin interactive layer on top, with input/output injected for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .store import Lesson, LessonStatus, LessonStore, clean_rule, now_iso

_STATUS_LABELS = {"draft": "草稿", "approved": "已批准", "rejected": "已拒绝"}


class LessonStateError(ValueError):
    """A lesson is missing or not in the state a transition needs."""


def approve(
    store: LessonStore,
    lesson_id: str,
    *,
    trigger: str | None = None,
    guidance: str | None = None,
) -> Lesson:
    """draft → approved, optionally with edited wording."""

    def change(lesson: Lesson) -> Lesson:
        _require(lesson, "draft", "批准")
        new_trigger, new_guidance = clean_rule(
            lesson.trigger if trigger is None else trigger,
            lesson.guidance if guidance is None else guidance,
        )
        return replace(
            lesson,
            status="approved",
            trigger=new_trigger,
            guidance=new_guidance,
            reviewed_at=now_iso(),
        )

    return _apply(store, lesson_id, change)


def edit(
    store: LessonStore,
    lesson_id: str,
    *,
    trigger: str | None = None,
    guidance: str | None = None,
) -> Lesson:
    """Reword a draft; it stays a draft until approved."""

    def change(lesson: Lesson) -> Lesson:
        _require(lesson, "draft", "编辑")
        new_trigger, new_guidance = clean_rule(
            lesson.trigger if trigger is None else trigger,
            lesson.guidance if guidance is None else guidance,
        )
        return replace(lesson, trigger=new_trigger, guidance=new_guidance)

    return _apply(store, lesson_id, change)


def reject(store: LessonStore, lesson_id: str) -> Lesson:
    """draft → rejected."""

    def change(lesson: Lesson) -> Lesson:
        _require(lesson, "draft", "拒绝")
        return replace(lesson, status="rejected", reviewed_at=now_iso())

    return _apply(store, lesson_id, change)


def retract(store: LessonStore, lesson_id: str) -> Lesson:
    """approved → rejected: stop injecting a lesson that turned out wrong."""

    def change(lesson: Lesson) -> Lesson:
        _require(lesson, "approved", "撤回")
        return replace(lesson, status="rejected", reviewed_at=now_iso())

    return _apply(store, lesson_id, change)


def _apply(
    store: LessonStore, lesson_id: str, change: Callable[[Lesson], Lesson]
) -> Lesson:
    updated = store.update(lesson_id, change)
    if updated is None:
        raise LessonStateError(f"没有这条经验: {lesson_id}")
    return updated


def _require(lesson: Lesson, status: LessonStatus, action: str) -> None:
    if lesson.status != status:
        raise LessonStateError(
            f"{lesson.id} 是{_STATUS_LABELS[lesson.status]}，"
            f"只有{_STATUS_LABELS[status]}可以{action}。"
        )


# ── rendering ─────────────────────────────────────────────────────


def format_lesson(lesson: Lesson, *, evidence_limit: int | None = 5) -> str:
    lines = [
        f"{lesson.id}  [{_STATUS_LABELS[lesson.status]}]  创建于 {lesson.created_at}"
        + (f"  审核于 {lesson.reviewed_at}" if lesson.reviewed_at else ""),
        f"  触发: {lesson.trigger}",
        f"  做法: {lesson.guidance}",
        f"  证据: {len(lesson.evidence)} 条",
    ]
    shown = lesson.evidence if evidence_limit is None else lesson.evidence[:evidence_limit]
    for item in shown:
        where = item.session_id + (f" / {item.run_id}" if item.run_id else "")
        lines.append(f"    - ({item.kind}) {where}: {item.excerpt}")
    hidden = len(lesson.evidence) - len(shown)
    if hidden > 0:
        lines.append(f"    …… 另有 {hidden} 条，用 show {lesson.id} 查看全部")
    return "\n".join(lines)


def format_list_row(lesson: Lesson) -> str:
    return (
        f"{lesson.id:<8} {_STATUS_LABELS[lesson.status]:<4} "
        f"证据 {len(lesson.evidence):>3}  {lesson.trigger}"
    )


# ── interactive layer ─────────────────────────────────────────────

_REVIEW_HELP = "[a] 批准  [e] 编辑  [r] 拒绝  [s] 跳过  [q] 退出"


def run_review(
    store: LessonStore,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> dict[str, int]:
    """Walk the drafts once; returns counts per action taken."""
    counts = {"approved": 0, "rejected": 0, "skipped": 0}
    drafts = store.list(status="draft")
    if not drafts:
        output("没有待审核的草稿。")
        return counts
    output(f"共 {len(drafts)} 条草稿待审核。")
    for position, draft in enumerate(drafts, start=1):
        lesson: Lesson | None = draft
        while lesson is not None:
            output("")
            output(f"({position}/{len(drafts)})")
            output(format_lesson(lesson))
            choice = input_fn(f"{_REVIEW_HELP} > ").strip().lower()
            try:
                if choice == "a":
                    approve(store, lesson.id)
                    counts["approved"] += 1
                    output(f"已批准 {lesson.id}。")
                    lesson = None
                elif choice == "r":
                    reject(store, lesson.id)
                    counts["rejected"] += 1
                    output(f"已拒绝 {lesson.id}。")
                    lesson = None
                elif choice == "e":
                    trigger = input_fn("新的触发（回车保留原文）: ").strip()
                    guidance = input_fn("新的做法（回车保留原文）: ").strip()
                    lesson = edit(
                        store,
                        lesson.id,
                        trigger=trigger or None,
                        guidance=guidance or None,
                    )
                    output("已修改，请确认。")
                elif choice in {"s", ""}:
                    counts["skipped"] += 1
                    lesson = None
                elif choice == "q":
                    return counts
                else:
                    output(f"无效输入: {choice!r}")
            except LessonStateError as exc:
                # Someone else (another review, retract) changed it meanwhile.
                output(str(exc))
                lesson = None
    return counts
