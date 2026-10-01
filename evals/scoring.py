"""Score one sandboxed run against its golden expectation, stage by stage.

The four core stages mirror how a turn unfolds:

1. ``skill_selection`` — did it load the right skill (or correctly none)?
2. ``tool_selection``  — did it call the tools it must, and none it must not?
3. ``tool_args``       — were the arguments right (paths, dates, ids, text)?
4. ``answer``          — does the final reply satisfy the rubric? (LLM judge)

plus ``outcome`` (state after the run, e.g. file contents) and ``end_to_end``
(every applicable stage passed). A stage the case doesn't specify is skipped,
not scored, so averages only cover what each case actually tests.

Expectation format (``expected`` in a golden case)::

    skill:           "reminders" | null (must read none) | absent (skip)
    tools_required:  ["mcp__macos_system__reminders_create", ...]
    tools_forbidden: ["exec", ...] or ["*"] (no tool calls at all)
    args:            {tool_name: {arg: check}}  — any one call may satisfy it
    files_after:     {path: check}
    answer_rubric:   "what a correct reply must contain or do"

A check is a dict whose keys must all hold: ``equals``, ``contains``
(str or list), ``not_contains``, ``regex``, ``date`` (``+Nd`` relative to
today, or ``nextweek:mon..sun``) and ``time`` (``HH:MM``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import re
from typing import Any

from ..llm import LLMClient
from ..runtime.messages import ModelMessage

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_TIME_RE = re.compile(r"(?:T|\s|^)(\d{1,2}):(\d{2})")
ANSWER_PASS = 0.5


@dataclass(frozen=True)
class StageScore:
    name: str
    value: float
    comment: str


def score_deterministic(
    output: dict[str, Any],
    expected: dict[str, Any],
    *,
    today: date,
) -> list[StageScore]:
    """Every stage except the judged answer; skipped stages are omitted."""
    scores: list[StageScore] = []
    calls = output.get("tool_calls") or []
    called = [str(call.get("name")) for call in calls]

    if "skill" in expected:
        want = expected["skill"]
        read = output.get("skills_read") or []
        if want is None:
            ok = not read
            comment = "未读取 skill" if ok else f"不该读取却读了 {read}"
        else:
            ok = want in read
            comment = f"读取了 {read}" if read else "没有读取任何 skill"
        scores.append(StageScore("skill_selection", float(ok), comment))

    required = expected.get("tools_required") or []
    forbidden = expected.get("tools_forbidden") or []
    if required or forbidden:
        missing = [name for name in required if name not in called]
        if "*" in forbidden:
            violated = called
        else:
            violated = [name for name in called if name in forbidden]
        problems = []
        if missing:
            problems.append(f"缺少 {missing}")
        if violated:
            problems.append(f"不该调用 {violated}")
        scores.append(
            StageScore(
                "tool_selection",
                float(not problems),
                "；".join(problems) or f"调用了 {called or '无'}",
            )
        )

    arg_specs: dict[str, dict[str, Any]] = expected.get("args") or {}
    if arg_specs:
        passed = 0
        notes: list[str] = []
        for tool, checks in arg_specs.items():
            candidates = [call for call in calls if call.get("name") == tool]
            failures = [
                _check_args(call.get("args") or {}, checks, today) for call in candidates
            ]
            if any(not failure for failure in failures):
                passed += 1
            elif not candidates:
                notes.append(f"{tool}: 未调用")
            else:
                notes.append(f"{tool}: {failures[0]}")
        scores.append(
            StageScore(
                "tool_args",
                passed / len(arg_specs),
                "；".join(notes) or "参数全部符合",
            )
        )

    file_specs: dict[str, Any] = expected.get("files_after") or {}
    if file_specs:
        files = output.get("files") or {}
        notes = []
        for path, check in file_specs.items():
            if path not in files:
                notes.append(f"{path}: 文件不存在")
                continue
            failure = _check_value(files[path], check, today)
            if failure:
                notes.append(f"{path}: {failure}")
        scores.append(
            StageScore("outcome", float(not notes), "；".join(notes) or "文件状态符合")
        )

    return scores


def end_to_end(scores: list[StageScore], *, error: str | None) -> StageScore:
    """Pass only if the run finished and every scored stage passed."""
    if error:
        return StageScore("end_to_end", 0.0, f"运行失败: {error}")
    failed = [
        score.name
        for score in scores
        if score.value < (ANSWER_PASS if score.name == "answer" else 1.0)
    ]
    return StageScore(
        "end_to_end",
        float(not failed),
        f"未通过: {failed}" if failed else "全部阶段通过",
    )


# ── LLM judge ─────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "你是严格的 AI 助手评测员。根据评分标准判断助手的最终回复是否合格。"
    "只依据给出的对话内容判断，不要脑补助手没说的话。"
    '只输出一个 JSON 对象：{"score": 0 或 0.5 或 1, "reason": "一句话理由"}。'
    "1 = 完全满足标准；0.5 = 部分满足或有明显瑕疵；0 = 不满足。"
)


def judge_answer(
    llm: LLMClient,
    *,
    prompt: str,
    reply: str | None,
    rubric: str,
    tool_calls: list[dict[str, Any]],
) -> StageScore:
    if not reply:
        return StageScore("answer", 0.0, "没有最终回复")
    tools = ", ".join(str(call.get("name")) for call in tool_calls) or "无"
    user = (
        f"## 用户输入\n{prompt}\n\n"
        f"## 助手调用过的工具\n{tools}\n\n"
        f"## 助手最终回复\n{reply}\n\n"
        f"## 评分标准\n{rubric}"
    )
    response = llm.chat(
        [
            ModelMessage.create(role="system", content=_JUDGE_SYSTEM),
            ModelMessage.create(role="user", content=user),
        ]
    )
    return _parse_judgement(response.content or "")


def _parse_judgement(text: str) -> StageScore:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
        value = float(data.get("score"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return StageScore("answer", 0.0, f"评测员输出无法解析: {text[:120]}")
    value = min(max(value, 0.0), 1.0)
    return StageScore("answer", value, str(data.get("reason") or ""))


# ── checks ────────────────────────────────────────────────────────


def _check_args(args: dict[str, Any], checks: dict[str, Any], today: date) -> str:
    """Empty string when every argument check holds, else the first failure."""
    for name, check in checks.items():
        if name not in args or args[name] is None:
            return f"缺少参数 {name}"
        failure = _check_value(args[name], check, today)
        if failure:
            return f"{name} {failure}"
    return ""


def _check_value(value: Any, check: dict[str, Any], today: date) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if "equals" in check and value != check["equals"]:
        return f"应为 {check['equals']!r}，实际 {value!r}"
    for needle in _as_list(check.get("contains")):
        if needle not in text:
            return f"应包含 {needle!r}，实际 {text[:80]!r}"
    for needle in _as_list(check.get("not_contains")):
        if needle in text:
            return f"不应包含 {needle!r}"
    if "regex" in check and not re.search(check["regex"], text):
        return f"不匹配 {check['regex']!r}，实际 {text[:80]!r}"
    if "date" in check:
        want = resolve_date(check["date"], today).isoformat()
        found = _DATE_RE.search(text)
        if not found or found.group(1) != want:
            return f"日期应为 {want}，实际 {text[:40]!r}"
    if "time" in check:
        hour, minute = (int(part) for part in str(check["time"]).split(":"))
        found = _TIME_RE.search(text)
        if not found or (int(found.group(1)), int(found.group(2))) != (hour, minute):
            return f"时间应为 {check['time']}，实际 {text[:40]!r}"
    return ""


def resolve_date(spec: str, today: date) -> date:
    """``+Nd`` → today+N; ``nextweek:wed`` → that weekday of next (Mon-start) week."""
    if spec.startswith("+") and spec.endswith("d"):
        return today + timedelta(days=int(spec[1:-1]))
    if spec.startswith("nextweek:"):
        weekday = _WEEKDAYS[spec.split(":", 1)[1]]
        next_monday = today + timedelta(days=7 - today.weekday())
        return next_monday + timedelta(days=weekday)
    raise ValueError(f"未知日期规则: {spec}")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)
