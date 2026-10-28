"""Turn a real run's Langfuse trace into a draft golden case.

A trace (see ``langfuse_tracing``) holds everything needed to replay a turn:
the ``minibot.turn`` agent observation has the user prompt and the reply,
each ``tool`` observation has a call's arguments, and the next ``model``
generation's input holds the exact tool result the model saw. From that,
``draft_case`` builds:

- ``input.prompt``, plus ``memory`` (from the system prompt's memory block)
  and ``files`` (from ``read_file`` results inside the workspace), so the
  real file and memory tools see the same state;
- ``input.tool_results`` in the sandbox's format, so the stand-in tools
  (macOS MCP, ``exec``, ``web_search``, ``fetch_url``) return what the real
  ones returned — in order, via ``{"sequence": [...]}`` when called twice;
- ``expected``: the tools actually called, conservative argument checks
  (dates become relative to the run's day), and the skill it read.

The answer rubric is drafted separately by the judge model
(``draft_rubric``); a human reviews the whole draft before it is saved.
``draft_case`` is pure: it takes the fetched observations, so it can be
tested against a fabricated trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from ..llm import LLMClient
from ..runtime.messages import ModelMessage

TRACE_NAME = "minibot.turn"
_MACOS_PREFIX = "mcp__macos_system__"
_REPLAYED = {"exec", "web_search", "fetch_url"}
# Tools whose calls are bookkeeping, not behaviour worth asserting on.
_NOT_REQUIRED = {"read_skill", "read_artifact"}
_MEMORY_RE = re.compile(r"^- id: (mem_(\d+)); fact: (.*)$", re.MULTILINE)
_WORKSPACE_RE = re.compile(r"当前工作目录:\s*(\S+)")
_LOCAL_DATE_RE = re.compile(r"当前本地时间\D*(\d{4}-\d{2}-\d{2})")
_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{1,2}):(\d{2}))?")
_MAX_CONTAINS_CHARS = 40
_NEVER = datetime.min.replace(tzinfo=UTC)


@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    args: dict[str, Any]
    ok: bool | None
    code: str | None
    result: dict[str, Any] | None  # the model-facing result envelope


@dataclass
class Draft:
    case: dict[str, Any]
    prompt: str
    reply: str | None
    calls: list[ToolCallRecord]
    run_failed: bool
    warnings: list[str] = field(default_factory=list)


def draft_case(
    observations: list[Any],
    *,
    trace_id: str,
    case_id: str | None = None,
) -> Draft:
    """Draft a golden case from one trace's observations (any order)."""
    agents = sorted(
        (o for o in observations if _attr(o, "type") == "AGENT" and _attr(o, "name") == TRACE_NAME),
        key=_start,
    )
    if not agents:
        raise ValueError(f"trace {trace_id} 里没有 {TRACE_NAME}，不是 MiniBot 的运行。")
    warnings: list[str] = []
    if len(agents) > 1:
        warnings.append(f"trace 里有 {len(agents)} 个 turn，只用第一个。")
    agent = agents[0]
    inside = _descendants(observations, _attr(agent, "id"))

    prompt = _text(_decode(_attr(agent, "input")))
    if not prompt:
        raise ValueError("trace 里没有用户输入。")
    reply = _text(_decode(_attr(agent, "output"))) or None
    run_failed = str(_attr(agent, "level") or "").upper() == "ERROR"

    generations = sorted(
        (o for o in inside if _attr(o, "type") == "GENERATION" and _attr(o, "name") == "model"),
        key=_start,
    )
    requests = [_decode(_attr(g, "input")) for g in generations]
    requests = [r for r in requests if isinstance(r, list)]
    order, results = _calls_and_results(generations, requests)
    calls = _tool_calls(inside, order, results)

    system = next(
        (m.get("content") for r in requests[:1] for m in r if _role(m) == "system"), ""
    ) or ""
    today = _run_date(requests, agent)
    workspace = _first(_WORKSPACE_RE, system)

    case_input: dict[str, Any] = {"prompt": prompt}
    memory, memory_ids = _memory(system, warnings)
    if memory:
        case_input["memory"] = memory
    files = _files(calls, workspace, warnings)
    if files:
        case_input["files"] = files
    tool_results = _tool_results(calls, warnings)
    if tool_results:
        case_input["tool_results"] = tool_results
    approval = _approval(inside, warnings)
    if approval:
        case_input["approval"] = approval

    expected: dict[str, Any] = {}
    skills = [str(c.args.get("name")) for c in calls if c.name == "read_skill"]
    if skills:
        expected["skill"] = skills[0]
    required = list(dict.fromkeys(c.name for c in calls if c.name not in _NOT_REQUIRED))
    if required:
        expected["tools_required"] = required
    elif not calls:
        # A turn that used no tools is a claim worth keeping.
        expected["tools_forbidden"] = ["*"]
    args = _arg_checks(calls, today=today, workspace=workspace, memory_ids=memory_ids)
    if args:
        expected["args"] = args

    if run_failed:
        warnings.append("这次运行失败了：草稿描述的是它实际做了什么，请把 expected 改成正确行为。")
    failed_calls = [c.name for c in calls if c.ok is False and c.code != "denied"]
    if failed_calls:
        warnings.append(f"有失败的工具调用 {failed_calls}；回放会重现同样的失败结果。")

    case = {
        "id": case_id or f"promoted-{trace_id[:8]}",
        "category": "promoted",
        "description": f"从真实运行提升: {_preview(prompt, 40)}",
        "input": case_input,
        "expected": expected,
        "source": {"trace_id": trace_id},
    }
    return Draft(case, prompt, reply, calls, run_failed, warnings)


# ── LLM-drafted rubric ────────────────────────────────────────────

_RUBRIC_SYSTEM = (
    "你在为一个 AI 助手的回归评测写评分标准。根据用户输入、助手调用的工具和最终回复，"
    "写一条简短的评分标准（一到两句中文），说明一个正确的回复必须包含或做到什么。"
    "只写可核对的要点（关键事实、数值、确认的动作），不写措辞要求。"
    "如果回复本身有错，按正确行为来写。只输出评分标准本身，不要解释。"
)


def draft_rubric(
    llm: LLMClient,
    *,
    prompt: str,
    reply: str | None,
    tool_calls: list[ToolCallRecord],
) -> str:
    tools = ", ".join(f"{c.name}(ok={c.ok})" for c in tool_calls) or "无"
    user = (
        f"## 用户输入\n{prompt}\n\n"
        f"## 助手调用过的工具\n{tools}\n\n"
        f"## 助手最终回复\n{reply or '（运行失败，没有回复）'}"
    )
    response = llm.chat(
        [
            ModelMessage.create(role="system", content=_RUBRIC_SYSTEM),
            ModelMessage.create(role="user", content=user),
        ]
    )
    return (response.content or "").strip().strip("\"'“”「」").strip()


def append_case(path: Path, case: dict[str, Any]) -> None:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if any(existing["id"] == case["id"] for existing in cases):
        raise ValueError(f"用例 id 已存在: {case['id']}（用 --id 换一个）")
    cases.append(case)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── fetching from Langfuse ────────────────────────────────────────


def fetch_observations(langfuse: Any, trace_id: str) -> list[Any]:
    """Every observation of one trace, via the v2 observations API."""
    observations: list[Any] = []
    cursor = None
    while True:
        page = langfuse.api.observations.get_many(
            trace_id=trace_id,
            fields="core,basic,io,metadata",
            limit=1000,
            cursor=cursor,
        )
        observations.extend(page.data)
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return observations


def list_traces(langfuse: Any, *, limit: int = 20, failed: bool = False) -> list[dict[str, Any]]:
    """Recent MiniBot turns, newest first; with *failed*, only problem runs."""
    rows: list[dict[str, Any]] = []
    cursor = None
    for _ in range(10):  # at most 10 pages when filtering for failures
        page = langfuse.api.observations.get_many(
            name=TRACE_NAME,
            type="AGENT",
            fields="core,basic,io,trace_context",
            limit=min(max(limit, 1) * (3 if failed else 1), 100),
            cursor=cursor,
        )
        if not page.data:
            break
        oldest = min(_start(o) for o in page.data)
        tool_errors = langfuse.api.observations.get_many(
            type="TOOL", level="ERROR", from_start_time=oldest, fields="core", limit=1000
        ).data
        rows.extend(summarize_traces(page.data, tool_errors))
        if failed:
            rows = [row for row in rows if row["run_failed"] or row["failed_tools"]]
        cursor = getattr(page.meta, "cursor", None)
        if len(rows) >= limit or not cursor or not failed:
            break
    return rows[:limit]


def summarize_traces(agents: list[Any], tool_errors: list[Any]) -> list[dict[str, Any]]:
    """One row per agent observation: id, time, source, input, failure flags."""
    failed_tools: dict[str, int] = {}
    for tool in tool_errors:
        trace = str(_attr(tool, "trace_id"))
        failed_tools[trace] = failed_tools.get(trace, 0) + 1
    rows = []
    for agent in sorted(agents, key=_start, reverse=True):
        trace = str(_attr(agent, "trace_id"))
        level = str(_attr(agent, "level") or "DEFAULT").upper()
        tags = _attr(agent, "tags") or []
        rows.append(
            {
                "trace_id": trace,
                "time": _start(agent) if _attr(agent, "start_time") else None,
                "source": ",".join(tags) or "",
                "input": _preview(_text(_decode(_attr(agent, "input"))), 50),
                "run_failed": level == "ERROR",
                "warning": level == "WARNING",
                "failed_tools": failed_tools.get(trace, 0),
            }
        )
    return rows


def format_traces(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "没有找到 MiniBot 的 trace。"
    lines = []
    for row in rows:
        status = []
        if row["run_failed"]:
            status.append("运行失败")
        elif row["warning"]:
            status.append("警告")
        if row["failed_tools"]:
            status.append(f"{row['failed_tools']} 个工具失败")
        when = row["time"].astimezone().strftime("%m-%d %H:%M") if row["time"] else "?"
        state = "✗ " + "，".join(status) if status else "✓"
        lines.append(
            f"{row['trace_id']}  {when}  {row['source'] or '-':<6}  "
            f"{_pad(state, 18)}  {row['input']}"
        )
    return "\n".join(lines)


# ── trace parsing helpers ─────────────────────────────────────────


def _calls_and_results(
    generations: list[Any], requests: list[list[Any]]
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Call order from the model outputs; results from the next requests."""
    order: dict[str, int] = {}
    for generation in generations:
        output = _decode(_attr(generation, "output"))
        planned = output.get("tool_calls") if isinstance(output, dict) else None
        for call in planned or []:
            if isinstance(call, dict) and call.get("id"):
                order.setdefault(str(call["id"]), len(order))
    results: dict[str, dict[str, Any]] = {}
    for messages in requests:
        for message in messages:
            if _role(message) != "tool" or not message.get("tool_call_id"):
                continue
            envelope = _decode(message.get("content"))
            if isinstance(envelope, dict):
                results[str(message["tool_call_id"])] = envelope
    return order, results


def _tool_calls(
    inside: list[Any], order: dict[str, int], results: dict[str, dict[str, Any]]
) -> list[ToolCallRecord]:
    calls = []
    for tool in (o for o in inside if _attr(o, "type") == "TOOL"):
        meta = _decode(_attr(tool, "metadata")) or {}
        call_id = str(meta.get("tool_call_id") or _attr(tool, "id"))
        args = _decode(_attr(tool, "input"))
        output = _decode(_attr(tool, "output"))
        output = output if isinstance(output, dict) else {}
        result = results.get(call_id)
        calls.append(
            ToolCallRecord(
                call_id=call_id,
                name=str(_attr(tool, "name")),
                args=args if isinstance(args, dict) else {},
                ok=(result or output).get("ok"),
                code=(result or output).get("code"),
                result=result,
            )
        )
    calls.sort(key=lambda c: (order.get(c.call_id, len(order)), c.call_id))
    return calls


def _tool_results(calls: list[ToolCallRecord], warnings: list[str]) -> dict[str, Any]:
    per_tool: dict[str, list[Any]] = {}
    web_results: list[dict[str, Any]] = []
    pages: dict[str, str] = {}
    unreplayable: list[str] = []
    for call in calls:
        if call.code == "denied":
            continue  # never executed, so the stand-in is never asked either
        data = (call.result or {}).get("data") or {}
        if call.result is None and (call.name in _REPLAYED or call.name.startswith(_MACOS_PREFIX)):
            warnings.append(f"{call.name}: 没有记录到结果（运行在它之后结束），回放用默认结果。")
        if call.name.startswith(_MACOS_PREFIX):
            per_tool.setdefault(call.name, []).append(_mcp_payload(call, data))
        elif call.name.startswith("mcp__"):
            unreplayable.append(call.name)
        elif call.name == "exec":
            per_tool.setdefault("exec", []).append(
                {
                    "stdout": data.get("stdout", data.get("stdout_preview", "")),
                    "stderr": data.get("stderr", data.get("stderr_preview", "")),
                    "exit_code": data.get("exit_code", 0),
                }
                if call.result is not None
                else None
            )
        elif call.name == "web_search":
            for item in data.get("results") or []:
                if isinstance(item, dict) and item not in web_results:
                    web_results.append(item)
        elif call.name == "fetch_url" and call.ok:
            page = data.get("content") or data.get("preview")
            if isinstance(page, str):
                for url in {call.args.get("url"), data.get("url")} - {None}:
                    pages[str(url)] = page
    if unreplayable:
        names = sorted(set(unreplayable))
        warnings.append(f"沙箱里没有这些 MCP 工具，回放时模型看不到它们: {names}")

    tool_results: dict[str, Any] = {}
    for name, payloads in per_tool.items():
        if all(payload is None for payload in payloads):
            continue  # the stand-in's default is what we'd write anyway
        tool_results[name] = payloads[0] if len(payloads) == 1 else {"sequence": payloads}
    if web_results:
        tool_results["web_search"] = web_results
    if pages:
        tool_results["fetch_url"] = pages
    return tool_results


def _mcp_payload(call: ToolCallRecord, data: dict[str, Any]) -> Any:
    """What the sandbox MCP stand-in should return for this call."""
    if call.result is None:
        return None
    raw = data.get("content", data.get("preview"))
    content = _decode(raw) if isinstance(raw, str) else raw
    if call.ok is False:
        if isinstance(content, dict):
            blocks = content.get("content") or []
            text = "\n\n".join(str(b.get("text")) for b in blocks if isinstance(b, dict) and b.get("text"))
        else:
            text = str(raw or call.result.get("summary") or "")
        return {"error": text or str(call.result.get("summary"))}
    if isinstance(content, dict) and "structured_content" in content:
        structured = content["structured_content"]
        if isinstance(structured, dict):
            return structured
    if isinstance(content, dict):
        return content
    # A text-only result: the stand-in only returns structured content.
    return {"result": raw} if raw is not None else None


def _files(
    calls: list[ToolCallRecord], workspace: str | None, warnings: list[str]
) -> dict[str, str]:
    files: dict[str, str] = {}
    outside: list[str] = []
    for call in calls:
        if call.name != "read_file" or not call.ok or call.result is None:
            continue
        data = call.result.get("data") or {}
        content = data.get("content")
        path = str(call.args.get("path") or data.get("path") or "")
        if not isinstance(content, str) or call.result.get("truncated"):
            warnings.append(f"read_file {path}: 结果被截断，没有种入沙箱。")
            continue
        relative = _relative(path, workspace)
        if relative is None:
            outside.append(path)
            continue
        files.setdefault(relative, content)
    if outside:
        warnings.append(f"工作区外的文件没有种入沙箱: {outside}")
    return files


def _relative(path: str, workspace: str | None) -> str | None:
    pure = PurePosixPath(path)
    if not pure.is_absolute():
        parts = [p for p in pure.parts if p not in {".", ""}]
        return None if ".." in parts or not parts else str(PurePosixPath(*parts))
    if workspace and pure.is_relative_to(workspace):
        rel = pure.relative_to(workspace)
        return str(rel) if rel.parts else None
    return None


def _memory(system: str, warnings: list[str]) -> tuple[list[str], dict[str, str]]:
    """Facts in id order; the sandbox renumbers them from mem_1."""
    found = sorted(
        ((int(number), mem_id, fact.strip()) for mem_id, number, fact in _MEMORY_RE.findall(system)),
    )
    ids = {mem_id: f"mem_{index}" for index, (_, mem_id, _) in enumerate(found, start=1)}
    if any(old != new for old, new in ids.items()):
        warnings.append(f"记忆 id 在沙箱里会重新编号: {ids}")
    return [fact for _, _, fact in found], ids


def _approval(inside: list[Any], warnings: list[str]) -> str | None:
    decisions = []
    for event in inside:
        if _attr(event, "name") != "approval.resolved":
            continue
        meta = _decode(_attr(event, "metadata")) or {}
        if isinstance(meta, dict) and "approved" in meta:
            decisions.append(meta["approved"] in (True, "true", "True"))
    if not decisions or all(decisions):
        return None
    if any(decisions):
        warnings.append("这次运行里审批有批准也有拒绝；用例只能设一种，草稿按全部批准处理。")
        return None
    return "deny"


def _arg_checks(
    calls: list[ToolCallRecord],
    *,
    today: date,
    workspace: str | None,
    memory_ids: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Per tool, checks drafted from its last successful call."""
    by_tool: dict[str, ToolCallRecord] = {}
    for call in calls:
        if call.name in _NOT_REQUIRED:
            continue
        if call.name not in by_tool or call.ok or not by_tool[call.name].ok:
            by_tool[call.name] = call
    specs: dict[str, dict[str, Any]] = {}
    for name, call in by_tool.items():
        checks = {}
        for key, value in call.args.items():
            check = _check_for(key, value, today=today, workspace=workspace, memory_ids=memory_ids)
            if check:
                checks[key] = check
        if checks:
            specs[name] = checks
    return specs


def _check_for(
    key: str,
    value: Any,
    *,
    today: date,
    workspace: str | None,
    memory_ids: dict[str, str],
) -> dict[str, Any] | None:
    if isinstance(value, (bool, int, float)):
        return {"equals": value}
    if not isinstance(value, str) or not value.strip():
        return None
    if key == "memory_id":
        return {"equals": memory_ids.get(value, value)}
    iso = _ISO_RE.match(value)
    if iso:
        days = (date.fromisoformat(iso.group(1)) - today).days
        if days < 0:
            return None
        check: dict[str, Any] = {"date": f"+{days}d"}
        if iso.group(2):
            check["time"] = f"{int(iso.group(2)):02d}:{iso.group(3)}"
        return check
    if key in {"path", "file", "file_path"}:
        relative = _relative(value, workspace)
        name = PurePosixPath(relative or value).name
        return {"contains": name} if name else None
    if len(value) <= _MAX_CONTAINS_CHARS:
        return {"contains": value}
    return None  # long free text (file bodies, essays) is the judge's job


def _run_date(requests: list[list[Any]], agent: Any) -> date:
    """The user's local day of the run, from the time note on the prompt."""
    for messages in requests[:1]:
        for message in reversed(messages):
            if _role(message) == "user":
                found = _first(_LOCAL_DATE_RE, _text(message.get("content")))
                if found:
                    return date.fromisoformat(found)
                break
    started = _start(agent)
    return started.astimezone().date() if started is not _NEVER else date.today()


def _descendants(observations: list[Any], root_id: Any) -> list[Any]:
    children: dict[Any, list[Any]] = {}
    for o in observations:
        children.setdefault(_attr(o, "parent_observation_id"), []).append(o)
    found, stack = [], [root_id]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(_attr(child, "id"))
    return found


def _decode(value: Any) -> Any:
    """The v2 API returns input/output as raw strings; JSON-decode when it is JSON."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped[:1] in {"{", "[", '"'}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # content parts
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in value
        )
    return json.dumps(value, ensure_ascii=False)


def _attr(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _start(obj: Any) -> datetime:
    started = _attr(obj, "start_time")
    if isinstance(started, str):
        started = datetime.fromisoformat(started.replace("Z", "+00:00"))
    return started or _NEVER


def _role(message: Any) -> Any:
    return message.get("role") if isinstance(message, dict) else None


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    found = pattern.search(text or "")
    return found.group(1) if found else None


def _pad(text: str, width: int) -> str:
    shown = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)  # CJK is two columns
    return text + " " * max(width - shown, 0)


def _preview(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


__all__ = [
    "Draft",
    "ToolCallRecord",
    "append_case",
    "draft_case",
    "draft_rubric",
    "fetch_observations",
    "format_traces",
    "list_traces",
    "summarize_traces",
]
