"""Terminal UI helpers for MiniBot.

Single source of truth for ANSI styling and every line of user-facing
output. Both the REPL (`cli.py`) and turn-time callbacks (tool events,
approval prompts) go through here so the app has one consistent look.

No domain knowledge lives here: functions take plain values (sessions,
memory items, strings) and produce styled output.
"""

from __future__ import annotations

import os
import sys
import json
from collections.abc import Iterable
from typing import Any

from .mcp_host import MCPHostSummary, MCPServerStatus
from .runtime.events import RuntimeEvent
from .user_memory import MemoryItem
from .session import Session


# ── colour primitives ────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_STYLES = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "cyan": "\x1b[36m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "magenta": "\x1b[35m",
    "gray": "\x1b[90m",
    "blue": "\x1b[34m",
}


def c(text: str, *styles: str) -> str:
    """Wrap *text* in ANSI styles. No-op when the stream is not a TTY."""
    if not _USE_COLOR:
        return text
    codes = "".join(_STYLES[s] for s in styles if s in _STYLES)
    return f"{codes}{text}{_STYLES['reset']}"


RULE = c("─" * 46, "gray")


# ── generic semantic prints ──────────────────────────────────────


def info(msg: str) -> None:
    print(c(f"  {msg}", "gray"))


def success(msg: str) -> None:
    print(c(f"  {msg}", "green"))


def warn(msg: str) -> None:
    print(c(f"  {msg}", "yellow"))


# ── turn-time callbacks ──────────────────────────────────────────


def tool_log(msg: str) -> None:
    """Print a single tool/runtime event during a turn."""
    print(f"  {c('›', 'dim')} {c(msg, 'dim')}")


def print_runtime_event(event: RuntimeEvent) -> None:
    """Render one structured runtime event for the terminal UI."""
    message = format_runtime_event(event)
    if message:
        tool_log(message)


def format_runtime_event(event: RuntimeEvent) -> str | None:
    payload = event.payload
    if event.type == "context.usage":
        return (
            "当前上下文占用(不含本次输入): "
            f"{payload.get('current_tokens')}/{payload.get('budget')} tokens"
        )
    if event.type == "model.request.started":
        return f"第 {payload.get('iteration')} 轮: 请求模型..."
    if event.type == "model.request.completed":
        if payload.get("empty_reply"):
            debug = payload.get("response_debug")
            return (
                f"第 {payload.get('iteration')} 轮: 模型返回空回答 · "
                f"{json.dumps(debug, ensure_ascii=False, default=str)}"
            )
        return (
            f"第 {payload.get('iteration')} 轮: 模型返回 "
            f"{payload.get('tool_call_count', 0)} 个工具调用"
        )
    if event.type == "tool_call.started":
        args = json.dumps(payload.get("args") or {}, ensure_ascii=False, default=str)
        label = payload.get("display_name") or payload.get("tool")
        prefix = "MCP 调用" if payload.get("source") == "mcp" else "工具"
        return f"{prefix}: {label}({args})"
    if event.type in {"tool_call.completed", "tool_call.failed"}:
        prefix = "MCP 返回" if payload.get("source") == "mcp" else "返回"
        if event.type == "tool_call.failed":
            prefix = "MCP 失败" if payload.get("source") == "mcp" else "工具失败"
        return f"{prefix}: {payload.get('summary')}"
    if event.type == "approval.required":
        return f"等待批准: {payload.get('tool')}"
    if event.type == "approval.resolved":
        state = "已批准" if payload.get("approved") else "已拒绝"
        return f"{state}: {payload.get('tool')}"
    if event.type == "message.completed":
        iteration = payload.get("iteration")
        if iteration is None:
            return "最终回答"
        return f"第 {iteration} 轮: 最终回答"
    if event.type == "run.failed":
        return f"运行失败: {payload.get('error_type')}: {payload.get('message')}"
    return None


def prompt_approval(tool_name: str, args: dict[str, Any]) -> bool:
    """Ask the user whether to run a sensitive tool call."""
    preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
    label = c("批准执行", "yellow", "bold")
    hint = c("[y/N]", "gray")
    line = f"  {label}  {c(tool_name, 'cyan')}({preview}) {hint} "
    return input(line).strip().lower() in {"y", "yes"}


# ── REPL input/output ────────────────────────────────────────────


def read_user_input() -> str:
    return input(c("\n  You › ", "bold", "cyan")).strip()


def print_agent_reply(reply: str) -> None:
    visible_reply = reply if reply.strip() else c("（模型返回空回复）", "yellow")
    print(f"\n  {c('Agent ›', 'bold', 'magenta')} {visible_reply}")


# ── REPL panels ──────────────────────────────────────────────────


_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/sessions", "查看所有会话"),
    ("/new", "新建会话"),
    ("/resume <id>", "恢复指定会话"),
    ("/delete <id|current>", "删除会话"),
    ("/compact", "压缩当前会话"),
    ("/mcp", "查看 MCP server 状态"),
    ("/mcp tools [server]", "查看 MCP 工具列表"),
    ("/skills", "查看当前可用 skills"),
    ("/memory", "查看长期记忆 (clear / forget <id>)"),
    ("/help", "显示帮助"),
    ("exit", "退出"),
)


def print_banner() -> None:
    print()
    print(c("  MiniBot", "bold", "magenta") + c("  · 本地 AI 助手", "gray"))
    print(RULE)


def print_help() -> None:
    print(c("  命令", "bold"))
    cmd_width = max(len(cmd) for cmd, _ in _COMMANDS) + 2
    for cmd, desc in _COMMANDS:
        print(f"    {c(cmd.ljust(cmd_width), 'cyan')}{c(desc, 'gray')}")


def print_status(
    session: Session,
    memory_count: int,
    resumed: bool,
    skill_count: int = 0,
    mcp_summary: MCPHostSummary | None = None,
) -> None:
    status_tag = c("已恢复", "green") if resumed else c("新建", "yellow")
    memory_label = (
        c(f"{memory_count} 条", "green") if memory_count > 0 else c("空", "gray")
    )
    skill_label = (
        c(f"{skill_count} 条", "green") if skill_count > 0 else c("空", "gray")
    )
    rows = [
        ("会话", f"{session.session_id}  {status_tag}"),
        ("标题", session.title),
        ("进度", f"{session.turn_count()} 轮 · {len(session.messages)} 条消息"),
        ("记忆", memory_label),
        ("技能", skill_label),
        ("MCP", _format_mcp_brief(mcp_summary)),
    ]
    for label, value in rows:
        print(f"  {c(label.ljust(4), 'gray')}  {value}")
    print()


def print_session(session: Session) -> None:
    print(
        f"  {c('当前会话', 'gray')}  "
        f"{session.session_id}  "
        f"{c('·', 'gray')} {session.title}  "
        f"{c('·', 'gray')} {session.turn_count()} 轮 / {len(session.messages)} 条"
    )


def print_sessions(sessions: Iterable[Session]) -> None:
    sessions = list(sessions)
    if not sessions:
        info("还没有历史会话。")
        return
    print()
    print(c("  会话列表", "bold"))
    for s in sessions:
        print(
            f"    {c(s.session_id, 'cyan')}  "
            f"{s.title}  "
            f"{c('·', 'gray')} {s.message_count} 条  "
            f"{c('·', 'gray')} {c(str(s.updated_at or '-'), 'dim')}"
        )
    print()


def print_memory_list(items: list[MemoryItem]) -> None:
    if not items:
        info("长期记忆为空。")
        return
    print()
    print(c(f"  长期记忆 · {len(items)} 条", "bold"))
    for m in items:
        print(
            f"    {c(m.id, 'cyan')}  {m.content}  {c(m.created_at, 'dim')}"
        )
    print()


def print_skills(skills: Iterable[tuple[str, str, tuple[str, ...]]]) -> None:
    items = list(skills)
    if not items:
        info("当前没有可用 skills。")
        return
    print()
    print(c(f"  Skills · {len(items)} 条", "bold"))
    for name, description, tools in items:
        tool_text = ", ".join(tools)
        print(
            f"    {c(name, 'cyan')}  {description}"
            f"  {c('·', 'gray')} tools: {c(tool_text, 'dim')}"
        )
    print()


def print_mcp_status(
    summary: MCPHostSummary,
    statuses: Iterable[MCPServerStatus],
) -> None:
    statuses = list(statuses)
    if summary.configured_servers == 0:
        info("当前未配置 MCP server。")
        return

    print()
    print(c("  MCP", "bold"))
    print(f"    {c('配置', 'gray')}  {summary.config_path or '未找到 mcp.json'}")
    overview = (
        f"{summary.connected_servers}/{summary.enabled_servers} 已连接"
        f"  {c('·', 'gray')} {summary.tool_count} tools"
    )
    if summary.failed_servers > 0:
        overview += f"  {c('·', 'gray')} {summary.failed_servers} failed"
    print(f"    {c('摘要', 'gray')}  {overview}")
    for status in statuses:
        state_text, state_style = _format_mcp_state(status)
        tool_suffix = c(f"{status.tool_count} tools", "dim")
        trust_suffix = c("trusted", "green") if status.trusted else c("approval", "yellow")
        print(
            f"    {c(status.name, 'cyan')}  "
            f"{c(status.transport, 'blue')}  "
            f"{c(state_text, state_style)}  "
            f"{tool_suffix}  "
            f"{trust_suffix}"
        )
        if status.last_error:
            print(f"      {c('error', 'yellow')}  {status.last_error}")
    print()


def print_mcp_tools(
    statuses: Iterable[MCPServerStatus],
    *,
    server_name: str | None = None,
) -> None:
    statuses = list(statuses)
    if server_name is not None:
        statuses = [status for status in statuses if status.name == server_name]
        if not statuses:
            warn(f"未找到 MCP server: {server_name}")
            return

    if not statuses:
        info("当前没有可展示的 MCP tool。")
        return

    print()
    print(c("  MCP Tools", "bold"))
    for status in statuses:
        state_text, _ = _format_mcp_state(status)
        print(
            f"    {c(status.name, 'cyan')}  "
            f"{c('·', 'gray')} {c(status.transport, 'blue')}  "
            f"{c('·', 'gray')} {state_text}"
        )
        if not status.tool_names:
            print(f"      {c('(无已发现工具)', 'dim')}")
            continue
        for tool_name in status.tool_names:
            print(f"      {c(tool_name, 'dim')}")
    print()


def _format_mcp_brief(summary: MCPHostSummary | None) -> str:
    if summary is None or summary.configured_servers == 0:
        return c("未配置", "gray")
    if summary.enabled_servers == 0:
        return c("全部禁用", "gray")

    parts = [
        c(f"{summary.connected_servers}/{summary.enabled_servers} 已连接", "green"),
        c(f"{summary.tool_count} tools", "dim"),
    ]
    if summary.failed_servers > 0:
        parts.append(c(f"{summary.failed_servers} failed", "yellow"))
    return f" {c('·', 'gray')} ".join(parts)


def _format_mcp_state(status: MCPServerStatus) -> tuple[str, str]:
    if not status.enabled:
        return "disabled", "gray"
    if status.connected:
        return "connected", "green"
    if status.last_error:
        return "failed", "yellow"
    return "pending", "gray"
