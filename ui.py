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
from collections.abc import Iterable
from typing import Any

from .memory import MemoryItem
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
    print(f"\n  {c('Agent ›', 'bold', 'magenta')} {reply}")


# ── REPL panels ──────────────────────────────────────────────────


_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/sessions", "查看所有会话"),
    ("/new", "新建会话"),
    ("/resume <id>", "恢复指定会话"),
    ("/delete <id|current>", "删除会话"),
    ("/compact", "压缩当前会话"),
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


def print_status(session: Session, memory_count: int, resumed: bool) -> None:
    status_tag = c("已恢复", "green") if resumed else c("新建", "yellow")
    memory_label = (
        c(f"{memory_count} 条", "green") if memory_count > 0 else c("空", "gray")
    )
    rows = (
        ("会话", f"{session.session_id}  {status_tag}"),
        ("标题", session.title),
        ("进度", f"{session.turn_count()} 轮 · {len(session.messages)} 条消息"),
        ("记忆", memory_label),
    )
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
