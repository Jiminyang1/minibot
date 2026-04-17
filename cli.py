"""Interactive REPL and slash-command dispatch for MiniBot.

All styling and print helpers live in :mod:`minibot.ui`. This module
keeps only the command wiring: read a line, route it to a handler,
hand user messages to the turn engine.
"""

from __future__ import annotations

try:
    import readline  # noqa: F401
except ImportError:
    readline = None

from typing import TYPE_CHECKING

from . import ui
from .session import Session, SessionManager

if TYPE_CHECKING:
    from .turn_engine import TurnEngine
    from .user_memory import UserMemoryStore


def _is_terminal_escape_sequence(text: str) -> bool:
    return text.startswith("\x1b") or text.startswith("^[[")


def _startup_session(manager: SessionManager) -> tuple[Session, bool]:
    """Resume current/latest, or create a new one. Returns (session, resumed)."""
    current = manager.load_current_session()
    if current is not None:
        return current, True
    latest = manager.latest_session(prefer_non_empty=True)
    if latest is not None:
        manager.set_current_session(latest.session_id)
        return latest, True
    session = manager.create_session()
    manager.set_current_session(session.session_id)
    return session, False


# ── slash-command handlers ───────────────────────────────────────


def _handle_new(manager: SessionManager) -> Session:
    session = manager.create_session()
    manager.set_current_session(session.session_id)
    ui.success(f"已创建新会话 {session.session_id}")
    ui.print_session(session)
    return session


def _handle_delete(raw: str, manager: SessionManager, current: Session) -> Session:
    target = raw[len("/delete"):].strip()
    if not target:
        ui.info("用法: /delete <session_id|current>")
        return current

    resolved = current.session_id if target == "current" else target
    if not manager.delete_session(resolved):
        ui.warn(f"未找到会话: {resolved}")
        return current

    ui.success(f"已删除会话 {resolved}")
    if resolved != current.session_id:
        return current

    replacement = manager.create_session()
    manager.set_current_session(replacement.session_id)
    ui.success(f"已创建新会话 {replacement.session_id}")
    ui.print_session(replacement)
    return replacement


def _handle_resume(raw: str, manager: SessionManager) -> Session | None:
    target = raw[len("/resume"):].strip()
    if not target:
        ui.info("用法: /resume <session_id>")
        return None
    loaded = manager.load(target)
    if loaded is None:
        ui.warn(f"未找到会话: {target}")
        return None
    manager.set_current_session(loaded.session_id)
    ui.success(f"已恢复会话 {loaded.session_id}")
    ui.print_session(loaded)
    return loaded


def _handle_compact(session: Session, turn_engine: TurnEngine) -> None:
    try:
        did_compact, message = turn_engine.compact_session(session)
    except Exception as exc:
        ui.warn(f"手动 compact 失败: {exc}")
        return
    if did_compact:
        ui.success(f"已压缩当前会话 · {message}")
    else:
        ui.info(message)


def _handle_memory(raw: str, memory_store: UserMemoryStore) -> None:
    parts = raw.strip().split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else ""

    if sub == "":
        ui.print_memory_list(memory_store.list())
        return
    if sub == "clear":
        count = memory_store.clear()
        ui.warn(f"已清空长期记忆 · 共删除 {count} 条")
        return
    if sub == "forget":
        if len(parts) < 3 or not parts[2].strip():
            ui.info("用法: /memory forget <memory_id>")
            return
        target = parts[2].strip()
        if memory_store.delete(target):
            ui.success(f"已删除记忆 {target}")
        else:
            ui.warn(f"未找到记忆 {target}")
        return

    ui.warn(
        f"未知子命令: /memory {sub}。用法: /memory | /memory clear | /memory forget <id>"
    )


# ── REPL loop ────────────────────────────────────────────────────


def run_repl(
    turn_engine: TurnEngine,
    manager: SessionManager,
    memory_store: UserMemoryStore,
) -> None:
    current, resumed = _startup_session(manager)

    ui.print_banner()
    ui.print_status(current, len(memory_store.list()), resumed)
    ui.print_help()
    print(ui.RULE)

    while True:
        try:
            user_msg = ui.read_user_input()
        except (EOFError, KeyboardInterrupt):
            ui.info("\n已退出。")
            break

        if user_msg.lower() in {"exit", "quit"}:
            ui.info("已退出。")
            break
        if not user_msg or _is_terminal_escape_sequence(user_msg):
            continue

        if user_msg in {"/help", "help"}:
            ui.print_help()
            continue
        if user_msg in {"/sessions", "/list"}:
            ui.print_sessions(manager.list_sessions())
            continue
        if user_msg == "/new":
            current = _handle_new(manager)
            continue
        if user_msg.startswith("/delete"):
            current = _handle_delete(user_msg, manager, current)
            continue
        if user_msg == "/compact":
            _handle_compact(current, turn_engine)
            continue
        if user_msg.startswith("/memory"):
            _handle_memory(user_msg, memory_store)
            continue
        if user_msg.startswith("/resume"):
            resumed_session = _handle_resume(user_msg, manager)
            if resumed_session is not None:
                current = resumed_session
            continue
        if user_msg.startswith("/"):
            ui.warn(f"未知命令: {user_msg}，用 /help 查看帮助。")
            continue

        try:
            result = turn_engine.handle_turn(current, user_msg)
        except Exception as exc:
            ui.warn(f"运行失败: {exc}")
            continue

        if result.did_compact and result.compact_message:
            ui.info(f"已自动压缩 · {result.compact_message}")

        ui.print_agent_reply(result.reply)
