"""Interactive REPL and slash-command dispatch for MiniBot.

All styling and print helpers live in :mod:`minibot.ui`. This module
keeps only the command wiring: read a line, route it to a handler,
hand user messages to the turn engine.
"""

from __future__ import annotations

try:
    import readline

    readline.parse_and_bind("set enable-bracketed-paste on")
except ImportError:
    pass
except Exception:
    pass

from typing import TYPE_CHECKING

from . import ui
from .mcp_host import MCPHost
from .session import Session, SessionManager

if TYPE_CHECKING:
    from .config import Config
    from .runtime.turn_engine import TurnEngine
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


def _handle_delete(
    raw: str,
    manager: SessionManager,
    turn_engine: TurnEngine,
    current: Session,
) -> Session:
    target = raw[len("/delete"):].strip()
    if not target:
        ui.info("用法: /delete <session_id|current>")
        return current

    resolved = current.session_id if target == "current" else target
    if not turn_engine.delete_session(resolved):
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


def _handle_skills(turn_engine: TurnEngine) -> None:
    ui.print_skills(turn_engine.list_available_skills())


def _handle_mcp(raw: str, mcp_host: MCPHost | None) -> None:
    if mcp_host is None:
        ui.info("当前未初始化 MCP host。")
        return

    parts = raw.strip().split()
    if len(parts) == 1:
        ui.print_mcp_status(mcp_host.summary(), mcp_host.status_snapshot())
        return
    if parts[1] == "tools":
        server_name = parts[2] if len(parts) >= 3 else None
        ui.print_mcp_tools(mcp_host.status_snapshot(), server_name=server_name)
        return
    ui.info("用法: /mcp | /mcp tools [server]")


def _handle_config(config: Config | None) -> None:
    if config is None:
        ui.info("当前未传入运行配置。")
        return
    ui.print_config(config)


def _handle_permission(raw: str, turn_engine: TurnEngine) -> None:
    parts = raw.strip().split(maxsplit=1)
    if len(parts) == 1:
        ui.print_permission_mode(turn_engine.runner.approval_mode)
        return

    mode = parts[1].strip().lower()
    aliases = {
        "ask": "ask",
        "permission": "ask",
        "prompt": "ask",
        "manual": "ask",
        "always": "always",
        "auto": "always",
        "approve": "always",
    }
    resolved = aliases.get(mode)
    if resolved is None:
        ui.info("用法: /permission [ask|always]")
        return
    turn_engine.runner.set_approval_mode(resolved)
    ui.success(f"审批模式已切换为 {resolved}")
    ui.print_permission_mode(turn_engine.runner.approval_mode)


# ── REPL loop ────────────────────────────────────────────────────


def run_repl(
    turn_engine: TurnEngine,
    manager: SessionManager,
    memory_store: UserMemoryStore,
    mcp_host: MCPHost | None = None,
    config: Config | None = None,
) -> None:
    current, resumed = _startup_session(manager)
    skill_count = len(turn_engine.list_available_skills())

    ui.print_banner()
    ui.print_status(
        current,
        len(memory_store.list()),
        resumed,
        skill_count,
        mcp_summary=None if mcp_host is None else mcp_host.summary(),
        approval_mode=turn_engine.runner.approval_mode,
    )
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
            current = _handle_delete(user_msg, manager, turn_engine, current)
            continue
        if user_msg == "/compact":
            _handle_compact(current, turn_engine)
            continue
        if user_msg == "/mcp" or user_msg.startswith("/mcp "):
            _handle_mcp(user_msg, mcp_host)
            continue
        if user_msg == "/skills":
            _handle_skills(turn_engine)
            continue
        if user_msg == "/permission" or user_msg.startswith("/permission "):
            _handle_permission(user_msg, turn_engine)
            continue
        if user_msg in {"/config", "/settings"}:
            if config is None:
                _handle_config(config)
            else:
                ui.print_config(config, approval_mode=turn_engine.runner.approval_mode)
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
        except KeyboardInterrupt:
            ui.info("\n已中断当前请求。")
            continue
        except Exception as exc:
            ui.warn(f"运行失败: {exc}")
            continue

        if result.did_compact and result.compact_message:
            ui.info(f"已自动压缩 · {result.compact_message}")

        ui.print_agent_reply(result.reply)
