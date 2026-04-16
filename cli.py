"""Interactive REPL and slash-command handling for MiniBot."""

from __future__ import annotations

try:
    import readline  # noqa: F401
except ImportError:
    readline = None

from typing import TYPE_CHECKING

from .session import Session, SessionManager

if TYPE_CHECKING:
    from .loop import AgentLoop


def _is_terminal_escape_sequence(text: str) -> bool:
    return text.startswith("\x1b") or text.startswith("^[[")


def _print_help() -> None:
    print(
        "命令: "
        "`/sessions` 查看会话, "
        "`/resume <session_id>` 恢复旧会话, "
        "`/new` 新建会话, "
        "`/delete <session_id|current>` 删除会话, "
        "`/compact` 压缩当前会话, "
        "`exit` 退出, "
        "`/help` 查看帮助"
    )


def _print_session(session: Session) -> None:
    print(
        f"当前会话: {session.session_id} | {session.title} | "
        f"{session.turn_count()} 轮对话 / {len(session.messages)} 条消息"
    )


def _print_sessions(manager: SessionManager) -> None:
    sessions = manager.list_sessions()
    if not sessions:
        print("还没有历史会话。")
        return
    print("\n会话列表:")
    for s in sessions:
        print(
            f"  - {s.session_id} | {s.title} | "
            f"{s.turn_count()} 轮对话 / {len(s.messages)} 条消息 | 更新于 {s.updated_at}"
        )


def _handle_compact(session: Session, loop: AgentLoop) -> None:
    try:
        did_compact, message = loop.compact_session(session)
    except Exception as exc:
        print(f"手动 compact 失败: {exc}")
        return
    if did_compact:
        print(f"已压缩当前会话: {message}")
    else:
        print(message)


def _handle_resume(raw: str, manager: SessionManager) -> Session | None:
    target = raw[len("/resume"):].strip()
    if not target:
        print("用法: /resume <session_id>")
        return None
    loaded = manager.load(target)
    if loaded is None:
        print(f"未找到会话: {target}")
        return None
    manager.set_current_session(loaded.session_id)
    print(f"已恢复会话: {loaded.session_id}")
    _print_session(loaded)
    return loaded


def _create_or_resume_startup_session(manager: SessionManager) -> tuple[Session, bool]:
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


def _handle_new(manager: SessionManager) -> Session:
    session = manager.create_session()
    manager.set_current_session(session.session_id)
    print(f"已创建新会话: {session.session_id}")
    _print_session(session)
    return session


def _handle_delete(raw: str, manager: SessionManager, current_session: Session) -> Session:
    target = raw[len("/delete"):].strip()
    if not target:
        print("用法: /delete <session_id|current>")
        return current_session

    resolved_target = current_session.session_id if target == "current" else target
    deleted = manager.delete_session(resolved_target)
    if not deleted:
        print(f"未找到会话: {resolved_target}")
        return current_session

    print(f"已删除会话: {resolved_target}")
    if resolved_target != current_session.session_id:
        return current_session

    replacement = manager.create_session()
    manager.set_current_session(replacement.session_id)
    print(f"已创建新会话: {replacement.session_id}")
    _print_session(replacement)
    return replacement


def run_repl(
    loop: AgentLoop,
    manager: SessionManager,
) -> None:
    """Run the interactive REPL loop."""
    current_session, resumed = _create_or_resume_startup_session(manager)

    print("MiniBot 已启动！")
    _print_help()
    if resumed:
        print(f"已自动恢复最近会话: {current_session.session_id}")
    else:
        print(f"已创建新会话: {current_session.session_id}")
    _print_session(current_session)

    while True:
        try:
            user_msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if user_msg.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not user_msg:
            continue
        if _is_terminal_escape_sequence(user_msg):
            continue
        if user_msg in {"/help", "help"}:
            _print_help()
            continue
        if user_msg in {"/sessions", "/list"}:
            _print_sessions(manager)
            continue
        if user_msg == "/new":
            current_session = _handle_new(manager)
            continue
        if user_msg.startswith("/delete"):
            current_session = _handle_delete(user_msg, manager, current_session)
            continue
        if user_msg == "/compact":
            _handle_compact(current_session, loop)
            continue
        if user_msg.startswith("/resume"):
            resumed = _handle_resume(user_msg, manager)
            if resumed is not None:
                current_session = resumed
            continue
        if user_msg.startswith("/"):
            print(f"未知命令: {user_msg}，用 /help 查看帮助。")
            continue

        try:
            result = loop.handle_turn(current_session, user_msg)
        except Exception as exc:
            print(f"\n❌ 运行失败: {exc}")
            continue

        if result.did_compact and result.compact_message:
            print(f"已自动压缩当前会话: {result.compact_message}")

        print(f"\nAgent: {result.reply}")
