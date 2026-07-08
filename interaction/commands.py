"""Slash-command dispatch independent of any concrete UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..config import ApprovalMode
from ..session import Session, SessionManager

if TYPE_CHECKING:
    from ..config import Config
    from ..mcp_host.host import MCPHost
    from ..runtime.approval import ApprovalPolicy
    from ..runtime.compactor import Compactor
    from ..runtime.context_builder import ContextBuilder
    from ..schedule_store import ScheduleStore
    from ..user_memory import UserMemoryStore


NoticeKind = Literal["info", "success", "warning", "error"]


@dataclass(frozen=True)
class CommandNotice:
    kind: NoticeKind
    title: str
    body: str = ""


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    notices: tuple[CommandNotice, ...] = ()
    current_session_id: str | None = None
    should_exit: bool = False


@dataclass(frozen=True)
class CommandContext:
    sessions: SessionManager
    compactor: "Compactor"
    context_builder: "ContextBuilder"
    memory_store: "UserMemoryStore"
    approval_policy: "ApprovalPolicy"
    mcp_host: "MCPHost | None" = None
    config: "Config | None" = None
    schedule_store: "ScheduleStore | None" = None


_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/sessions", "查看所有会话"),
    ("/new", "新建会话"),
    ("/resume <id>", "恢复指定会话"),
    ("/delete <id|current>", "删除会话"),
    ("/compact", "压缩当前会话"),
    ("/mcp", "查看 MCP server 状态"),
    ("/mcp tools [server]", "查看 MCP 工具列表"),
    ("/skills", "查看当前可用 skills"),
    ("/tasks [cancel <id>]", "查看或取消定时任务"),
    ("/permission [ask|always]", "查看或切换审批模式"),
    ("/config", "查看当前运行配置"),
    ("/memory", "查看长期记忆 (clear / forget <id>)"),
    ("/help", "显示帮助"),
    ("exit", "退出"),
)


def command_catalog(*, include_exit: bool = True) -> tuple[tuple[str, str], ...]:
    """Return the user-facing command catalog for help and completion UIs."""
    if include_exit:
        return _COMMANDS
    return tuple((command, description) for command, description in _COMMANDS if command.startswith("/"))


def dispatch_command(
    raw: str,
    current_session_id: str,
    context: CommandContext,
) -> CommandResult:
    """Dispatch a slash command or exit command.

    Returns ``handled=False`` when *raw* is a normal user prompt.
    """
    text = raw.strip()
    if not text:
        return CommandResult(handled=True)

    lowered = text.lower()
    if lowered in {"exit", "quit"}:
        return CommandResult(
            handled=True,
            should_exit=True,
            notices=(_notice("info", "已退出。"),),
            current_session_id=current_session_id,
        )

    if text in {"/help", "help"}:
        return CommandResult(
            handled=True,
            notices=(_notice("info", "命令", _format_help()),),
            current_session_id=current_session_id,
        )
    if text in {"/sessions", "/list"}:
        return _sessions(context, current_session_id)
    if text == "/new":
        return _new(context)
    if text.startswith("/delete"):
        return _delete(text, context, current_session_id)
    if text == "/compact":
        return _compact(context, current_session_id)
    if text == "/mcp" or text.startswith("/mcp "):
        return _mcp(text, context, current_session_id)
    if text == "/skills":
        return _skills(context, current_session_id)
    if text == "/tasks" or text.startswith("/tasks "):
        return _tasks(text, context, current_session_id)
    if text == "/permission" or text.startswith("/permission "):
        return _permission(text, context, current_session_id)
    if text in {"/config", "/settings"}:
        return _config(context, current_session_id)
    if text.startswith("/memory"):
        return _memory(text, context, current_session_id)
    if text.startswith("/resume"):
        return _resume(text, context, current_session_id)
    if text.startswith("/"):
        return CommandResult(
            handled=True,
            notices=(_notice("warning", f"未知命令: {text}", "用 /help 查看帮助。"),),
            current_session_id=current_session_id,
        )
    return CommandResult(handled=False, current_session_id=current_session_id)


def _new(context: CommandContext) -> CommandResult:
    session = context.sessions.create_current_session()
    return CommandResult(
        handled=True,
        current_session_id=session.session_id,
        notices=(
            _notice("success", f"已创建新会话 {session.session_id}"),
            _session_notice(session),
        ),
    )


def _delete(
    raw: str,
    context: CommandContext,
    current_session_id: str,
) -> CommandResult:
    target = raw[len("/delete") :].strip()
    if not target:
        return _usage("用法: /delete <session_id|current>", current_session_id)

    resolved = current_session_id if target == "current" else target
    if not context.sessions.delete_session(resolved):
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("warning", f"未找到会话: {resolved}"),),
        )

    notices = [_notice("success", f"已删除会话 {resolved}")]
    if resolved != current_session_id:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=tuple(notices),
        )

    replacement = context.sessions.create_current_session()
    notices.extend(
        [
            _notice("success", f"已创建新会话 {replacement.session_id}"),
            _session_notice(replacement),
        ]
    )
    return CommandResult(
        handled=True,
        current_session_id=replacement.session_id,
        notices=tuple(notices),
    )


def _resume(
    raw: str,
    context: CommandContext,
    current_session_id: str,
) -> CommandResult:
    target = raw[len("/resume") :].strip()
    if not target:
        return _usage("用法: /resume <session_id>", current_session_id)
    loaded = context.sessions.resume_session(target)
    if loaded is None:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("warning", f"未找到会话: {target}"),),
        )
    return CommandResult(
        handled=True,
        current_session_id=loaded.session_id,
        notices=(
            _notice("success", f"已恢复会话 {loaded.session_id}"),
            _session_notice(loaded),
        ),
    )


def _compact(context: CommandContext, current_session_id: str) -> CommandResult:
    session = context.sessions.load(current_session_id)
    if session is None:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("warning", f"未找到会话: {current_session_id}"),),
        )
    try:
        did_compact, message = context.compactor.compact_now(session)
    except Exception as exc:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("warning", f"手动 compact 失败: {exc}"),),
        )
    kind: NoticeKind = "success" if did_compact else "info"
    title = "已压缩当前会话" if did_compact else "无需压缩"
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(_notice(kind, title, message),),
    )


def _memory(
    raw: str,
    context: CommandContext,
    current_session_id: str,
) -> CommandResult:
    parts = raw.strip().split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else ""

    if sub == "":
        items = context.memory_store.list()
        if not items:
            return CommandResult(
                handled=True,
                current_session_id=current_session_id,
                notices=(_notice("info", "长期记忆为空。"),),
            )
        lines = [f"{item.id}  {item.content}  {item.created_at}" for item in items]
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", f"长期记忆 · {len(items)} 条", "\n".join(lines)),),
        )
    if sub == "clear":
        count = context.memory_store.clear()
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("warning", f"已清空长期记忆 · 共删除 {count} 条"),),
        )
    if sub == "forget":
        if len(parts) < 3 or not parts[2].strip():
            return _usage("用法: /memory forget <memory_id>", current_session_id)
        target = parts[2].strip()
        if context.memory_store.delete(target):
            notice = _notice("success", f"已删除记忆 {target}")
        else:
            notice = _notice("warning", f"未找到记忆 {target}")
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(notice,),
        )

    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(
            _notice(
                "warning",
                f"未知子命令: /memory {sub}",
                "用法: /memory | /memory clear | /memory forget <id>",
            ),
        ),
    )


def _tasks(
    raw: str,
    context: CommandContext,
    current_session_id: str,
) -> CommandResult:
    if context.schedule_store is None:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", "当前未初始化定时任务存储。"),),
        )

    parts = raw.strip().split()
    if len(parts) >= 2:
        if parts[1] != "cancel" or len(parts) < 3:
            return _usage("用法: /tasks | /tasks cancel <task_id>", current_session_id)
        task_id = parts[2]
        if context.schedule_store.remove(task_id):
            return CommandResult(
                handled=True,
                current_session_id=current_session_id,
                notices=(_notice("success", f"已取消定时任务 {task_id}"),),
            )
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("warning", f"未找到定时任务 {task_id}"),),
        )

    tasks = context.schedule_store.list()
    if not tasks:
        body = "当前没有定时任务。用自然语言让 MiniBot 创建,例如“每天早上 8 点给我生成今日简报”。"
    else:
        lines = []
        for task in tasks:
            next_run = task.next_run()
            when = "不再触发" if next_run is None else f"{next_run:%m-%d %H:%M}"
            status = task.last_status or "-"
            lines.append(
                f"{task.id}  {task.title} · {task.kind}: {task.expr} · "
                f"下次 {when} · 上次 {status}"
            )
        body = "\n".join(lines)
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(_notice("info", f"定时任务 · {len(tasks)} 个", body),),
    )


def _skills(context: CommandContext, current_session_id: str) -> CommandResult:
    skills = context.context_builder.list_available_skills()
    if not skills:
        body = "当前没有可用 skills。"
    else:
        body = "\n".join(
            f"{name}  {description} · tools: {', '.join(tools)}"
            for name, description, tools in skills
        )
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(_notice("info", f"Skills · {len(skills)} 条", body),),
    )


def _mcp(
    raw: str,
    context: CommandContext,
    current_session_id: str,
) -> CommandResult:
    if context.mcp_host is None:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", "当前未初始化 MCP host。"),),
        )

    parts = raw.strip().split()
    if len(parts) == 1:
        summary = context.mcp_host.summary()
        statuses = context.mcp_host.status_snapshot()
        body = [
            f"配置: {summary.config_path or '未找到 mcp.json'}",
            (
                f"摘要: {summary.connected_servers}/{summary.enabled_servers} 已连接 · "
                f"{summary.tool_count} tools · {summary.failed_servers} failed"
            ),
        ]
        for status in statuses:
            state = _mcp_state(status.enabled, status.connected, status.last_error)
            trust = "trusted" if status.trusted else "approval"
            line = (
                f"{status.name}  {status.transport}  {state}  "
                f"{status.tool_count} tools  {trust}"
            )
            if status.last_error:
                line += f"\n  error: {status.last_error}"
            body.append(line)
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", "MCP", "\n".join(body)),),
        )

    if parts[1] == "tools":
        server_name = parts[2] if len(parts) >= 3 else None
        statuses = context.mcp_host.status_snapshot()
        if server_name is not None:
            statuses = [status for status in statuses if status.name == server_name]
            if not statuses:
                return CommandResult(
                    handled=True,
                    current_session_id=current_session_id,
                    notices=(_notice("warning", f"未找到 MCP server: {server_name}"),),
                )
        lines: list[str] = []
        for status in statuses:
            lines.append(f"{status.name} · {status.transport} · {_mcp_state(status.enabled, status.connected, status.last_error)}")
            if status.tool_names:
                lines.extend(f"  {tool_name}" for tool_name in status.tool_names)
            else:
                lines.append("  (无已发现工具)")
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", "MCP Tools", "\n".join(lines) or "当前没有可展示的 MCP tool。"),),
        )

    return _usage("用法: /mcp | /mcp tools [server]", current_session_id)


def _config(context: CommandContext, current_session_id: str) -> CommandResult:
    if context.config is None:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", "当前未传入运行配置。"),),
        )
    config = context.config
    rows = [
        ("model", config.model),
        ("approval_mode", _format_approval_mode(context.approval_policy.mode)),
        ("max_iterations", str(config.max_iterations)),
        ("max_parallel_tools", str(config.max_parallel_tools)),
        ("compact_token_threshold", str(config.compact_token_threshold)),
        ("reserved_completion_tokens", str(config.reserved_completion_tokens)),
        ("compact_keep_recent_tokens", str(config.compact_keep_recent_tokens)),
    ]
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(_notice("info", "Config", "\n".join(f"{k}: {v}" for k, v in rows)),),
    )


def _permission(
    raw: str,
    context: CommandContext,
    current_session_id: str,
) -> CommandResult:
    parts = raw.strip().split(maxsplit=1)
    if len(parts) == 1:
        return CommandResult(
            handled=True,
            current_session_id=current_session_id,
            notices=(_notice("info", "Permission", f"mode: {_format_approval_mode(context.approval_policy.mode)}"),),
        )

    aliases: dict[str, ApprovalMode] = {
        "ask": "ask",
        "permission": "ask",
        "prompt": "ask",
        "manual": "ask",
        "always": "always",
        "auto": "always",
        "approve": "always",
    }
    mode = aliases.get(parts[1].strip().lower())
    if mode is None:
        return _usage("用法: /permission [ask|always]", current_session_id)
    context.approval_policy.set_mode(mode)
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(
            _notice("success", f"审批模式已切换为 {mode}"),
            _notice("info", "Permission", f"mode: {_format_approval_mode(context.approval_policy.mode)}"),
        ),
    )


def _sessions(context: CommandContext, current_session_id: str) -> CommandResult:
    sessions = context.sessions.list_sessions()
    if not sessions:
        body = "还没有历史会话。"
    else:
        body = "\n".join(
            f"{session.session_id}  {session.title} · {session.message_count} 条 · {session.updated_at or '-'}"
            for session in sessions
        )
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(_notice("info", "会话列表", body),),
    )


def _usage(message: str, current_session_id: str) -> CommandResult:
    return CommandResult(
        handled=True,
        current_session_id=current_session_id,
        notices=(_notice("info", message),),
    )


def _notice(kind: NoticeKind, title: str, body: str = "") -> CommandNotice:
    return CommandNotice(kind=kind, title=title, body=body)


def _session_notice(session: Session) -> CommandNotice:
    return _notice(
        "info",
        "当前会话",
        (
            f"{session.session_id} · {session.title} · "
            f"{session.turn_count()} 轮 / {len(session.messages)} 条"
        ),
    )


def _format_help() -> str:
    commands = command_catalog()
    width = max(len(command) for command, _ in commands) + 2
    return "\n".join(f"{command.ljust(width)}{description}" for command, description in commands)


def _format_approval_mode(mode: str) -> str:
    if mode == "always":
        return "always · 自动批准敏感工具"
    return "ask · 敏感工具需要确认"


def _mcp_state(enabled: bool, connected: bool, error: str | None) -> str:
    if not enabled:
        return "disabled"
    if connected:
        return "connected"
    if error:
        return "failed"
    return "pending"
