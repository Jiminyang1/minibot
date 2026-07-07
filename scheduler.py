"""The proactivity daemon: fire scheduled prompts as headless agent runs.

Each firing creates a fresh session (searchable later via search_history),
runs it through the same ``AgentSession.prompt`` every frontend uses, and
delivers the outcome as a macOS notification. Missed schedules fire on
catch-up within a grace window; older misses are recorded and skipped.

Headless runs deny sensitive tools by default — there is nobody to ask.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timedelta
import fcntl
import functools
import json
from pathlib import Path
import subprocess
import threading

from .run_log import preview_text
from .runtime.agent_session import AgentSession, RunCancelled
from .schedule_store import ScheduledTask, ScheduleStore, local_now
from .session import Session, SessionManager

Notifier = Callable[[str, str], None]

_GRACE = timedelta(hours=1)

HEARTBEAT_OK = "HEARTBEAT_OK"

_HEARTBEAT_TEMPLATE = """\
# Heartbeat 巡逻清单
#
# 每次心跳时 agent 会读这份清单逐项检查。井号开头的行是注释。
# 用祈使句写检查项,例如:
#
# - 检查未读邮件,有重要的立即通知我
# - 如果 30 分钟内有日程开始,提醒我
# - 看看 ~/Downloads 里有没有超过一周没整理的文件
#
# 清单为空时心跳会安静地跳过。
"""



def macos_notify(title: str, body: str) -> None:
    """Best-effort desktop notification via osascript (no-op elsewhere)."""
    script = (
        f"display notification {json.dumps(body, ensure_ascii=False)} "
        f"with title {json.dumps(title, ensure_ascii=False)}"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


class Scheduler:
    """Poll the schedule store and fire due tasks; ``tick`` is the testable unit."""

    def __init__(
        self,
        *,
        store: ScheduleStore,
        agent_session: AgentSession,
        session_manager: SessionManager,
        notifier: Notifier | None = macos_notify,
        grace: timedelta = _GRACE,
        tick_seconds: float = 30.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.agent_session = agent_session
        self.session_manager = session_manager
        self.notifier = notifier
        self.grace = grace
        self.tick_seconds = tick_seconds
        self.log = log or (lambda message: None)

    def run_forever(self, stop_event: threading.Event) -> None:
        self.log("scheduler 已启动")
        while not stop_event.is_set():
            try:
                self.tick(local_now())
            except Exception as exc:
                self.log(f"scheduler tick 失败: {exc}")
            stop_event.wait(self.tick_seconds)

    def tick(self, now: datetime) -> list[str]:
        """Fire every due task once; return the fired task ids."""
        fired: list[str] = []
        for task in self.store.list():
            due_at = task.next_run(now=now)
            if due_at is None or due_at > now:
                continue
            if now - due_at > self.grace:
                # Too stale to fire (laptop was asleep/daemon down): record
                # the miss and let the schedule advance from now.
                self._mark(task, status="missed", now=now)
                self.log(f"错过触发窗口,跳过: {task.title} (应于 {due_at:%m-%d %H:%M})")
                continue
            self._fire(task, now)
            fired.append(task.id)
        return fired

    def _fire(self, task: ScheduledTask, now: datetime) -> None:
        if task.kind == "heartbeat":
            self._fire_heartbeat(task, now)
            return
        self.log(f"触发定时任务: {task.title}")
        session = self.session_manager.create_session(
            title=f"[定时] {task.title} · {now:%m-%d %H:%M}"
        )
        prompt = (
            f"[定时任务「{task.title}」的无人值守运行。没有用户在场:不要提问,"
            f"敏感工具默认会被拒绝,直接产出最终结果。]\n{task.prompt}"
        )
        try:
            outcome = self.agent_session.prompt(session.session_id, prompt)
        except RunCancelled:
            self._mark(task, status="cancelled", now=now)
            return
        except Exception as exc:
            self._mark(task, status=f"failed: {type(exc).__name__}", now=now)
            self._notify(f"MiniBot 任务失败: {task.title}", preview_text(str(exc), 120))
            return
        self._mark(task, status="success", now=now)
        self._notify(
            f"MiniBot: {task.title}",
            preview_text(outcome.reply, 160) or "（完成，无文字输出）",
        )

    def _fire_heartbeat(self, task: ScheduledTask, now: datetime) -> None:
        """One patrol turn: reused session, inlined checklist, quiet-by-default."""
        self.log(f"心跳巡逻: {task.title}")
        session = self._heartbeat_session(task)
        checklist = self._read_checklist()
        prompt = self._heartbeat_prompt(task, checklist, now)
        try:
            outcome = self.agent_session.prompt(session.session_id, prompt)
        except RunCancelled:
            self._mark(task, status="cancelled", now=now)
            return
        except Exception as exc:
            self._mark(task, status=f"failed: {type(exc).__name__}", now=now)
            self._notify(f"MiniBot 心跳失败: {task.title}", preview_text(str(exc), 120))
            return
        if HEARTBEAT_OK in outcome.reply:
            # Nothing needed attention: no notification, quiet status.
            self._mark(task, status="ok-quiet", now=now)
            return
        self._mark(task, status="attention", now=now)
        self._notify(
            f"MiniBot 心跳: {task.title}",
            preview_text(outcome.reply, 160) or "（有情况，详见会话）",
        )

    def _heartbeat_session(self, task: ScheduledTask) -> Session:
        """Load the heartbeat's persistent session, creating it once."""
        if task.session_id is not None:
            existing = self.session_manager.load(task.session_id)
            if existing is not None:
                return existing
        session = self.session_manager.create_session(
            title=f"[心跳] {task.title}"
        )
        current = self.store.get(task.id)
        if current is not None:
            self.store.update(current, session_id=session.session_id)
        return session

    def _read_checklist(self) -> str:
        path = self.store.state_dir / "HEARTBEAT.md"
        if not path.exists():
            path.write_text(_HEARTBEAT_TEMPLATE, encoding="utf-8")
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return "\n".join(lines)

    def _heartbeat_prompt(
        self,
        task: ScheduledTask,
        checklist: str,
        now: datetime,
    ) -> str:
        parts = [
            f"[心跳巡逻 · {now:%m-%d %H:%M} · 无人值守:不要提问,敏感工具默认被拒。]",
            "逐项检查下面的巡逻清单,该做的直接做;只有真正需要用户注意的事才写进回复。",
            f"如果没有任何需要用户注意的事,回复中必须包含 {HEARTBEAT_OK}"
            "(可以附一句简短原因)——这会让通知保持安静。",
        ]
        if task.prompt.strip():
            parts.append(f"附加常设指令: {task.prompt.strip()}")
        if checklist:
            parts.append(
                "巡逻清单如下(内容已从 ~/.minibot/HEARTBEAT.md 内联,"
                "不需要也不要再去读任何清单文件):\n" + checklist
            )
        else:
            parts.append(
                f"巡逻清单为空(用户可编辑 ~/.minibot/HEARTBEAT.md),"
                f"直接回复 {HEARTBEAT_OK}。"
            )
        return "\n".join(parts)

    def _mark(self, task: ScheduledTask, *, status: str, now: datetime) -> None:
        current = self.store.get(task.id)
        if current is None:
            return
        changes: dict[str, object] = {
            "last_run_at": now.isoformat(timespec="seconds"),
            "last_status": status,
        }
        if current.kind == "once":
            changes["enabled"] = False
        self.store.update(current, **changes)

    def _notify(self, title: str, body: str) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(title, body)
        except Exception:
            return


def _deny_headless_approvals(request: object, cancel_event: object) -> bool:
    del request, cancel_event
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the MiniBot scheduler daemon (fires scheduled tasks).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="workspace for tool roots in scheduled runs (default: cwd)",
    )
    args = parser.parse_args()

    from .bootstrap import build_runtime
    from .config import Config, load_env, resolve_state_home

    load_env()
    config = Config.from_env()
    state_home = resolve_state_home()

    # Single daemon per state home: the lock outlives the loop on purpose.
    lock_handle = (state_home / "scheduler.pid.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("已有 scheduler daemon 在运行,退出。")
        return 1

    # Unbuffered logging: daemons get killed, buffered lines get lost.
    log = functools.partial(print, flush=True)
    runtime = build_runtime(
        config=config,
        workspace=Path(args.workspace).resolve() if args.workspace else None,
        approval_handler=_deny_headless_approvals,
        log_handler=log,
    )
    scheduler = Scheduler(
        store=ScheduleStore(state_home),
        agent_session=runtime.agent_session,
        session_manager=runtime.manager,
        log=log,
    )
    stop_event = threading.Event()
    try:
        scheduler.run_forever(stop_event)
    except KeyboardInterrupt:
        print("scheduler 已停止。")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
