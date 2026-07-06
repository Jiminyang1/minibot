"""Tools that let the agent schedule its own future runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Tool, ToolExecutionContext
from .result import ToolOutput

if TYPE_CHECKING:
    from ..schedule_store import ScheduleStore


class ScheduleTaskTool(Tool):
    """Create a recurring (cron) or one-shot scheduled task."""

    def __init__(self, store: "ScheduleStore", *, workspace: str | None = None) -> None:
        super().__init__()
        self._store = store
        self._workspace_label = workspace

    @property
    def name(self) -> str:
        return "schedule_task"

    @property
    def description(self) -> str:
        return (
            "创建定时任务:到点后由 scheduler daemon 以无人值守方式执行 prompt,"
            "结果通过系统通知投递并存为新会话。周期任务用 cron(本地时间,5 字段,"
            "如每天 8 点 = '0 8 * * *');一次性提醒用 at(ISO 时间,如 "
            "'2026-07-07T09:00')。cron 与 at 恰好提供一个。"
            "prompt 要写成自包含指令——运行时没有当前对话的上下文。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "任务短标题"},
                "prompt": {
                    "type": "string",
                    "description": "到点执行的自包含指令(无人值守,不能依赖当前对话上下文)",
                },
                "cron": {
                    "type": "string",
                    "description": "5 字段 cron 表达式(本地时间),周期任务用",
                },
                "at": {
                    "type": "string",
                    "description": "一次性触发时间,ISO 格式(无时区视为本地时间)",
                },
            },
            "required": ["title", "prompt"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        # Creating future autonomous runs is a sensitive act.
        return True

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        title: str,
        prompt: str,
        cron: str | None = None,
        at: str | None = None,
    ) -> ToolOutput:
        del context
        if (cron is None) == (at is None):
            return ToolOutput.failure(
                "invalid_args",
                "cron 与 at 必须恰好提供一个。",
                data={"tool": self.name},
            )
        if not prompt.strip():
            return ToolOutput.failure(
                "invalid_args", "prompt 不能为空。", data={"tool": self.name}
            )
        try:
            task = self._store.add(
                title=title,
                prompt=prompt,
                kind="cron" if cron is not None else "once",
                expr=cron if cron is not None else str(at),
                workspace=self._workspace_label,
            )
        except ValueError as exc:
            return ToolOutput.failure(
                "invalid_args", f"时间表达式无效: {exc}", data={"tool": self.name}
            )
        next_run = task.next_run()
        return ToolOutput.success(
            f"已创建定时任务 {task.id}「{task.title}」,"
            f"下次触发: {next_run:%Y-%m-%d %H:%M}" if next_run else
            f"已创建定时任务 {task.id}「{task.title}」",
            data={
                "task_id": task.id,
                "kind": task.kind,
                "expr": task.expr,
                "next_run": None if next_run is None else next_run.isoformat(timespec="minutes"),
            },
        )


class ListScheduledTasksTool(Tool):
    """List scheduled tasks with their next fire time and last status."""

    def __init__(self, store: "ScheduleStore") -> None:
        super().__init__()
        self._store = store

    @property
    def name(self) -> str:
        return "list_scheduled_tasks"

    @property
    def description(self) -> str:
        return "查看全部定时任务:id、标题、时间表达式、下次触发、上次运行状态。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    @property
    def read_only(self) -> bool:
        return True

    def execute(self, *, context: ToolExecutionContext, **kwargs: Any) -> ToolOutput:
        del context, kwargs
        tasks = self._store.list()
        if not tasks:
            return ToolOutput.success("当前没有定时任务。", data={"tasks": []})
        rows = []
        for task in tasks:
            next_run = task.next_run()
            rows.append(
                {
                    "task_id": task.id,
                    "title": task.title,
                    "kind": task.kind,
                    "expr": task.expr,
                    "enabled": task.enabled,
                    "next_run": None
                    if next_run is None
                    else next_run.isoformat(timespec="minutes"),
                    "last_run_at": task.last_run_at,
                    "last_status": task.last_status,
                }
            )
        return ToolOutput.success(
            f"共 {len(tasks)} 个定时任务。", data={"tasks": rows}
        )


class CancelScheduledTaskTool(Tool):
    """Remove a scheduled task by id."""

    def __init__(self, store: "ScheduleStore") -> None:
        super().__init__()
        self._store = store

    @property
    def name(self) -> str:
        return "cancel_scheduled_task"

    @property
    def description(self) -> str:
        return "取消(删除)一个定时任务。task_id 可用 list_scheduled_tasks 查询。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "要取消的任务 id"}
            },
            "required": ["task_id"],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, task_id: str) -> ToolOutput:
        del context
        if self._store.remove(task_id):
            return ToolOutput.success(
                f"已取消定时任务 {task_id}。", data={"task_id": task_id}
            )
        return ToolOutput.failure(
            "not_found", f"未找到定时任务 {task_id}。", data={"task_id": task_id}
        )
