"""High-level macOS builtin app tools."""

from __future__ import annotations

from typing import Any

from ..macos import (
    AppleScriptBridge,
    AppleScriptBridgeError,
    CalendarEventRecord,
    NoteRecord,
    ReminderRecord,
)
from .base import Tool, ToolExecutionContext
from .result import ToolOutput


def _serialize_calendar_event(event: CalendarEventRecord) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "title": event.title,
        "calendar_name": event.calendar_name,
        "start_at": event.start_at,
        "end_at": event.end_at,
        "location": event.location,
        "notes": event.notes,
    }


def _serialize_reminder(reminder: ReminderRecord) -> dict[str, Any]:
    return {
        "reminder_id": reminder.reminder_id,
        "title": reminder.title,
        "list_name": reminder.list_name,
        "completed": reminder.completed,
        "due_at": reminder.due_at,
        "notes": reminder.notes,
    }


def _serialize_note(note: NoteRecord) -> dict[str, Any]:
    return {
        "note_id": note.note_id,
        "title": note.title,
        "folder_name": note.folder_name,
        "preview": note.preview,
    }


class _MacOSAppTool(Tool):
    def __init__(self, bridge: AppleScriptBridge) -> None:
        super().__init__()
        self.bridge = bridge

    @property
    def exclusive(self) -> bool:
        return True

    def _bridge_failure(self, exc: AppleScriptBridgeError) -> ToolOutput:
        return ToolOutput.failure(exc.code, exc.message, data=exc.data)


class CalendarListEventsTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "calendar_list_events"

    @property
    def description(self) -> str:
        return "列出指定时间窗口内的日历事件"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start_at": {
                    "type": "string",
                    "description": "起始本地时间，格式如 2026-04-18T09:00",
                },
                "end_at": {
                    "type": "string",
                    "description": "结束本地时间，格式如 2026-04-18T18:00",
                },
                "calendar_name": {
                    "type": "string",
                    "description": "可选，限定某个日历名称；省略时查询所有日历",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少条，1-25，默认 10",
                    "minimum": 1,
                    "maximum": 25,
                },
            },
            "required": ["start_at", "end_at"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        start_at: str,
        end_at: str,
        calendar_name: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            events = self.bridge.list_calendar_events(
                start_at=start_at,
                end_at=end_at,
                calendar_name=calendar_name,
                limit=limit,
            )
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已找到 {len(events)} 条日历事件。",
            data={
                "events": [_serialize_calendar_event(event) for event in events],
                "count": len(events),
                "calendar_name": calendar_name,
                "start_at": start_at,
                "end_at": end_at,
            },
        )


class CalendarCreateEventTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "calendar_create_event"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "在 macOS 日历中创建事件"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "事件标题",
                },
                "start_at": {
                    "type": "string",
                    "description": "开始本地时间，格式如 2026-04-18T09:00",
                },
                "end_at": {
                    "type": "string",
                    "description": "结束本地时间，格式如 2026-04-18T10:00",
                },
                "calendar_name": {
                    "type": "string",
                    "description": "可选，写入指定日历名称；省略时自动选择第一个可写日历",
                },
                "location": {
                    "type": "string",
                    "description": "可选，地点",
                },
                "notes": {
                    "type": "string",
                    "description": "可选，备注",
                },
            },
            "required": ["title", "start_at", "end_at"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        title: str,
        start_at: str,
        end_at: str,
        calendar_name: str | None = None,
        location: str | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            event = self.bridge.create_calendar_event(
                title=title,
                start_at=start_at,
                end_at=end_at,
                calendar_name=calendar_name,
                location=location,
                notes=notes,
            )
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已创建日历事件：{event.title}。",
            data=_serialize_calendar_event(event),
        )


class RemindersListTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "reminders_list"

    @property
    def description(self) -> str:
        return "列出提醒事项"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "可选，限定某个提醒列表名称；省略时查询所有列表",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "completed", "all"],
                    "description": "筛选 open、completed 或 all，默认 open",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 25,
                    "description": "最多返回多少条，1-25，默认 10",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        list_name: str | None = None,
        status: str = "open",
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            reminders = self.bridge.list_reminders(
                list_name=list_name,
                status=status,
                limit=limit,
            )
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已找到 {len(reminders)} 条提醒事项。",
            data={
                "reminders": [_serialize_reminder(item) for item in reminders],
                "count": len(reminders),
                "list_name": list_name,
                "status": status or "open",
            },
        )


class RemindersCreateTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "reminders_create"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "在 macOS 提醒事项中创建提醒"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "提醒标题",
                },
                "due_at": {
                    "type": "string",
                    "description": "可选，本地截止时间，格式如 2026-04-18T20:00",
                },
                "list_name": {
                    "type": "string",
                    "description": "可选，目标提醒列表；省略时自动选择默认列表",
                },
                "notes": {
                    "type": "string",
                    "description": "可选，提醒备注",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        title: str,
        due_at: str | None = None,
        list_name: str | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            reminder = self.bridge.create_reminder(
                title=title,
                due_at=due_at,
                list_name=list_name,
                notes=notes,
            )
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已创建提醒事项：{reminder.title}。",
            data=_serialize_reminder(reminder),
        )


class RemindersCompleteTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "reminders_complete"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "将提醒事项标记为已完成"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reminder_id": {
                    "type": "string",
                    "description": "提醒事项 ID",
                }
            },
            "required": ["reminder_id"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        reminder_id: str,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            reminder = self.bridge.complete_reminder(reminder_id=reminder_id)
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已完成提醒事项：{reminder.title}。",
            data=_serialize_reminder(reminder),
        )


class NotesSearchTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "notes_search"

    @property
    def description(self) -> str:
        return "搜索 macOS 备忘录/Notes"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "folder_name": {
                    "type": "string",
                    "description": "可选，限定某个文件夹名称；省略时搜索全部文件夹",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 25,
                    "description": "最多返回多少条，1-25，默认 10",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        query: str,
        folder_name: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            notes = self.bridge.search_notes(
                query=query,
                folder_name=folder_name,
                limit=limit,
            )
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已找到 {len(notes)} 条备忘录。",
            data={
                "notes": [_serialize_note(item) for item in notes],
                "count": len(notes),
                "query": query,
                "folder_name": folder_name,
            },
        )


class NotesCreateTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "notes_create"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "在 macOS 备忘录/Notes 中创建新笔记"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "笔记标题",
                },
                "content": {
                    "type": "string",
                    "description": "笔记正文，纯文本",
                },
                "folder_name": {
                    "type": "string",
                    "description": "可选，目标文件夹名称；省略时自动选择默认文件夹",
                },
            },
            "required": ["title", "content"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        title: str,
        content: str,
        folder_name: str | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            note = self.bridge.create_note(
                title=title,
                content=content,
                folder_name=folder_name,
            )
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已创建备忘录：{note.title}。",
            data=_serialize_note(note),
        )


class NotesAppendTool(_MacOSAppTool):
    @property
    def name(self) -> str:
        return "notes_append"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "向现有 macOS 备忘录/Notes 追加内容"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "笔记 ID",
                },
                "content": {
                    "type": "string",
                    "description": "要追加的纯文本内容",
                },
            },
            "required": ["note_id", "content"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        note_id: str,
        content: str,
        **kwargs: Any,
    ) -> ToolOutput:
        del context, kwargs
        try:
            note = self.bridge.append_note(note_id=note_id, content=content)
        except AppleScriptBridgeError as exc:
            return self._bridge_failure(exc)
        return ToolOutput.success(
            f"已追加到备忘录：{note.title}。",
            data=_serialize_note(note),
        )
