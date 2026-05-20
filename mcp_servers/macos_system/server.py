"""Bundled stdio MCP server for macOS Calendar, Reminders, Notes, and Mail."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bridge import (  # type: ignore
        AppleScriptBridge,
        AppleScriptBridgeError,
        CalendarEventRecord,
        MailDraftRecord,
        MailMessageBodyRecord,
        MailMessageRecord,
        MailSendRecord,
        MailboxRecord,
        NoteRecord,
        ReminderRecord,
    )
else:
    from .bridge import (
        AppleScriptBridge,
        AppleScriptBridgeError,
        CalendarEventRecord,
        MailDraftRecord,
        MailMessageBodyRecord,
        MailMessageRecord,
        MailSendRecord,
        MailboxRecord,
        NoteRecord,
        ReminderRecord,
    )


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


def _serialize_mailbox(mailbox: MailboxRecord) -> dict[str, Any]:
    return {
        "account_name": mailbox.account_name,
        "mailbox_name": mailbox.mailbox_name,
        "unread_count": mailbox.unread_count,
        "message_count": mailbox.message_count,
    }


def _serialize_mail_message(message: MailMessageRecord) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "subject": message.subject,
        "sender": message.sender,
        "received_at": message.received_at,
        "mailbox_name": message.mailbox_name,
        "account_name": message.account_name,
        "read": message.read,
        "preview": message.preview,
    }


def _serialize_mail_message_body(message: MailMessageBodyRecord) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "subject": message.subject,
        "sender": message.sender,
        "received_at": message.received_at,
        "mailbox_name": message.mailbox_name,
        "account_name": message.account_name,
        "read": message.read,
        "body": message.body,
    }


def _serialize_mail_draft(draft: MailDraftRecord) -> dict[str, Any]:
    return {
        "subject": draft.subject,
        "to": draft.to,
        "cc": draft.cc,
        "bcc": draft.bcc,
        "sender": draft.sender,
        "visible": draft.visible,
        "preview": draft.preview,
    }


def _serialize_mail_send(result: MailSendRecord) -> dict[str, Any]:
    return {
        "subject": result.subject,
        "to": result.to,
        "cc": result.cc,
        "bcc": result.bcc,
        "sender": result.sender,
        "sent": result.sent,
        "preview": result.preview,
    }


class MacOSSystemService:
    """Thin service wrapper around ``AppleScriptBridge`` for MCP exposure."""

    def __init__(self, bridge: AppleScriptBridge) -> None:
        self.bridge = bridge

    def list_calendar_events(
        self,
        *,
        start_at: str,
        end_at: str,
        calendar_name: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        events = self.bridge.list_calendar_events(
            start_at=start_at,
            end_at=end_at,
            calendar_name=calendar_name,
            limit=limit,
        )
        return {
            "events": [_serialize_calendar_event(event) for event in events],
            "count": len(events),
            "calendar_name": calendar_name,
            "start_at": start_at,
            "end_at": end_at,
        }

    def create_calendar_event(
        self,
        *,
        title: str,
        start_at: str,
        end_at: str,
        calendar_name: str | None = None,
        location: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        event = self.bridge.create_calendar_event(
            title=title,
            start_at=start_at,
            end_at=end_at,
            calendar_name=calendar_name,
            location=location,
            notes=notes,
        )
        return _serialize_calendar_event(event)

    def list_reminders(
        self,
        *,
        list_name: str | None = None,
        status: str = "open",
        limit: int = 10,
    ) -> dict[str, Any]:
        reminders = self.bridge.list_reminders(
            list_name=list_name,
            status=status,
            limit=limit,
        )
        return {
            "reminders": [_serialize_reminder(item) for item in reminders],
            "count": len(reminders),
            "list_name": list_name,
            "status": status,
        }

    def create_reminder(
        self,
        *,
        title: str,
        due_at: str | None = None,
        list_name: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        reminder = self.bridge.create_reminder(
            title=title,
            due_at=due_at,
            list_name=list_name,
            notes=notes,
        )
        return _serialize_reminder(reminder)

    def complete_reminder(self, *, reminder_id: str) -> dict[str, Any]:
        reminder = self.bridge.complete_reminder(reminder_id=reminder_id)
        return _serialize_reminder(reminder)

    def search_notes(
        self,
        *,
        query: str,
        folder_name: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        notes = self.bridge.search_notes(
            query=query,
            folder_name=folder_name,
            limit=limit,
        )
        return {
            "notes": [_serialize_note(item) for item in notes],
            "count": len(notes),
            "query": query,
            "folder_name": folder_name,
        }

    def create_note(
        self,
        *,
        title: str,
        content: str,
        folder_name: str | None = None,
    ) -> dict[str, Any]:
        note = self.bridge.create_note(
            title=title,
            content=content,
            folder_name=folder_name,
        )
        return _serialize_note(note)

    def append_note(self, *, note_id: str, content: str) -> dict[str, Any]:
        note = self.bridge.append_note(note_id=note_id, content=content)
        return _serialize_note(note)

    def list_mailboxes(
        self,
        *,
        account_name: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        mailboxes = self.bridge.list_mailboxes(
            account_name=account_name,
            limit=limit,
        )
        return {
            "mailboxes": [_serialize_mailbox(item) for item in mailboxes],
            "count": len(mailboxes),
            "account_name": account_name,
        }

    def search_mail_messages(
        self,
        *,
        query: str,
        account_name: str | None = None,
        mailbox_name: str | None = None,
        limit: int = 10,
        include_body: bool = False,
    ) -> dict[str, Any]:
        messages = self.bridge.search_mail_messages(
            query=query,
            account_name=account_name,
            mailbox_name=mailbox_name,
            limit=limit,
            include_body=include_body,
        )
        return {
            "messages": [_serialize_mail_message(item) for item in messages],
            "count": len(messages),
            "query": query,
            "account_name": account_name,
            "mailbox_name": mailbox_name,
            "include_body": include_body,
        }

    def list_mail_messages(
        self,
        *,
        account_name: str | None = None,
        mailbox_name: str | None = None,
        limit: int = 10,
        unread_only: bool = False,
        days_back: int | None = 7,
    ) -> dict[str, Any]:
        messages = self.bridge.list_mail_messages(
            account_name=account_name,
            mailbox_name=mailbox_name,
            limit=limit,
            unread_only=unread_only,
            days_back=days_back,
        )
        return {
            "messages": [_serialize_mail_message(item) for item in messages],
            "count": len(messages),
            "account_name": account_name,
            "mailbox_name": mailbox_name,
            "unread_only": unread_only,
            "days_back": days_back,
        }

    def get_mail_message(
        self,
        *,
        message_id: str,
        account_name: str | None = None,
        mailbox_name: str | None = None,
    ) -> dict[str, Any]:
        message = self.bridge.get_mail_message(
            message_id=message_id,
            account_name=account_name,
            mailbox_name=mailbox_name,
        )
        return _serialize_mail_message_body(message)

    def create_mail_draft(
        self,
        *,
        subject: str,
        body: str,
        to: list[str] | str | None = None,
        cc: list[str] | str | None = None,
        bcc: list[str] | str | None = None,
        sender: str | None = None,
        visible: bool = True,
    ) -> dict[str, Any]:
        draft = self.bridge.create_mail_draft(
            subject=subject,
            body=body,
            to=to,
            cc=cc,
            bcc=bcc,
            sender=sender,
            visible=visible,
        )
        return _serialize_mail_draft(draft)

    def send_mail_message(
        self,
        *,
        subject: str,
        body: str,
        to: list[str] | str,
        cc: list[str] | str | None = None,
        bcc: list[str] | str | None = None,
        sender: str | None = None,
        confirm_send: bool = False,
    ) -> dict[str, Any]:
        if not confirm_send:
            raise AppleScriptBridgeError(
                "invalid_args",
                "confirm_send 必须为 true，才会发送邮件。",
                data={"field": "confirm_send"},
            )
        result = self.bridge.send_mail_message(
            subject=subject,
            body=body,
            to=to,
            cc=cc,
            bcc=bcc,
            sender=sender,
        )
        return _serialize_mail_send(result)


def build_macos_server(*, bridge: AppleScriptBridge | None = None) -> FastMCP:
    if bridge is None:
        if not AppleScriptBridge.is_supported():
            raise RuntimeError("当前环境不支持 macOS AppleScript 集成。")
        bridge = AppleScriptBridge()

    service = MacOSSystemService(bridge)
    app = FastMCP("MiniBot macOS System")

    @app.tool(name="calendar_list_events")
    def calendar_list_events(
        start_at: str,
        end_at: str,
        calendar_name: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        return service.list_calendar_events(
            start_at=start_at,
            end_at=end_at,
            calendar_name=calendar_name,
            limit=limit,
        )

    @app.tool(name="calendar_create_event")
    def calendar_create_event(
        title: str,
        start_at: str,
        end_at: str,
        calendar_name: str | None = None,
        location: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return service.create_calendar_event(
            title=title,
            start_at=start_at,
            end_at=end_at,
            calendar_name=calendar_name,
            location=location,
            notes=notes,
        )

    @app.tool(name="reminders_list")
    def reminders_list(
        list_name: str | None = None,
        status: str = "open",
        limit: int = 10,
    ) -> dict[str, Any]:
        return service.list_reminders(
            list_name=list_name,
            status=status,
            limit=limit,
        )

    @app.tool(name="reminders_create")
    def reminders_create(
        title: str,
        due_at: str | None = None,
        list_name: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return service.create_reminder(
            title=title,
            due_at=due_at,
            list_name=list_name,
            notes=notes,
        )

    @app.tool(name="reminders_complete")
    def reminders_complete(reminder_id: str) -> dict[str, Any]:
        return service.complete_reminder(reminder_id=reminder_id)

    @app.tool(name="notes_search")
    def notes_search(
        query: str,
        folder_name: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        return service.search_notes(
            query=query,
            folder_name=folder_name,
            limit=limit,
        )

    @app.tool(name="notes_create")
    def notes_create(
        title: str,
        content: str,
        folder_name: str | None = None,
    ) -> dict[str, Any]:
        return service.create_note(
            title=title,
            content=content,
            folder_name=folder_name,
        )

    @app.tool(name="notes_append")
    def notes_append(note_id: str, content: str) -> dict[str, Any]:
        return service.append_note(note_id=note_id, content=content)

    @app.tool(name="mail_list_mailboxes")
    def mail_list_mailboxes(
        account_name: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        return service.list_mailboxes(account_name=account_name, limit=limit)

    @app.tool(name="mail_search_messages")
    def mail_search_messages(
        query: str,
        account_name: str | None = None,
        mailbox_name: str | None = None,
        limit: int = 10,
        include_body: bool = False,
    ) -> dict[str, Any]:
        """Search Mail messages by subject/sender, optionally including body text."""
        return service.search_mail_messages(
            query=query,
            account_name=account_name,
            mailbox_name=mailbox_name,
            limit=limit,
            include_body=include_body,
        )

    @app.tool(name="mail_list_messages")
    def mail_list_messages(
        account_name: str | None = None,
        mailbox_name: str | None = None,
        limit: int = 10,
        unread_only: bool = False,
        days_back: int | None = 7,
    ) -> dict[str, Any]:
        """List recent Mail messages within days_back, optionally unread only."""
        return service.list_mail_messages(
            account_name=account_name,
            mailbox_name=mailbox_name,
            limit=limit,
            unread_only=unread_only,
            days_back=days_back,
        )

    @app.tool(name="mail_get_message")
    def mail_get_message(
        message_id: str,
        account_name: str | None = None,
        mailbox_name: str | None = None,
    ) -> dict[str, Any]:
        return service.get_mail_message(
            message_id=message_id,
            account_name=account_name,
            mailbox_name=mailbox_name,
        )

    @app.tool(name="mail_create_draft")
    def mail_create_draft(
        subject: str,
        body: str,
        to: list[str] | str | None = None,
        cc: list[str] | str | None = None,
        bcc: list[str] | str | None = None,
        sender: str | None = None,
        visible: bool = True,
    ) -> dict[str, Any]:
        return service.create_mail_draft(
            subject=subject,
            body=body,
            to=to,
            cc=cc,
            bcc=bcc,
            sender=sender,
            visible=visible,
        )

    @app.tool(name="mail_send_message")
    def mail_send_message(
        subject: str,
        body: str,
        to: list[str] | str,
        cc: list[str] | str | None = None,
        bcc: list[str] | str | None = None,
        sender: str | None = None,
        confirm_send: bool = False,
    ) -> dict[str, Any]:
        return service.send_mail_message(
            subject=subject,
            body=body,
            to=to,
            cc=cc,
            bcc=bcc,
            sender=sender,
            confirm_send=confirm_send,
        )

    return app


def main() -> None:
    try:
        app = build_macos_server()
    except (RuntimeError, AppleScriptBridgeError) as exc:
        raise SystemExit(str(exc)) from exc
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
