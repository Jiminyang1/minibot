from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.mcp_servers.macos_system.bridge import (
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
from minibot.mcp_servers.macos_system.server import MacOSSystemService


class _FakeBridge:
    def __init__(self) -> None:
        self.should_fail = False
        self.last_list_mail_messages_kwargs: dict[str, object] | None = None

    def _maybe_fail(self) -> None:
        if self.should_fail:
            raise AppleScriptBridgeError("permission_denied", "automation denied")

    def list_calendar_events(self, **kwargs: object) -> list[CalendarEventRecord]:
        del kwargs
        self._maybe_fail()
        return [
            CalendarEventRecord(
                event_id="evt_1",
                title="Team sync",
                calendar_name="Work",
                start_at="2026-04-18T09:00:00",
                end_at="2026-04-18T10:00:00",
                location="Zoom",
                notes="Agenda",
            )
        ]

    def create_calendar_event(self, **kwargs: object) -> CalendarEventRecord:
        del kwargs
        self._maybe_fail()
        return CalendarEventRecord(
            event_id="evt_2",
            title="Office Hours",
            calendar_name="Work",
            start_at="2026-04-18T11:00:00",
            end_at="2026-04-18T12:00:00",
            location="Room 101",
            notes="Bring slides",
        )

    def list_reminders(self, **kwargs: object) -> list[ReminderRecord]:
        del kwargs
        self._maybe_fail()
        return [
            ReminderRecord(
                reminder_id="rem_1",
                title="Pay rent",
                list_name="Reminders",
                completed=False,
                due_at="2026-04-19T20:00:00",
                notes="bank transfer",
            )
        ]

    def create_reminder(self, **kwargs: object) -> ReminderRecord:
        del kwargs
        self._maybe_fail()
        return ReminderRecord(
            reminder_id="rem_2",
            title="Submit homework",
            list_name="School",
            completed=False,
            due_at=None,
            notes="CS598",
        )

    def complete_reminder(self, **kwargs: object) -> ReminderRecord:
        del kwargs
        self._maybe_fail()
        return ReminderRecord(
            reminder_id="rem_3",
            title="Book flights",
            list_name="Travel",
            completed=True,
            due_at=None,
            notes="",
        )

    def search_notes(self, **kwargs: object) -> list[NoteRecord]:
        del kwargs
        self._maybe_fail()
        return [
            NoteRecord(
                note_id="note_1",
                title="Agent ideas",
                folder_name="Notes",
                preview="workflow and local assistant",
            )
        ]

    def create_note(self, **kwargs: object) -> NoteRecord:
        del kwargs
        self._maybe_fail()
        return NoteRecord(
            note_id="note_2",
            title="Research log",
            folder_name="Notes",
            preview="new entry",
        )

    def append_note(self, **kwargs: object) -> NoteRecord:
        del kwargs
        self._maybe_fail()
        return NoteRecord(
            note_id="note_3",
            title="Research log",
            folder_name="Notes",
            preview="existing entry plus more",
        )

    def list_mailboxes(self, **kwargs: object) -> list[MailboxRecord]:
        del kwargs
        self._maybe_fail()
        return [
            MailboxRecord(
                account_name="iCloud",
                mailbox_name="Inbox",
                unread_count=2,
                message_count=42,
            )
        ]

    def search_mail_messages(self, **kwargs: object) -> list[MailMessageRecord]:
        del kwargs
        self._maybe_fail()
        return [
            MailMessageRecord(
                message_id="mail_1",
                subject="Project update",
                sender="Ada <ada@example.com>",
                received_at="2026-04-18T08:30:00",
                mailbox_name="Inbox",
                account_name="iCloud",
                read=False,
                preview="Here is the update",
            )
        ]

    def list_mail_messages(self, **kwargs: object) -> list[MailMessageRecord]:
        self.last_list_mail_messages_kwargs = kwargs
        self._maybe_fail()
        return [
            MailMessageRecord(
                message_id="mail_recent_1",
                subject="Recent update",
                sender="Grace <grace@example.com>",
                received_at="2026-04-19T08:30:00",
                mailbox_name="INBOX",
                account_name="iCloud",
                read=True,
                preview="",
            )
        ]

    def get_mail_message(self, **kwargs: object) -> MailMessageBodyRecord:
        del kwargs
        self._maybe_fail()
        return MailMessageBodyRecord(
            message_id="mail_1",
            subject="Project update",
            sender="Ada <ada@example.com>",
            received_at="2026-04-18T08:30:00",
            mailbox_name="Inbox",
            account_name="iCloud",
            read=False,
            body="Here is the full update",
        )

    def create_mail_draft(self, **kwargs: object) -> MailDraftRecord:
        del kwargs
        self._maybe_fail()
        return MailDraftRecord(
            subject="Hello",
            to=["ada@example.com"],
            cc=[],
            bcc=[],
            sender="",
            visible=True,
            preview="Draft body",
        )

    def send_mail_message(self, **kwargs: object) -> MailSendRecord:
        del kwargs
        self._maybe_fail()
        return MailSendRecord(
            subject="Hello",
            to=["ada@example.com"],
            cc=[],
            bcc=[],
            sender="",
            sent=True,
            preview="Sent body",
        )


class MacOSSystemServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = _FakeBridge()
        self.service = MacOSSystemService(self.bridge)

    def test_read_methods_return_serialized_payloads(self) -> None:
        calendar_result = self.service.list_calendar_events(
            start_at="2026-04-18T00:00",
            end_at="2026-04-19T00:00",
        )
        reminder_result = self.service.list_reminders()
        notes_result = self.service.search_notes(query="agent")
        mailbox_result = self.service.list_mailboxes()
        mail_search_result = self.service.search_mail_messages(query="project")
        mail_list_result = self.service.list_mail_messages(mailbox_name="INBOX")
        mail_get_result = self.service.get_mail_message(message_id="mail_1")

        self.assertEqual(calendar_result["events"][0]["event_id"], "evt_1")
        self.assertEqual(reminder_result["reminders"][0]["reminder_id"], "rem_1")
        self.assertEqual(notes_result["notes"][0]["note_id"], "note_1")
        self.assertEqual(mailbox_result["mailboxes"][0]["mailbox_name"], "Inbox")
        self.assertEqual(mail_search_result["messages"][0]["message_id"], "mail_1")
        self.assertEqual(
            mail_list_result["messages"][0]["message_id"],
            "mail_recent_1",
        )
        self.assertEqual(mail_list_result["days_back"], 7)
        self.assertEqual(self.bridge.last_list_mail_messages_kwargs["days_back"], 7)
        self.assertEqual(mail_get_result["body"], "Here is the full update")

    def test_list_mail_messages_allows_count_only_recent_mode(self) -> None:
        result = self.service.list_mail_messages(
            mailbox_name="INBOX",
            limit=20,
            days_back=None,
        )

        self.assertEqual(result["days_back"], None)
        self.assertEqual(self.bridge.last_list_mail_messages_kwargs["limit"], 20)
        self.assertEqual(self.bridge.last_list_mail_messages_kwargs["days_back"], None)

    def test_write_methods_return_serialized_payloads(self) -> None:
        calendar_result = self.service.create_calendar_event(
            title="Office Hours",
            start_at="2026-04-18T11:00",
            end_at="2026-04-18T12:00",
        )
        reminder_result = self.service.create_reminder(title="Submit homework")
        complete_result = self.service.complete_reminder(reminder_id="rem_3")
        note_create_result = self.service.create_note(
            title="Research log",
            content="new entry",
        )
        note_append_result = self.service.append_note(
            note_id="note_3",
            content="more",
        )
        draft_result = self.service.create_mail_draft(
            subject="Hello",
            body="Draft body",
            to=["ada@example.com"],
        )
        send_result = self.service.send_mail_message(
            subject="Hello",
            body="Sent body",
            to=["ada@example.com"],
            confirm_send=True,
        )

        self.assertEqual(calendar_result["event_id"], "evt_2")
        self.assertEqual(reminder_result["reminder_id"], "rem_2")
        self.assertTrue(complete_result["completed"])
        self.assertEqual(note_create_result["note_id"], "note_2")
        self.assertEqual(note_append_result["note_id"], "note_3")
        self.assertEqual(draft_result["to"], ["ada@example.com"])
        self.assertTrue(send_result["sent"])

    def test_send_mail_requires_explicit_confirmation(self) -> None:
        with self.assertRaises(AppleScriptBridgeError) as ctx:
            self.service.send_mail_message(
                subject="Hello",
                body="Sent body",
                to=["ada@example.com"],
            )

        self.assertEqual(ctx.exception.code, "invalid_args")
        self.assertEqual(ctx.exception.data["field"], "confirm_send")

    def test_bridge_errors_are_not_swallowed(self) -> None:
        self.bridge.should_fail = True

        with self.assertRaises(AppleScriptBridgeError):
            self.service.create_calendar_event(
                title="Office Hours",
                start_at="2026-04-18T11:00",
                end_at="2026-04-18T12:00",
            )


if __name__ == "__main__":
    unittest.main()
