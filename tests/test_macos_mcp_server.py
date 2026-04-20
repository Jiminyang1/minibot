from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.macos import (
    AppleScriptBridgeError,
    CalendarEventRecord,
    NoteRecord,
    ReminderRecord,
)
from minibot.mcp_servers.macos_system_server import MacOSSystemService


class _FakeBridge:
    def __init__(self) -> None:
        self.should_fail = False

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

        self.assertEqual(calendar_result["events"][0]["event_id"], "evt_1")
        self.assertEqual(reminder_result["reminders"][0]["reminder_id"], "rem_1")
        self.assertEqual(notes_result["notes"][0]["note_id"], "note_1")

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

        self.assertEqual(calendar_result["event_id"], "evt_2")
        self.assertEqual(reminder_result["reminder_id"], "rem_2")
        self.assertTrue(complete_result["completed"])
        self.assertEqual(note_create_result["note_id"], "note_2")
        self.assertEqual(note_append_result["note_id"], "note_3")

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
