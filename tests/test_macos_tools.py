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
from minibot.tools.base import ToolExecutionContext
from minibot.tools.macos_apps import (
    CalendarCreateEventTool,
    CalendarListEventsTool,
    NotesAppendTool,
    NotesCreateTool,
    NotesSearchTool,
    RemindersCompleteTool,
    RemindersCreateTool,
    RemindersListTool,
)


class _FakeBridge:
    def __init__(self) -> None:
        self.should_fail = False
        self.fail_code = "error"
        self.fail_message = "bridge failed"
        self.fail_data = {"reason": "boom"}

    def _maybe_fail(self) -> None:
        if self.should_fail:
            raise AppleScriptBridgeError(
                self.fail_code,
                self.fail_message,
                data=self.fail_data,
            )

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


class MacOSToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = _FakeBridge()
        self.context = ToolExecutionContext(session_id="s_test")

    def test_read_tools_return_structured_data(self) -> None:
        calendar_result = CalendarListEventsTool(self.bridge).execute(
            context=self.context,
            start_at="2026-04-18T00:00",
            end_at="2026-04-19T00:00",
        )
        reminder_result = RemindersListTool(self.bridge).execute(context=self.context)
        notes_result = NotesSearchTool(self.bridge).execute(
            context=self.context,
            query="agent",
        )

        self.assertTrue(calendar_result.ok)
        self.assertEqual(calendar_result.data["events"][0]["event_id"], "evt_1")
        self.assertTrue(reminder_result.ok)
        self.assertEqual(reminder_result.data["reminders"][0]["reminder_id"], "rem_1")
        self.assertTrue(notes_result.ok)
        self.assertEqual(notes_result.data["notes"][0]["note_id"], "note_1")

    def test_write_tools_require_approval(self) -> None:
        self.assertTrue(CalendarCreateEventTool(self.bridge).requires_approval)
        self.assertTrue(RemindersCreateTool(self.bridge).requires_approval)
        self.assertTrue(RemindersCompleteTool(self.bridge).requires_approval)
        self.assertTrue(NotesCreateTool(self.bridge).requires_approval)
        self.assertTrue(NotesAppendTool(self.bridge).requires_approval)

    def test_write_tools_return_stable_refs(self) -> None:
        calendar_result = CalendarCreateEventTool(self.bridge).execute(
            context=self.context,
            title="Office Hours",
            start_at="2026-04-18T11:00",
            end_at="2026-04-18T12:00",
        )
        reminder_result = RemindersCreateTool(self.bridge).execute(
            context=self.context,
            title="Submit homework",
        )
        complete_result = RemindersCompleteTool(self.bridge).execute(
            context=self.context,
            reminder_id="rem_3",
        )
        note_create_result = NotesCreateTool(self.bridge).execute(
            context=self.context,
            title="Research log",
            content="new entry",
        )
        note_append_result = NotesAppendTool(self.bridge).execute(
            context=self.context,
            note_id="note_3",
            content="more",
        )

        self.assertEqual(calendar_result.data["event_id"], "evt_2")
        self.assertEqual(reminder_result.data["reminder_id"], "rem_2")
        self.assertTrue(complete_result.data["completed"])
        self.assertEqual(note_create_result.data["note_id"], "note_2")
        self.assertEqual(note_append_result.data["note_id"], "note_3")

    def test_bridge_errors_map_to_tool_result(self) -> None:
        self.bridge.should_fail = True
        self.bridge.fail_code = "permission_denied"
        self.bridge.fail_message = "automation denied"
        self.bridge.fail_data = {"app": "Calendar"}

        result = CalendarCreateEventTool(self.bridge).execute(
            context=self.context,
            title="Office Hours",
            start_at="2026-04-18T11:00",
            end_at="2026-04-18T12:00",
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "permission_denied")
        self.assertEqual(result.summary, "automation denied")
        self.assertEqual(result.data["app"], "Calendar")


if __name__ == "__main__":
    unittest.main()
