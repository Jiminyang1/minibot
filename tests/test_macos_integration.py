from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.macos import AppleScriptBridge
from minibot.tools.base import ToolExecutionContext
from minibot.tools.macos_apps import (
    CalendarCreateEventTool,
    CalendarListEventsTool,
    NotesAppendTool,
    NotesCreateTool,
    NotesSearchTool,
    RemindersCompleteTool,
    RemindersCreateTool,
)


_RUN_MACOS_INTEGRATION = (
    sys.platform == "darwin"
    and os.environ.get("MINIBOT_RUN_MACOS_INTEGRATION") == "1"
)


def _iso_local(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat(timespec="minutes")


def _run_cleanup(lines: list[str], *, args: list[str]) -> None:
    command = ["/usr/bin/osascript", "-l", "AppleScript"]
    for line in lines:
        command.extend(["-e", line])
    if args:
        command.append("--")
        command.extend(args)
    subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)


@unittest.skipUnless(
    _RUN_MACOS_INTEGRATION,
    "需要显式设置 MINIBOT_RUN_MACOS_INTEGRATION=1 才运行 macOS 集成测试。",
)
class MacOSIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = AppleScriptBridge()
        self.context = ToolExecutionContext(session_id="s_integration")

    def test_calendar_create_and_list_event(self) -> None:
        title = f"minibot-calendar-{uuid.uuid4().hex[:8]}"
        start_dt = datetime.now() + timedelta(days=1)
        start_dt = start_dt.replace(hour=10, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(hours=1)
        event_id: str | None = None
        calendar_name: str | None = None

        try:
            created = CalendarCreateEventTool(self.bridge).execute(
                context=self.context,
                title=title,
                start_at=_iso_local(start_dt),
                end_at=_iso_local(end_dt),
            )
            self.assertTrue(created.ok, created.summary)
            event_id = created.data["event_id"]
            calendar_name = created.data["calendar_name"]

            listed = CalendarListEventsTool(self.bridge).execute(
                context=self.context,
                start_at=_iso_local(start_dt - timedelta(hours=1)),
                end_at=_iso_local(end_dt + timedelta(hours=1)),
                calendar_name=calendar_name,
                limit=10,
            )
            self.assertTrue(listed.ok, listed.summary)
            self.assertIn(
                event_id,
                {item["event_id"] for item in listed.data["events"]},
            )
        finally:
            if event_id and calendar_name:
                _run_cleanup(
                    [
                        "on run argv",
                        "tell application \"Calendar\" to delete (first event of calendar (item 1 of argv) whose uid is (item 2 of argv))",
                        "end run",
                    ],
                    args=[calendar_name, event_id],
                )

    def test_reminders_create_and_complete(self) -> None:
        title = f"minibot-reminder-{uuid.uuid4().hex[:8]}"
        reminder_id: str | None = None

        try:
            created = RemindersCreateTool(self.bridge).execute(
                context=self.context,
                title=title,
            )
            self.assertTrue(created.ok, created.summary)
            reminder_id = created.data["reminder_id"]

            completed = RemindersCompleteTool(self.bridge).execute(
                context=self.context,
                reminder_id=reminder_id,
            )
            self.assertTrue(completed.ok, completed.summary)
            self.assertTrue(completed.data["completed"])
        finally:
            if reminder_id:
                _run_cleanup(
                    [
                        "on run argv",
                        "tell application \"Reminders\" to delete (reminder id (item 1 of argv))",
                        "end run",
                    ],
                    args=[reminder_id],
                )

    def test_notes_create_append_and_search(self) -> None:
        title = f"minibot-note-{uuid.uuid4().hex[:8]}"
        note_id: str | None = None

        try:
            created = NotesCreateTool(self.bridge).execute(
                context=self.context,
                title=title,
                content="first line",
            )
            self.assertTrue(created.ok, created.summary)
            note_id = created.data["note_id"]

            appended = NotesAppendTool(self.bridge).execute(
                context=self.context,
                note_id=note_id,
                content="second line",
            )
            self.assertTrue(appended.ok, appended.summary)

            listed = NotesSearchTool(self.bridge).execute(
                context=self.context,
                query=title,
                limit=10,
            )
            self.assertTrue(listed.ok, listed.summary)
            self.assertIn(
                note_id,
                {item["note_id"] for item in listed.data["notes"]},
            )
        finally:
            if note_id:
                _run_cleanup(
                    [
                        "on run argv",
                        "tell application \"Notes\" to delete (note id (item 1 of argv))",
                        "end run",
                    ],
                    args=[note_id],
                )


if __name__ == "__main__":
    unittest.main()
