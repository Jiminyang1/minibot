"""AppleScript bridge for macOS builtin apps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import shutil
import subprocess
from typing import Any


_FIELD_SEP = chr(31)
_RECORD_SEP = chr(30)
_COMMON_HELPERS = [
    "on pad2(n)",
    "set valueInt to n as integer",
    "if valueInt < 10 then return \"0\" & (valueInt as string)",
    "return valueInt as string",
    "end pad2",
    "on makeDate(y, m, d, hh, mm, ss)",
    "set dt to current date",
    "set year of dt to (y as integer)",
    "set month of dt to (m as integer)",
    "set day of dt to (d as integer)",
    "set time of dt to ((hh as integer) * hours + (mm as integer) * minutes + (ss as integer))",
    "return dt",
    "end makeDate",
    "on formatDate(dt)",
    "if dt is missing value then return \"\"",
    "return (year of dt as string) & \"-\" & pad2(month of dt as integer) & \"-\" & pad2(day of dt as integer) & \"T\" & pad2(hours of dt) & \":\" & pad2(minutes of dt) & \":\" & pad2(seconds of dt)",
    "end formatDate",
    "on sanitizeText(txt)",
    "if txt is missing value then return \"\"",
    "set normalized to txt as string",
    "set oldDelims to AppleScript's text item delimiters",
    "set AppleScript's text item delimiters to {return, linefeed, tab, (character id 31), (character id 30)}",
    "set parts to text items of normalized",
    "set AppleScript's text item delimiters to \" \"",
    "set flattened to parts as text",
    "set AppleScript's text item delimiters to oldDelims",
    "return flattened",
    "end sanitizeText",
    "on splitCommaText(rawText)",
    "if rawText is \"\" then return {}",
    "set oldDelims to AppleScript's text item delimiters",
    "set AppleScript's text item delimiters to \",\"",
    "set rawParts to text items of (rawText as string)",
    "set AppleScript's text item delimiters to oldDelims",
    "set cleanedParts to {}",
    "repeat with rawPart in rawParts",
    "set partText to rawPart as string",
    "if partText is not \"\" then set end of cleanedParts to partText",
    "end repeat",
    "return cleanedParts",
    "end splitCommaText",
    "on truncateText(txt, maxChars)",
    "set cleaned to sanitizeText(txt)",
    "if (length of cleaned) <= maxChars then return cleaned",
    "return text 1 thru maxChars of cleaned",
    "end truncateText",
    "on joinFields(fieldValues)",
    "set outputText to \"\"",
    "set fieldCount to count of fieldValues",
    "repeat with idx from 1 to fieldCount",
    "if idx > 1 then set outputText to outputText & (character id 31)",
    "set outputText to outputText & (item idx of fieldValues as string)",
    "end repeat",
    "return outputText",
    "end joinFields",
    "on joinRecords(recordValues)",
    "set outputText to \"\"",
    "set recordCount to count of recordValues",
    "repeat with idx from 1 to recordCount",
    "if idx > 1 then set outputText to outputText & (character id 30)",
    "set outputText to outputText & (item idx of recordValues as string)",
    "end repeat",
    "return outputText",
    "end joinRecords",
]


@dataclass(frozen=True)
class CalendarEventRecord:
    event_id: str
    title: str
    calendar_name: str
    start_at: str
    end_at: str
    location: str
    notes: str


@dataclass(frozen=True)
class ReminderRecord:
    reminder_id: str
    title: str
    list_name: str
    completed: bool
    due_at: str | None
    notes: str


@dataclass(frozen=True)
class NoteRecord:
    note_id: str
    title: str
    folder_name: str
    preview: str


@dataclass(frozen=True)
class MailboxRecord:
    account_name: str
    mailbox_name: str
    unread_count: int
    message_count: int


@dataclass(frozen=True)
class MailMessageRecord:
    message_id: str
    subject: str
    sender: str
    received_at: str
    mailbox_name: str
    account_name: str
    read: bool
    preview: str


@dataclass(frozen=True)
class MailMessageBodyRecord:
    message_id: str
    subject: str
    sender: str
    received_at: str
    mailbox_name: str
    account_name: str
    read: bool
    body: str


@dataclass(frozen=True)
class MailDraftRecord:
    subject: str
    to: list[str]
    cc: list[str]
    bcc: list[str]
    sender: str
    visible: bool
    preview: str


@dataclass(frozen=True)
class MailSendRecord:
    subject: str
    to: list[str]
    cc: list[str]
    bcc: list[str]
    sender: str
    sent: bool
    preview: str


class AppleScriptBridgeError(RuntimeError):
    """Structured AppleScript failure surfaced to tool layer."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class AppleScriptBridge:
    """Typed AppleScript operations for Calendar, Reminders, Notes, and Mail."""

    _DEFAULT_TIMEOUT_SECONDS = 20

    def __init__(
        self,
        *,
        osascript_path: str | None = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.osascript_path = osascript_path or shutil.which("osascript")
        self.timeout_seconds = timeout_seconds
        if not self.osascript_path:
            raise RuntimeError("当前系统不可用 osascript。")

    @staticmethod
    def is_supported() -> bool:
        return shutil.which("osascript") is not None

    def list_calendar_events(
        self,
        *,
        start_at: str,
        end_at: str,
        calendar_name: str | None,
        limit: int,
    ) -> list[CalendarEventRecord]:
        start_dt = self._parse_local_datetime(start_at, field_name="start_at")
        end_dt = self._parse_local_datetime(end_at, field_name="end_at")
        self._ensure_range(start_dt, end_dt, start_field="start_at", end_field="end_at")
        self._validate_limit(limit)

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set startDate to my makeDate(item 1 of argv, item 2 of argv, item 3 of argv, item 4 of argv, item 5 of argv, item 6 of argv)",
                "set endDate to my makeDate(item 7 of argv, item 8 of argv, item 9 of argv, item 10 of argv, item 11 of argv, item 12 of argv)",
                "set calendarFilter to item 13 of argv",
                "tell application \"Calendar\"",
                "if calendarFilter is \"\" then",
                "set targetCalendars to calendars",
                "else",
                "set targetCalendars to {calendar calendarFilter}",
                "end if",
                "set rows to {}",
                "repeat with targetCalendar in targetCalendars",
                "set hits to every event of targetCalendar whose start date >= startDate and start date < endDate",
                "repeat with eachEvent in hits",
                "set locationText to \"\"",
                "try",
                "set locationText to location of eachEvent",
                "end try",
                "set notesText to \"\"",
                "try",
                "set notesText to description of eachEvent",
                "end try",
                "set end of rows to my joinFields({uid of eachEvent, my sanitizeText(summary of eachEvent), my sanitizeText(name of targetCalendar), my formatDate(start date of eachEvent), my formatDate(end date of eachEvent), my sanitizeText(locationText), my sanitizeText(notesText)})",
                "end repeat",
                "end repeat",
                "if (count of rows) is 0 then return \"\"",
                "return my joinRecords(rows)",
                "end tell",
                "end run",
            ],
            args=[
                *self._datetime_components(start_dt),
                *self._datetime_components(end_dt),
                calendar_name or "",
            ],
        )
        records = [
            CalendarEventRecord(
                event_id=item["event_id"],
                title=item["title"],
                calendar_name=item["calendar_name"],
                start_at=item["start_at"],
                end_at=item["end_at"],
                location=item["location"],
                notes=item["notes"],
            )
            for item in self._parse_records(
                raw,
                (
                    "event_id",
                    "title",
                    "calendar_name",
                    "start_at",
                    "end_at",
                    "location",
                    "notes",
                ),
            )
        ]
        return sorted(
            records,
            key=lambda item: (item.start_at, item.end_at, item.title, item.event_id),
        )[:limit]

    def create_calendar_event(
        self,
        *,
        title: str,
        start_at: str,
        end_at: str,
        calendar_name: str | None,
        location: str | None,
        notes: str | None,
    ) -> CalendarEventRecord:
        normalized_title = title.strip()
        if not normalized_title:
            raise AppleScriptBridgeError(
                "invalid_args",
                "title 不能为空。",
                data={"field": "title"},
            )
        start_dt = self._parse_local_datetime(start_at, field_name="start_at")
        end_dt = self._parse_local_datetime(end_at, field_name="end_at")
        self._ensure_range(start_dt, end_dt, start_field="start_at", end_field="end_at")

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set eventTitle to item 1 of argv",
                "set startDate to my makeDate(item 2 of argv, item 3 of argv, item 4 of argv, item 5 of argv, item 6 of argv, item 7 of argv)",
                "set endDate to my makeDate(item 8 of argv, item 9 of argv, item 10 of argv, item 11 of argv, item 12 of argv, item 13 of argv)",
                "set calendarFilter to item 14 of argv",
                "set locationText to item 15 of argv",
                "set notesText to item 16 of argv",
                "tell application \"Calendar\"",
                "if calendarFilter is \"\" then",
                "set targetCalendar to missing value",
                "repeat with candidateCalendar in calendars",
                "try",
                "if writable of candidateCalendar then",
                "set targetCalendar to candidateCalendar",
                "exit repeat",
                "end if",
                "end try",
                "end repeat",
                "if targetCalendar is missing value then error \"No writable calendar available\"",
                "else",
                "set targetCalendar to calendar calendarFilter",
                "if writable of targetCalendar is false then error \"Selected calendar is read-only\"",
                "end if",
                "set newEvent to make new event at end of events of targetCalendar with properties {summary:eventTitle, start date:startDate, end date:endDate}",
                "if locationText is not \"\" then set location of newEvent to locationText",
                "if notesText is not \"\" then set description of newEvent to notesText",
                "set locationOut to \"\"",
                "try",
                "set locationOut to location of newEvent",
                "end try",
                "set notesOut to \"\"",
                "try",
                "set notesOut to description of newEvent",
                "end try",
                "return my joinFields({uid of newEvent, my sanitizeText(summary of newEvent), my sanitizeText(name of targetCalendar), my formatDate(start date of newEvent), my formatDate(end date of newEvent), my sanitizeText(locationOut), my sanitizeText(notesOut)})",
                "end tell",
                "end run",
            ],
            args=[
                normalized_title,
                *self._datetime_components(start_dt),
                *self._datetime_components(end_dt),
                calendar_name or "",
                (location or "").strip(),
                (notes or "").strip(),
            ],
        )
        item = self._parse_single_record(
            raw,
            (
                "event_id",
                "title",
                "calendar_name",
                "start_at",
                "end_at",
                "location",
                "notes",
            ),
        )
        return CalendarEventRecord(
            event_id=item["event_id"],
            title=item["title"],
            calendar_name=item["calendar_name"],
            start_at=item["start_at"],
            end_at=item["end_at"],
            location=item["location"],
            notes=item["notes"],
        )

    def list_reminders(
        self,
        *,
        list_name: str | None,
        status: str,
        limit: int,
    ) -> list[ReminderRecord]:
        normalized_status = status.strip().lower() if status else "open"
        if normalized_status not in {"open", "completed", "all"}:
            raise AppleScriptBridgeError(
                "invalid_args",
                "status 只能是 open、completed 或 all。",
                data={"field": "status", "value": status},
            )
        self._validate_limit(limit)

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set listFilter to item 1 of argv",
                "set statusFilter to item 2 of argv",
                "tell application \"Reminders\"",
                "if listFilter is \"\" then",
                "set targetLists to lists",
                "else",
                "set targetLists to {list listFilter}",
                "end if",
                "set rows to {}",
                "repeat with targetList in targetLists",
                "repeat with eachReminder in reminders of targetList",
                "set isCompleted to completed of eachReminder",
                "if statusFilter is \"open\" and isCompleted then",
                "else if statusFilter is \"completed\" and (isCompleted is false) then",
                "else",
                "set dueText to \"\"",
                "try",
                "set dueText to my formatDate(due date of eachReminder)",
                "end try",
                "set notesText to \"\"",
                "try",
                "set notesText to body of eachReminder",
                "end try",
                "set completedText to \"false\"",
                "if isCompleted then set completedText to \"true\"",
                "set end of rows to my joinFields({id of eachReminder, my sanitizeText(name of eachReminder), my sanitizeText(name of targetList), completedText, dueText, my sanitizeText(notesText)})",
                "end if",
                "end repeat",
                "end repeat",
                "if (count of rows) is 0 then return \"\"",
                "return my joinRecords(rows)",
                "end tell",
                "end run",
            ],
            args=[list_name or "", normalized_status],
        )
        records = [
            ReminderRecord(
                reminder_id=item["reminder_id"],
                title=item["title"],
                list_name=item["list_name"],
                completed=item["completed"] == "true",
                due_at=item["due_at"] or None,
                notes=item["notes"],
            )
            for item in self._parse_records(
                raw,
                (
                    "reminder_id",
                    "title",
                    "list_name",
                    "completed",
                    "due_at",
                    "notes",
                ),
            )
        ]
        records.sort(
            key=lambda item: (
                item.completed,
                item.due_at or "9999-99-99T99:99:99",
                item.title,
                item.reminder_id,
            )
        )
        return records[:limit]

    def create_reminder(
        self,
        *,
        title: str,
        due_at: str | None,
        list_name: str | None,
        notes: str | None,
    ) -> ReminderRecord:
        normalized_title = title.strip()
        if not normalized_title:
            raise AppleScriptBridgeError(
                "invalid_args",
                "title 不能为空。",
                data={"field": "title"},
            )

        due_args = ["0", "0", "0", "0", "0", "0", "0"]
        if due_at:
            due_dt = self._parse_local_datetime(due_at, field_name="due_at")
            due_args = ["1", *self._datetime_components(due_dt)]

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set reminderTitle to item 1 of argv",
                "set dueFlag to item 2 of argv",
                "set listFilter to item 9 of argv",
                "set notesText to item 10 of argv",
                "tell application \"Reminders\"",
                "if listFilter is \"\" then",
                "try",
                "set targetList to first list of default account",
                "on error",
                "set targetList to first list",
                "end try",
                "else",
                "set targetList to list listFilter",
                "end if",
                "set newReminder to make new reminder at end of reminders of targetList with properties {name:reminderTitle}",
                "if dueFlag is \"1\" then set due date of newReminder to my makeDate(item 3 of argv, item 4 of argv, item 5 of argv, item 6 of argv, item 7 of argv, item 8 of argv)",
                "if notesText is not \"\" then set body of newReminder to notesText",
                "set dueText to \"\"",
                "try",
                "set dueText to my formatDate(due date of newReminder)",
                "end try",
                "set notesOut to \"\"",
                "try",
                "set notesOut to body of newReminder",
                "end try",
                "return my joinFields({id of newReminder, my sanitizeText(name of newReminder), my sanitizeText(name of targetList), \"false\", dueText, my sanitizeText(notesOut)})",
                "end tell",
                "end run",
            ],
            args=[
                normalized_title,
                *due_args,
                list_name or "",
                (notes or "").strip(),
            ],
        )
        item = self._parse_single_record(
            raw,
            (
                "reminder_id",
                "title",
                "list_name",
                "completed",
                "due_at",
                "notes",
            ),
        )
        return ReminderRecord(
            reminder_id=item["reminder_id"],
            title=item["title"],
            list_name=item["list_name"],
            completed=item["completed"] == "true",
            due_at=item["due_at"] or None,
            notes=item["notes"],
        )

    def complete_reminder(self, *, reminder_id: str) -> ReminderRecord:
        normalized_id = reminder_id.strip()
        if not normalized_id:
            raise AppleScriptBridgeError(
                "invalid_args",
                "reminder_id 不能为空。",
                data={"field": "reminder_id"},
            )
        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set reminderId to item 1 of argv",
                "tell application \"Reminders\"",
                "set targetReminder to reminder id reminderId",
                "set completed of targetReminder to true",
                "set dueText to \"\"",
                "try",
                "set dueText to my formatDate(due date of targetReminder)",
                "end try",
                "set notesText to \"\"",
                "try",
                "set notesText to body of targetReminder",
                "end try",
                "set listName to \"\"",
                "repeat with candidateList in lists",
                "try",
                "set matchedReminders to every reminder of candidateList whose id is reminderId",
                "if (count of matchedReminders) > 0 then",
                "set listName to name of candidateList",
                "exit repeat",
                "end if",
                "end try",
                "end repeat",
                "return my joinFields({id of targetReminder, my sanitizeText(name of targetReminder), my sanitizeText(listName), \"true\", dueText, my sanitizeText(notesText)})",
                "end tell",
                "end run",
            ],
            args=[normalized_id],
        )
        item = self._parse_single_record(
            raw,
            (
                "reminder_id",
                "title",
                "list_name",
                "completed",
                "due_at",
                "notes",
            ),
        )
        return ReminderRecord(
            reminder_id=item["reminder_id"],
            title=item["title"],
            list_name=item["list_name"],
            completed=True,
            due_at=item["due_at"] or None,
            notes=item["notes"],
        )

    def search_notes(
        self,
        *,
        query: str,
        folder_name: str | None,
        limit: int,
    ) -> list[NoteRecord]:
        normalized_query = query.strip()
        if not normalized_query:
            raise AppleScriptBridgeError(
                "invalid_args",
                "query 不能为空。",
                data={"field": "query"},
            )
        self._validate_limit(limit)

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set queryText to item 1 of argv",
                "set folderFilter to item 2 of argv",
                "tell application \"Notes\"",
                "if folderFilter is \"\" then",
                "set targetFolders to folders",
                "else",
                "set targetFolders to {folder folderFilter}",
                "end if",
                "set rows to {}",
                "ignoring case",
                "repeat with targetFolder in targetFolders",
                "repeat with eachNote in notes of targetFolder",
                "set noteText to plaintext of eachNote",
                "if ((name of eachNote contains queryText) or (noteText contains queryText)) then",
                "set end of rows to my joinFields({id of eachNote, my sanitizeText(name of eachNote), my sanitizeText(name of targetFolder), my truncateText(noteText, 180)})",
                "end if",
                "end repeat",
                "end repeat",
                "end ignoring",
                "if (count of rows) is 0 then return \"\"",
                "return my joinRecords(rows)",
                "end tell",
                "end run",
            ],
            args=[normalized_query, folder_name or ""],
        )
        records = [
            NoteRecord(
                note_id=item["note_id"],
                title=item["title"],
                folder_name=item["folder_name"],
                preview=item["preview"],
            )
            for item in self._parse_records(
                raw,
                ("note_id", "title", "folder_name", "preview"),
            )
        ]
        records.sort(key=lambda item: (item.folder_name, item.title, item.note_id))
        return records[:limit]

    def create_note(
        self,
        *,
        title: str,
        content: str,
        folder_name: str | None,
    ) -> NoteRecord:
        normalized_title = title.strip()
        if not normalized_title:
            raise AppleScriptBridgeError(
                "invalid_args",
                "title 不能为空。",
                data={"field": "title"},
            )
        if not content.strip():
            raise AppleScriptBridgeError(
                "invalid_args",
                "content 不能为空。",
                data={"field": "content"},
            )
        html_body = self._plain_text_to_notes_html(content)
        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set noteTitle to item 1 of argv",
                "set noteBody to item 2 of argv",
                "set folderFilter to item 3 of argv",
                "tell application \"Notes\"",
                "if folderFilter is \"\" then",
                "try",
                "set targetFolder to folder \"Notes\"",
                "on error",
                "set targetFolder to first folder",
                "end try",
                "else",
                "set targetFolder to folder folderFilter",
                "end if",
                "set newNote to make new note at targetFolder with properties {name:noteTitle, body:noteBody}",
                "return my joinFields({id of newNote, my sanitizeText(name of newNote), my sanitizeText(name of targetFolder), my truncateText(plaintext of newNote, 180)})",
                "end tell",
                "end run",
            ],
            args=[normalized_title, html_body, folder_name or ""],
        )
        item = self._parse_single_record(
            raw,
            ("note_id", "title", "folder_name", "preview"),
        )
        return NoteRecord(
            note_id=item["note_id"],
            title=item["title"],
            folder_name=item["folder_name"],
            preview=item["preview"],
        )

    def append_note(
        self,
        *,
        note_id: str,
        content: str,
    ) -> NoteRecord:
        normalized_id = note_id.strip()
        if not normalized_id:
            raise AppleScriptBridgeError(
                "invalid_args",
                "note_id 不能为空。",
                data={"field": "note_id"},
            )
        if not content.strip():
            raise AppleScriptBridgeError(
                "invalid_args",
                "content 不能为空。",
                data={"field": "content"},
            )
        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set noteId to item 1 of argv",
                "set htmlFragment to item 2 of argv",
                "tell application \"Notes\"",
                "set targetNote to note id noteId",
                "set body of targetNote to (body of targetNote) & htmlFragment",
                "set folderName to \"\"",
                "repeat with candidateFolder in folders",
                "try",
                "set matchedNotes to every note of candidateFolder whose id is noteId",
                "if (count of matchedNotes) > 0 then",
                "set folderName to name of candidateFolder",
                "exit repeat",
                "end if",
                "end try",
                "end repeat",
                "return my joinFields({id of targetNote, my sanitizeText(name of targetNote), my sanitizeText(folderName), my truncateText(plaintext of targetNote, 180)})",
                "end tell",
                "end run",
            ],
            args=[normalized_id, self._plain_text_to_notes_html(content)],
        )
        item = self._parse_single_record(
            raw,
            ("note_id", "title", "folder_name", "preview"),
        )
        return NoteRecord(
            note_id=item["note_id"],
            title=item["title"],
            folder_name=item["folder_name"],
            preview=item["preview"],
        )

    def list_mailboxes(
        self,
        *,
        account_name: str | None,
        limit: int,
    ) -> list[MailboxRecord]:
        self._validate_limit(limit)
        normalized_account = (account_name or "").strip()

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set accountFilter to item 1 of argv",
                "tell application \"Mail\"",
                "if accountFilter is \"\" then",
                "set targetAccounts to accounts",
                "else",
                "set targetAccounts to {account accountFilter}",
                "end if",
                "set rows to {}",
                "repeat with targetAccount in targetAccounts",
                "set accountName to name of targetAccount",
                "repeat with targetMailbox in mailboxes of targetAccount",
                "set unreadText to \"0\"",
                "try",
                "set unreadText to unread count of targetMailbox as string",
                "end try",
                "set messageCountText to \"0\"",
                "try",
                "set messageCountText to count of messages of targetMailbox as string",
                "end try",
                "set end of rows to my joinFields({my sanitizeText(accountName), my sanitizeText(name of targetMailbox), unreadText, messageCountText})",
                "end repeat",
                "end repeat",
                "if (count of rows) is 0 then return \"\"",
                "return my joinRecords(rows)",
                "end tell",
                "end run",
            ],
            args=[normalized_account],
        )
        records = [
            MailboxRecord(
                account_name=item["account_name"],
                mailbox_name=item["mailbox_name"],
                unread_count=self._coerce_int(item["unread_count"]),
                message_count=self._coerce_int(item["message_count"]),
            )
            for item in self._parse_records(
                raw,
                ("account_name", "mailbox_name", "unread_count", "message_count"),
            )
        ]
        records.sort(key=lambda item: (item.account_name, item.mailbox_name))
        return records[:limit]

    def search_mail_messages(
        self,
        *,
        query: str,
        account_name: str | None,
        mailbox_name: str | None,
        limit: int,
        include_body: bool = False,
    ) -> list[MailMessageRecord]:
        normalized_query = query.strip()
        if not normalized_query:
            raise AppleScriptBridgeError(
                "invalid_args",
                "query 不能为空。",
                data={"field": "query"},
            )
        self._validate_limit(limit)

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set queryText to item 1 of argv",
                "set accountFilter to item 2 of argv",
                "set mailboxFilter to item 3 of argv",
                "set rowLimit to item 4 of argv as integer",
                "set includeBody to false",
                "if item 5 of argv is \"1\" then set includeBody to true",
                "tell application \"Mail\"",
                "if accountFilter is \"\" then",
                "set targetAccounts to accounts",
                "else",
                "set targetAccounts to {account accountFilter}",
                "end if",
                "set rows to {}",
                "repeat with targetAccount in targetAccounts",
                "set accountName to name of targetAccount",
                "if mailboxFilter is \"\" then",
                "set targetMailboxes to mailboxes of targetAccount",
                "else",
                "set targetMailboxes to {mailbox mailboxFilter of targetAccount}",
                "end if",
                "repeat with targetMailbox in targetMailboxes",
                "set mailboxName to name of targetMailbox",
                "repeat with eachMessage in messages of targetMailbox",
                "set subjectText to \"\"",
                "try",
                "set subjectText to subject of eachMessage",
                "end try",
                "set senderText to \"\"",
                "try",
                "set senderText to sender of eachMessage",
                "end try",
                "set contentText to \"\"",
                "if includeBody then",
                "try",
                "set contentText to content of eachMessage",
                "end try",
                "end if",
                "set matchedMessage to false",
                "ignoring case",
                "if ((subjectText contains queryText) or (senderText contains queryText)) then set matchedMessage to true",
                "if includeBody is true then",
                "if contentText contains queryText then set matchedMessage to true",
                "end if",
                "end ignoring",
                "if matchedMessage then",
                "set receivedText to \"\"",
                "try",
                "set receivedText to my formatDate(date received of eachMessage)",
                "end try",
                "set readText to \"false\"",
                "try",
                "set readText to read status of eachMessage as string",
                "end try",
                "set end of rows to my joinFields({id of eachMessage as string, my sanitizeText(subjectText), my sanitizeText(senderText), receivedText, my sanitizeText(mailboxName), my sanitizeText(accountName), readText, my truncateText(contentText, 180)})",
                "end if",
                "if (count of rows) >= rowLimit then exit repeat",
                "end repeat",
                "if (count of rows) >= rowLimit then exit repeat",
                "end repeat",
                "if (count of rows) >= rowLimit then exit repeat",
                "end repeat",
                "if (count of rows) is 0 then return \"\"",
                "return my joinRecords(rows)",
                "end tell",
                "end run",
            ],
            args=[
                normalized_query,
                account_name or "",
                mailbox_name or "",
                str(limit),
                "1" if include_body else "0",
            ],
        )
        records = [
            MailMessageRecord(
                message_id=item["message_id"],
                subject=item["subject"],
                sender=item["sender"],
                received_at=item["received_at"],
                mailbox_name=item["mailbox_name"],
                account_name=item["account_name"],
                read=self._coerce_bool(item["read"]),
                preview=item["preview"],
            )
            for item in self._parse_records(
                raw,
                (
                    "message_id",
                    "subject",
                    "sender",
                    "received_at",
                    "mailbox_name",
                    "account_name",
                    "read",
                    "preview",
                ),
            )
        ]
        records.sort(
            key=lambda item: (item.received_at, item.subject, item.message_id),
            reverse=True,
        )
        return records[:limit]

    def list_mail_messages(
        self,
        *,
        account_name: str | None,
        mailbox_name: str | None,
        limit: int,
        unread_only: bool,
        days_back: int | None,
    ) -> list[MailMessageRecord]:
        self._validate_limit(limit)
        self._validate_days_back(days_back)

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set accountFilter to item 1 of argv",
                "set mailboxFilter to item 2 of argv",
                "set rowLimit to item 3 of argv as integer",
                "set unreadOnly to false",
                "if item 4 of argv is \"1\" then set unreadOnly to true",
                "set daysBackText to item 5 of argv",
                "set sinceDate to missing value",
                "if daysBackText is not \"\" then",
                "set sinceDate to (current date) - ((daysBackText as integer) * days)",
                "end if",
                "tell application \"Mail\"",
                "if accountFilter is \"\" then",
                "set targetAccounts to accounts",
                "else",
                "set targetAccounts to {account accountFilter}",
                "end if",
                "set rows to {}",
                "repeat with targetAccount in targetAccounts",
                "set accountName to name of targetAccount",
                "if mailboxFilter is \"\" then",
                "try",
                "set targetMailboxes to {mailbox \"INBOX\" of targetAccount}",
                "on error",
                "set targetMailboxes to mailboxes of targetAccount",
                "end try",
                "else",
                "set targetMailboxes to {mailbox mailboxFilter of targetAccount}",
                "end if",
                "repeat with targetMailbox in targetMailboxes",
                "set mailboxName to name of targetMailbox",
                "set targetMessages to messages of targetMailbox",
                "set messageIndex to count of targetMessages",
                "repeat while messageIndex >= 1",
                "set eachMessage to item messageIndex of targetMessages",
                "set readText to \"false\"",
                "try",
                "set readText to read status of eachMessage as string",
                "end try",
                "set receivedDate to missing value",
                "set receivedText to \"\"",
                "try",
                "set receivedDate to date received of eachMessage",
                "set receivedText to my formatDate(receivedDate)",
                "end try",
                "set shouldInclude to true",
                "if sinceDate is not missing value then",
                "if receivedDate is missing value then",
                "set shouldInclude to false",
                "else",
                "if receivedDate < sinceDate then set shouldInclude to false",
                "end if",
                "end if",
                "if unreadOnly is true then",
                "if readText is \"true\" then set shouldInclude to false",
                "end if",
                "if shouldInclude then",
                "set subjectText to \"\"",
                "try",
                "set subjectText to subject of eachMessage",
                "end try",
                "set senderText to \"\"",
                "try",
                "set senderText to sender of eachMessage",
                "end try",
                "set end of rows to my joinFields({id of eachMessage as string, my sanitizeText(subjectText), my sanitizeText(senderText), receivedText, my sanitizeText(mailboxName), my sanitizeText(accountName), readText, \"\"})",
                "end if",
                "set messageIndex to messageIndex - 1",
                "end repeat",
                "end repeat",
                "end repeat",
                "if (count of rows) is 0 then return \"\"",
                "return my joinRecords(rows)",
                "end tell",
                "end run",
            ],
            args=[
                account_name or "",
                mailbox_name or "",
                str(limit),
                "1" if unread_only else "0",
                "" if days_back is None else str(days_back),
            ],
        )
        records = [
            MailMessageRecord(
                message_id=item["message_id"],
                subject=item["subject"],
                sender=item["sender"],
                received_at=item["received_at"],
                mailbox_name=item["mailbox_name"],
                account_name=item["account_name"],
                read=self._coerce_bool(item["read"]),
                preview=item["preview"],
            )
            for item in self._parse_records(
                raw,
                (
                    "message_id",
                    "subject",
                    "sender",
                    "received_at",
                    "mailbox_name",
                    "account_name",
                    "read",
                    "preview",
                ),
            )
        ]
        records.sort(
            key=lambda item: (item.received_at, item.subject, item.message_id),
            reverse=True,
        )
        return records[:limit]

    def get_mail_message(
        self,
        *,
        message_id: str,
        account_name: str | None,
        mailbox_name: str | None,
    ) -> MailMessageBodyRecord:
        normalized_id = message_id.strip()
        if not normalized_id:
            raise AppleScriptBridgeError(
                "invalid_args",
                "message_id 不能为空。",
                data={"field": "message_id"},
            )

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set targetId to item 1 of argv",
                "set accountFilter to item 2 of argv",
                "set mailboxFilter to item 3 of argv",
                "tell application \"Mail\"",
                "if accountFilter is \"\" then",
                "set targetAccounts to accounts",
                "else",
                "set targetAccounts to {account accountFilter}",
                "end if",
                "repeat with targetAccount in targetAccounts",
                "set accountName to name of targetAccount",
                "if mailboxFilter is \"\" then",
                "set targetMailboxes to mailboxes of targetAccount",
                "else",
                "set targetMailboxes to {mailbox mailboxFilter of targetAccount}",
                "end if",
                "repeat with targetMailbox in targetMailboxes",
                "set mailboxName to name of targetMailbox",
                "repeat with eachMessage in messages of targetMailbox",
                "if (id of eachMessage as string) is targetId then",
                "set subjectText to \"\"",
                "try",
                "set subjectText to subject of eachMessage",
                "end try",
                "set senderText to \"\"",
                "try",
                "set senderText to sender of eachMessage",
                "end try",
                "set receivedText to \"\"",
                "try",
                "set receivedText to my formatDate(date received of eachMessage)",
                "end try",
                "set readText to \"false\"",
                "try",
                "set readText to read status of eachMessage as string",
                "end try",
                "set bodyText to \"\"",
                "try",
                "set bodyText to content of eachMessage",
                "end try",
                "return my joinFields({id of eachMessage as string, my sanitizeText(subjectText), my sanitizeText(senderText), receivedText, my sanitizeText(mailboxName), my sanitizeText(accountName), readText, my truncateText(bodyText, 5000)})",
                "end if",
                "end repeat",
                "end repeat",
                "end repeat",
                "error \"Mail message not found\"",
                "end tell",
                "end run",
            ],
            args=[normalized_id, account_name or "", mailbox_name or ""],
        )
        item = self._parse_single_record(
            raw,
            (
                "message_id",
                "subject",
                "sender",
                "received_at",
                "mailbox_name",
                "account_name",
                "read",
                "body",
            ),
        )
        return MailMessageBodyRecord(
            message_id=item["message_id"],
            subject=item["subject"],
            sender=item["sender"],
            received_at=item["received_at"],
            mailbox_name=item["mailbox_name"],
            account_name=item["account_name"],
            read=self._coerce_bool(item["read"]),
            body=item["body"],
        )

    def create_mail_draft(
        self,
        *,
        subject: str,
        body: str,
        to: list[str] | tuple[str, ...] | str | None,
        cc: list[str] | tuple[str, ...] | str | None,
        bcc: list[str] | tuple[str, ...] | str | None,
        sender: str | None,
        visible: bool,
    ) -> MailDraftRecord:
        normalized_subject = self._require_text(subject, field_name="subject")
        normalized_body = self._require_text(body, field_name="body")
        to_items = self._normalize_recipients(to, field_name="to", required=False)
        cc_items = self._normalize_recipients(cc, field_name="cc", required=False)
        bcc_items = self._normalize_recipients(bcc, field_name="bcc", required=False)
        normalized_sender = (sender or "").strip()

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set messageSubject to item 1 of argv",
                "set messageBody to item 2 of argv",
                "set toCsv to item 3 of argv",
                "set ccCsv to item 4 of argv",
                "set bccCsv to item 5 of argv",
                "set senderText to item 6 of argv",
                "set visibleFlag to item 7 of argv",
                "set visibleValue to false",
                "if visibleFlag is \"1\" then set visibleValue to true",
                "tell application \"Mail\"",
                "set newMessage to make new outgoing message with properties {subject:messageSubject, content:messageBody, visible:visibleValue}",
                "if senderText is not \"\" then",
                "try",
                "set sender of newMessage to senderText",
                "on error senderError",
                "error \"Invalid Mail sender: \" & senderError",
                "end try",
                "end if",
                "tell newMessage",
                "repeat with addressText in my splitCommaText(toCsv)",
                "make new to recipient at end of to recipients with properties {address:(addressText as string)}",
                "end repeat",
                "repeat with addressText in my splitCommaText(ccCsv)",
                "make new cc recipient at end of cc recipients with properties {address:(addressText as string)}",
                "end repeat",
                "repeat with addressText in my splitCommaText(bccCsv)",
                "make new bcc recipient at end of bcc recipients with properties {address:(addressText as string)}",
                "end repeat",
                "end tell",
                "return my joinFields({my sanitizeText(messageSubject), toCsv, ccCsv, bccCsv, my sanitizeText(senderText), visibleValue as string, my truncateText(messageBody, 180)})",
                "end tell",
                "end run",
            ],
            args=[
                normalized_subject,
                normalized_body,
                ",".join(to_items),
                ",".join(cc_items),
                ",".join(bcc_items),
                normalized_sender,
                "1" if visible else "0",
            ],
        )
        item = self._parse_single_record(
            raw,
            ("subject", "to", "cc", "bcc", "sender", "visible", "preview"),
        )
        return MailDraftRecord(
            subject=item["subject"],
            to=self._split_recipients_field(item["to"]),
            cc=self._split_recipients_field(item["cc"]),
            bcc=self._split_recipients_field(item["bcc"]),
            sender=item["sender"],
            visible=self._coerce_bool(item["visible"]),
            preview=item["preview"],
        )

    def send_mail_message(
        self,
        *,
        subject: str,
        body: str,
        to: list[str] | tuple[str, ...] | str | None,
        cc: list[str] | tuple[str, ...] | str | None,
        bcc: list[str] | tuple[str, ...] | str | None,
        sender: str | None,
    ) -> MailSendRecord:
        normalized_subject = self._require_text(subject, field_name="subject")
        normalized_body = self._require_text(body, field_name="body")
        to_items = self._normalize_recipients(to, field_name="to", required=True)
        cc_items = self._normalize_recipients(cc, field_name="cc", required=False)
        bcc_items = self._normalize_recipients(bcc, field_name="bcc", required=False)
        normalized_sender = (sender or "").strip()

        raw = self._run_lines(
            [
                *_COMMON_HELPERS,
                "on run argv",
                "set messageSubject to item 1 of argv",
                "set messageBody to item 2 of argv",
                "set toCsv to item 3 of argv",
                "set ccCsv to item 4 of argv",
                "set bccCsv to item 5 of argv",
                "set senderText to item 6 of argv",
                "tell application \"Mail\"",
                "set newMessage to make new outgoing message with properties {subject:messageSubject, content:messageBody, visible:false}",
                "if senderText is not \"\" then",
                "try",
                "set sender of newMessage to senderText",
                "on error senderError",
                "error \"Invalid Mail sender: \" & senderError",
                "end try",
                "end if",
                "tell newMessage",
                "repeat with addressText in my splitCommaText(toCsv)",
                "make new to recipient at end of to recipients with properties {address:(addressText as string)}",
                "end repeat",
                "repeat with addressText in my splitCommaText(ccCsv)",
                "make new cc recipient at end of cc recipients with properties {address:(addressText as string)}",
                "end repeat",
                "repeat with addressText in my splitCommaText(bccCsv)",
                "make new bcc recipient at end of bcc recipients with properties {address:(addressText as string)}",
                "end repeat",
                "send",
                "end tell",
                "return my joinFields({my sanitizeText(messageSubject), toCsv, ccCsv, bccCsv, my sanitizeText(senderText), \"true\", my truncateText(messageBody, 180)})",
                "end tell",
                "end run",
            ],
            args=[
                normalized_subject,
                normalized_body,
                ",".join(to_items),
                ",".join(cc_items),
                ",".join(bcc_items),
                normalized_sender,
            ],
        )
        item = self._parse_single_record(
            raw,
            ("subject", "to", "cc", "bcc", "sender", "sent", "preview"),
        )
        return MailSendRecord(
            subject=item["subject"],
            to=self._split_recipients_field(item["to"]),
            cc=self._split_recipients_field(item["cc"]),
            bcc=self._split_recipients_field(item["bcc"]),
            sender=item["sender"],
            sent=self._coerce_bool(item["sent"]),
            preview=item["preview"],
        )

    def _run_lines(self, lines: list[str], *, args: list[str]) -> str:
        command = [self.osascript_path, "-l", "AppleScript"]
        for line in lines:
            command.extend(["-e", line])
        if args:
            command.append("--")
            command.extend(args)

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppleScriptBridgeError(
                "timeout",
                f"AppleScript 执行超过 {self.timeout_seconds} 秒，已终止。",
            ) from exc

        stdout = (completed.stdout or "").rstrip("\r\n")
        if completed.returncode == 0:
            return stdout

        detail = (completed.stderr or stdout or "AppleScript 执行失败。").strip()
        raise AppleScriptBridgeError(
            self._map_error_code(detail),
            detail,
        )

    @staticmethod
    def _parse_local_datetime(value: str, *, field_name: str) -> datetime:
        text = value.strip()
        if not text:
            raise AppleScriptBridgeError(
                "invalid_args",
                f"{field_name} 不能为空。",
                data={"field": field_name},
            )
        if "T" not in text and " " not in text:
            raise AppleScriptBridgeError(
                "invalid_args",
                f"{field_name} 必须是本地日期时间，例如 2026-04-18T10:30 或 2026-04-18 10:30。",
                data={"field": field_name, "value": value},
            )
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise AppleScriptBridgeError(
                "invalid_args",
                f"{field_name} 格式无效: {value}",
                data={"field": field_name, "value": value},
            ) from exc
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt.replace(microsecond=0)

    @staticmethod
    def _ensure_range(
        start_dt: datetime,
        end_dt: datetime,
        *,
        start_field: str,
        end_field: str,
    ) -> None:
        if end_dt <= start_dt:
            raise AppleScriptBridgeError(
                "invalid_args",
                f"{end_field} 必须晚于 {start_field}。",
                data={start_field: start_dt.isoformat(), end_field: end_dt.isoformat()},
            )

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit <= 0 or limit > 25:
            raise AppleScriptBridgeError(
                "invalid_args",
                "limit 必须在 1 到 25 之间。",
                data={"field": "limit", "value": limit},
            )

    @staticmethod
    def _validate_days_back(days_back: int | None) -> None:
        if days_back is None:
            return
        if days_back <= 0 or days_back > 365:
            raise AppleScriptBridgeError(
                "invalid_args",
                "days_back 必须在 1 到 365 之间，或传 null 表示不限制日期。",
                data={"field": "days_back", "value": days_back},
            )

    @staticmethod
    def _require_text(value: str, *, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise AppleScriptBridgeError(
                "invalid_args",
                f"{field_name} 不能为空。",
                data={"field": field_name},
            )
        return normalized

    @staticmethod
    def _normalize_recipients(
        value: list[str] | tuple[str, ...] | str | None,
        *,
        field_name: str,
        required: bool,
    ) -> list[str]:
        if value is None:
            raw_items: list[str] = []
        elif isinstance(value, str):
            raw_items = [value]
        else:
            raw_items = [str(item) for item in value]

        recipients: list[str] = []
        for raw_item in raw_items:
            for part in raw_item.split(","):
                normalized = part.strip()
                if normalized:
                    recipients.append(normalized)

        if required and not recipients:
            raise AppleScriptBridgeError(
                "invalid_args",
                f"{field_name} 至少需要一个收件人地址。",
                data={"field": field_name},
            )
        return recipients

    @staticmethod
    def _split_recipients_field(value: str) -> list[str]:
        return [part for part in value.split(",") if part]

    @staticmethod
    def _coerce_int(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0

    @staticmethod
    def _coerce_bool(value: str) -> bool:
        return value.strip().lower() in {"true", "1", "yes"}

    @staticmethod
    def _datetime_components(dt: datetime) -> list[str]:
        return [
            str(dt.year),
            str(dt.month),
            str(dt.day),
            str(dt.hour),
            str(dt.minute),
            str(dt.second),
        ]

    @staticmethod
    def _parse_records(raw: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
        if not raw:
            return []
        rows: list[dict[str, str]] = []
        for record in raw.split(_RECORD_SEP):
            if not record:
                continue
            values = record.split(_FIELD_SEP)
            if len(values) != len(fields):
                raise AppleScriptBridgeError(
                    "error",
                    "AppleScript 返回结构无法解析。",
                    data={"expected_fields": fields, "raw": raw},
                )
            rows.append(dict(zip(fields, values, strict=True)))
        return rows

    def _parse_single_record(self, raw: str, fields: tuple[str, ...]) -> dict[str, str]:
        rows = self._parse_records(raw, fields)
        if not rows:
            raise AppleScriptBridgeError("error", "AppleScript 未返回结果。")
        return rows[0]

    @staticmethod
    def _plain_text_to_notes_html(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return "<div><br></div>"
        fragments: list[str] = []
        for line in lines:
            if not line.strip():
                fragments.append("<div><br></div>")
            else:
                fragments.append(f"<div>{html.escape(line)}</div>")
        return "".join(fragments)

    @staticmethod
    def _map_error_code(detail: str) -> str:
        normalized = detail.lower()
        if "-1743" in detail or "not authorized to send apple events" in normalized:
            return "permission_denied"
        if "read-only" in normalized or "access not allowed" in normalized:
            return "permission_denied"
        if (
            "-1728" in detail
            or "can’t get" in normalized
            or "can't get" in normalized
            or "not found" in normalized
        ):
            return "not_found"
        if (
            "-1700" in detail
            or "-1703" in detail
            or "格式无效" in normalized
            or "invalid mail sender" in normalized
        ):
            return "invalid_args"
        return "error"
