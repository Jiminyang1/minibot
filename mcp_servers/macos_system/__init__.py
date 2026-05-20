"""Self-contained macOS MCP service implementation."""

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

__all__ = [
    "AppleScriptBridge",
    "AppleScriptBridgeError",
    "CalendarEventRecord",
    "MailDraftRecord",
    "MailMessageBodyRecord",
    "MailMessageRecord",
    "MailSendRecord",
    "MailboxRecord",
    "NoteRecord",
    "ReminderRecord",
]
