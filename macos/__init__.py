"""macOS-specific integrations."""

from .bridge import (
    AppleScriptBridge,
    AppleScriptBridgeError,
    CalendarEventRecord,
    NoteRecord,
    ReminderRecord,
)

__all__ = [
    "AppleScriptBridge",
    "AppleScriptBridgeError",
    "CalendarEventRecord",
    "NoteRecord",
    "ReminderRecord",
]
