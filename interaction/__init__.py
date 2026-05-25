"""UI-independent interaction helpers for MiniBot."""

from .commands import (
    CommandContext,
    CommandNotice,
    CommandResult,
    dispatch_command,
)

__all__ = [
    "CommandContext",
    "CommandNotice",
    "CommandResult",
    "dispatch_command",
]
