"""Prompt assembly for MiniBot."""

from __future__ import annotations

from typing import Any


def build_messages(
    *,
    system_prompt: str,
    history: list[dict[str, Any]],
    user_input: str | None = None,
) -> list[dict[str, Any]]:
    """Build the concrete messages payload for a model request."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *history,
    ]
    if user_input is not None:
        messages.append({"role": "user", "content": user_input})
    return messages
