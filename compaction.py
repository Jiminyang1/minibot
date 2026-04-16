"""Session compaction strategy for MiniBot.

Handles both the compaction decision logic and the LLM-backed
summarisation that was previously coupled into the agent runner.
Compaction is triggered based on estimated token count, not message count.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .llm import LLMClient
from .prompts import SUMMARY_SYSTEM_PROMPT
from .session import Session, SessionManager


# ── token estimation ─────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken if available, else heuristic."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except (ImportError, KeyError):
        # Heuristic: ~2 chars per token for mixed Chinese/English
        return max(1, len(text) // 2)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of OpenAI-format messages."""
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        for value in msg.values():
            if isinstance(value, str):
                total += _estimate_tokens(value)
            elif isinstance(value, list):
                total += _estimate_tokens(json.dumps(value, ensure_ascii=False))
    total += 2  # reply priming
    return total


# ── summarisation ────────────────────────────────────────────────


def _format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Flatten a message list into a readable transcript for the summariser."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "assistant")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
        if tool_calls := message.get("tool_calls"):
            for call in tool_calls:
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                name = fn.get("name", "unknown_tool")
                args = fn.get("arguments", "{}")
                lines.append(f"ASSISTANT_TOOL_CALL: {name}({args})")
        if message.get("role") == "tool":
            name = message.get("name", "tool")
            lines.append(f"TOOL_RESULT[{name}]: {content}")
    return "\n".join(lines)


def make_summarizer(llm: LLMClient) -> Callable[[list[dict[str, Any]]], str]:
    """Create a summariser closure backed by the given LLM client."""

    def summarize(messages: list[dict[str, Any]]) -> str:
        if not messages:
            raise ValueError("没有可供摘要的历史消息。")
        formatted = _format_messages_for_summary(messages)
        resp = llm.chat([
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": formatted},
        ])
        summary = (resp.content or "").strip()
        if not summary:
            raise RuntimeError("模型没有返回有效摘要。")
        return summary

    return summarize


# ── compaction logic ─────────────────────────────────────────────


def maybe_compact(
    session: Session,
    manager: SessionManager,
    summarizer: Callable[[list[dict[str, Any]]], str],
    *,
    token_threshold: int,
    keep_recent: int,
) -> tuple[bool, str]:
    """Compact a session when its estimated token count exceeds *token_threshold*.

    Returns ``(did_compact, message)`` so the caller decides how to display it.
    """
    all_dicts = [m.to_model_message() for m in session.messages]
    current_tokens = estimate_message_tokens(all_dicts)

    if current_tokens <= token_threshold:
        return False, (
            f"当前会话约 {current_tokens} tokens，"
            f"未超过阈值 {token_threshold}，无需压缩。"
        )

    old_messages = session.messages_to_compact(keep_recent)
    if not old_messages:
        return False, "当前会话没有可压缩的旧轮次。"

    summary = summarizer([m.to_model_message() for m in old_messages])
    before, after = session.compact_with_summary(summary, keep_recent)
    manager.save(session)

    after_dicts = [m.to_model_message() for m in session.messages]
    after_tokens = estimate_message_tokens(after_dicts)
    return True, (
        f"已压缩: {before} -> {after} 条消息, "
        f"{current_tokens} -> {after_tokens} tokens"
    )
