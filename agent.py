"""Agent Core for MiniBot — identity, tools, and execution.

The Agent is the single "protagonist": it knows who it is (system prompt),
what it can do (tools), and how to execute (tool-calling loop).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .llm import LLMClient, LLMResponse
from .prompts import SYSTEM_PROMPT
from .session import MessageEvent
from .tools import TOOL_DEFINITIONS, execute_tool


# ── helpers ──────────────────────────────────────────────────────


def _preview(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ── Agent Core ───────────────────────────────────────────────────


class Agent:
    """Core agent: identity + tools + tool-calling execution loop."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], str] = execute_tool,
        max_iterations: int = 20,
        event_handler: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else TOOL_DEFINITIONS
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.event_handler = event_handler

    # ── public API ───────────────────────────────────────────────

    def run(
        self, history: list[dict[str, Any]], user_input: str
    ) -> tuple[str, list[MessageEvent]]:
        """Build context, run tool-call loop, return (reply, events)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_input},
        ]
        self._emit(f"开始处理: {_preview(user_input)}")

        events: list[MessageEvent] = []

        for iteration in range(1, self.max_iterations + 1):
            started = time.perf_counter()
            self._emit(f"第 {iteration} 轮: 请求模型...")

            resp = self.llm.chat(messages, self.tools)
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            # Record the assistant message
            assistant_msg = self._response_to_message(resp)
            messages.append(assistant_msg)
            events.append(_to_event(assistant_msg))

            # No tool calls → final answer
            if not resp.tool_calls:
                self._emit(f"第 {iteration} 轮: 最终回答 ({elapsed_ms} ms)")
                return resp.content or "", events

            self._emit(
                f"第 {iteration} 轮: 调用 {len(resp.tool_calls)} 个工具 ({elapsed_ms} ms)"
            )

            # Execute each tool call
            for tc in resp.tool_calls:
                args = _parse_args(tc.arguments)
                self._emit(f"工具: {tc.name}({args})")
                result = self.tool_executor(tc.name, args)
                self._emit(f"返回: {_preview(result, 100)}")

                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": str(result),
                }
                messages.append(tool_msg)
                events.append(_to_event(tool_msg))

        # Safety: max iterations reached
        fallback = "抱歉，工具调用轮次已达上限，请简化问题后重试。"
        self._emit(f"已达最大迭代次数 {self.max_iterations}")
        events.append(_to_event({"role": "assistant", "content": fallback}))
        return fallback, events

    # ── internals ────────────────────────────────────────────────

    @staticmethod
    def _response_to_message(resp: LLMResponse) -> dict[str, Any]:
        """Convert an LLMResponse to an OpenAI-format message dict."""
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
        if resp.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in resp.tool_calls
            ]
        return msg

    def _emit(self, message: str) -> None:
        if self.event_handler:
            self.event_handler(message)


# ── shared utility ───────────────────────────────────────────────


def _to_event(message: dict[str, Any]) -> MessageEvent:
    """Convert a raw message dict to a MessageEvent for session storage."""
    return MessageEvent.create(
        role=str(message["role"]),
        content=str(message.get("content", "")),
        tool_calls=message.get("tool_calls"),
        tool_call_id=message.get("tool_call_id"),
        name=message.get("name"),
    )
