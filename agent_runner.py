"""Agent run spec and execution loop for MiniBot."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, LLMResponse
from .session import MessageEvent
from .tools import ToolRegistry


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


def _latest_user_input(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


@dataclass(frozen=True)
class RunSpec:
    """One concrete agent execution prepared by the turn engine."""

    model: str
    max_iterations: int = 20
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner:
    """Execute the tool-calling LLM loop for one prepared request."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        event_handler: Callable[[str], None] | None = None,
        approval_handler: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        self.llm = llm
        self.event_handler = event_handler
        self.approval_handler = approval_handler

    def run(
        self,
        run_spec: RunSpec,
        tool_registry: ToolRegistry,
    ) -> tuple[str, list[MessageEvent]]:
        """Run one prepared request until the model returns a final answer."""
        messages = list(run_spec.messages)
        self._emit(f"开始处理: {_preview(_latest_user_input(messages))}")

        events: list[MessageEvent] = []
        for iteration in range(1, run_spec.max_iterations + 1):
            started = time.perf_counter()
            self._emit(f"第 {iteration} 轮: 请求模型...")

            resp = self.llm.chat(
                messages,
                run_spec.tool_definitions,
                model=run_spec.model,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            assistant_msg = self._response_to_message(resp)
            messages.append(assistant_msg)
            events.append(_to_event(assistant_msg))

            if not resp.tool_calls:
                self._emit(f"第 {iteration} 轮: 最终回答 ({elapsed_ms} ms)")
                return resp.content or "", events

            self._emit(
                f"第 {iteration} 轮: 调用 {len(resp.tool_calls)} 个工具 ({elapsed_ms} ms)"
            )
            for tc in resp.tool_calls:
                args = _parse_args(tc.arguments)
                self._emit(f"工具: {tc.name}({args})")

                tool = tool_registry.get(tc.name)
                if tool and tool.requires_approval and not self._approve(tc.name, args):
                    result = f"[用户拒绝] 工具 {tc.name} 未获批准执行。"
                else:
                    result = tool_registry.execute(tc.name, args)
                self._emit(f"返回: {_preview(result, 100)}")

                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": str(result),
                }
                messages.append(tool_msg)
                events.append(_to_event(tool_msg))

        fallback = "抱歉，工具调用轮次已达上限，请简化问题后重试。"
        self._emit(f"已达最大迭代次数 {run_spec.max_iterations}")
        events.append(_to_event({"role": "assistant", "content": fallback}))
        return fallback, events

    @staticmethod
    def _response_to_message(resp: LLMResponse) -> dict[str, Any]:
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

    def _approve(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self.approval_handler is None:
            return True
        return self.approval_handler(tool_name, args)

    def _emit(self, message: str) -> None:
        if self.event_handler:
            self.event_handler(message)


def _to_event(message: dict[str, Any]) -> MessageEvent:
    """Convert a raw message dict to a MessageEvent for session storage."""
    return MessageEvent.create(
        role=str(message["role"]),
        content=str(message.get("content", "")),
        tool_calls=message.get("tool_calls"),
        tool_call_id=message.get("tool_call_id"),
        name=message.get("name"),
    )
