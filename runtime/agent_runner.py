"""Agent run spec and execution loop for MiniBot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from ..session import MessageEvent
from ..tools import ToolExecutionContext, ToolRegistry
from ..tools.base import Tool
from ..tools.registry import PreparedToolCall
from ..tools.result import ToolOutput
from .tool_output_materializer import ToolOutputMaterializer


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


def _tool_label(tool: Tool | None, fallback: str) -> str:
    if tool is None:
        return fallback
    return tool.display_name


def _is_mcp_tool(tool: Tool | None) -> bool:
    return tool is not None and tool.source == "mcp"


@dataclass(frozen=True)
class RunSpec:
    """One concrete agent execution prepared by the turn engine."""

    session_id: str
    model: str
    messages: list[dict[str, Any]]
    tool_definitions: list[dict[str, Any]]
    max_iterations: int = 20


@dataclass(frozen=True)
class RunOutcome:
    """Full outcome of one prepared agent run."""

    reply: str
    events: list[MessageEvent]
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class PartialRunError(Exception):
    """Internal carrier for failures after partial agent progress."""

    cause: Exception
    events: list[MessageEvent]
    usage: TokenUsage | None = None
    reply: str | None = None


@dataclass(frozen=True)
class _PlannedToolCall:
    """One model-requested tool call annotated with local execution metadata."""

    tool_call: ToolCall
    args: dict[str, Any]
    tool: Tool | None


class AgentRunner:
    """Execute the tool-calling LLM loop for one prepared request."""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        *,
        materializer: ToolOutputMaterializer,
        event_handler: Callable[[str], None] | None = None,
        approval_handler: Callable[[str, dict[str, Any]], bool] | None = None,
        max_parallel_tools: int = 4,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.materializer = materializer
        self.event_handler = event_handler
        self.approval_handler = approval_handler
        self.max_parallel_tools = max_parallel_tools

    def run(self, run_spec: RunSpec) -> RunOutcome:
        """Run one prepared request until the model returns a final answer."""
        messages = list(run_spec.messages)
        tool_context = ToolExecutionContext(session_id=run_spec.session_id)
        self._emit(f"开始处理: {_preview(_latest_user_input(messages))}")

        events: list[MessageEvent] = []
        usage: TokenUsage | None = None
        for iteration in range(1, run_spec.max_iterations + 1):
            started = time.perf_counter()
            self._emit(f"第 {iteration} 轮: 请求模型...")

            try:
                resp = self.llm.chat(
                    messages,
                    run_spec.tool_definitions,
                    model=run_spec.model,
                )
            except Exception as exc:
                if events:
                    raise PartialRunError(
                        cause=exc,
                        events=list(events),
                        usage=usage,
                    ) from exc
                raise
            usage = _merge_usage(usage, resp.usage)
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            assistant_msg = self._response_to_message(resp)
            messages.append(assistant_msg)
            events.append(_to_event(assistant_msg))

            if not resp.tool_calls:
                self._emit(f"第 {iteration} 轮: 最终回答 ({elapsed_ms} ms)")
                return RunOutcome(reply=resp.content or "", events=events, usage=usage)

            self._emit(
                f"第 {iteration} 轮: 调用 {len(resp.tool_calls)} 个工具 ({elapsed_ms} ms)"
            )
            planned_tool_calls = self._plan_tool_calls(resp.tool_calls)
            tool_outputs = self._execute_tool_calls(planned_tool_calls, tool_context)

            for planned, output in zip(planned_tool_calls, tool_outputs):
                result = self.materializer.materialize(output, context=tool_context)
                if _is_mcp_tool(planned.tool):
                    self._emit(f"MCP 返回: {result.summary}")
                else:
                    self._emit(f"返回: {result.summary}")

                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": planned.tool_call.id,
                    "name": planned.tool_call.name,
                    "content": result.to_model_content(),
                }
                messages.append(tool_msg)
                events.append(_to_event(tool_msg))

        fallback = "抱歉，工具调用轮次已达上限，请简化问题后重试。"
        self._emit(f"已达最大迭代次数 {run_spec.max_iterations}")
        events.append(_to_event({"role": "assistant", "content": fallback}))
        return RunOutcome(reply=fallback, events=events, usage=usage)

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

    def _plan_tool_calls(self, tool_calls: list[ToolCall]) -> list[_PlannedToolCall]:
        planned: list[_PlannedToolCall] = []
        for tool_call in tool_calls:
            args = _parse_args(tool_call.arguments)
            tool = self.tool_registry.get(tool_call.name)
            if _is_mcp_tool(tool):
                self._emit(f"MCP 调用: {_tool_label(tool, tool_call.name)}({args})")
            else:
                self._emit(f"工具: {tool_call.name}({args})")
            planned.append(
                _PlannedToolCall(
                    tool_call=tool_call,
                    args=args,
                    tool=tool,
                )
            )
        return planned

    def _execute_tool_calls(
        self,
        planned_tool_calls: list[_PlannedToolCall],
        context: ToolExecutionContext,
    ) -> list[ToolOutput]:
        outputs: list[ToolOutput] = []
        for batch in self._partition_tool_batches(planned_tool_calls):
            if len(batch) > 1:
                outputs.extend(self._execute_parallel_batch(batch, context))
                continue
            outputs.append(self._execute_planned_tool_call(batch[0], context))
        return outputs

    def _partition_tool_batches(
        self,
        planned_tool_calls: list[_PlannedToolCall],
    ) -> list[list[_PlannedToolCall]]:
        if self.max_parallel_tools <= 1:
            return [[planned_tool_call] for planned_tool_call in planned_tool_calls]

        batches: list[list[_PlannedToolCall]] = []
        current_batch: list[_PlannedToolCall] = []
        for planned_tool_call in planned_tool_calls:
            if planned_tool_call.tool and planned_tool_call.tool.concurrency_safe:
                current_batch.append(planned_tool_call)
                continue
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append([planned_tool_call])
        if current_batch:
            batches.append(current_batch)
        return batches

    def _execute_parallel_batch(
        self,
        batch: list[_PlannedToolCall],
        context: ToolExecutionContext,
    ) -> list[ToolOutput]:
        outputs: list[ToolOutput | None] = [None] * len(batch)
        submitted: list[tuple[int, _PlannedToolCall, Any]] = []

        for index, planned in enumerate(batch):
            prepared_or_output = self._prepare_tool_call(planned, context)
            if isinstance(prepared_or_output, ToolOutput):
                outputs[index] = prepared_or_output
                continue
            submitted.append((index, planned, prepared_or_output))

        if submitted:
            max_workers = min(self.max_parallel_tools, len(submitted))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    (index, planned, executor.submit(self.tool_registry.invoke, prepared))
                    for index, planned, prepared in submitted
                ]
                for index, planned, future in futures:
                    try:
                        outputs[index] = future.result()
                    except Exception as exc:
                        outputs[index] = ToolOutput.failure(
                            "error",
                            f"工具 {planned.tool_call.name} 执行失败: {exc}",
                            data={"tool": planned.tool_call.name},
                            meta={"exception": repr(exc)},
                        )

        return [
            output
            if output is not None
            else ToolOutput.failure(
                "error",
                f"工具 {planned.tool_call.name} 未返回结果。",
                data={"tool": planned.tool_call.name},
            )
            for planned, output in zip(batch, outputs)
        ]

    def _execute_planned_tool_call(
        self,
        planned: _PlannedToolCall,
        context: ToolExecutionContext,
    ) -> ToolOutput:
        prepared_or_output = self._prepare_tool_call(planned, context)
        if isinstance(prepared_or_output, ToolOutput):
            return prepared_or_output
        return self.tool_registry.invoke(prepared_or_output)

    def _prepare_tool_call(
        self,
        planned: _PlannedToolCall,
        context: ToolExecutionContext,
    ) -> PreparedToolCall | ToolOutput:
        if planned.tool and planned.tool.requires_approval:
            if not self._approve(planned.tool_call.name, planned.args):
                return ToolOutput.failure(
                    "denied",
                    f"工具 {planned.tool_call.name} 未获批准执行。",
                    data={"tool": planned.tool_call.name, "args": planned.args},
                )
        return self.tool_registry.prepare(
            planned.tool_call.name,
            planned.args,
            context=context,
        )

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


def _merge_usage(
    accumulated: TokenUsage | None,
    current: TokenUsage | None,
) -> TokenUsage | None:
    if accumulated is None:
        return current
    if current is None:
        return accumulated
    return TokenUsage(
        input_tokens=_sum_optional(accumulated.input_tokens, current.input_tokens),
        output_tokens=_sum_optional(accumulated.output_tokens, current.output_tokens),
        total_tokens=_sum_optional(accumulated.total_tokens, current.total_tokens),
    )


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left + right
