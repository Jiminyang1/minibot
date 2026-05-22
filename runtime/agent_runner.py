"""Agent run spec and execution loop for MiniBot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
import uuid


class RunCancelled(RuntimeError):
    """Raised when a run is cooperatively cancelled."""

from ..llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from ..tools.base import Tool, ToolExecutionContext
from ..tools.definitions import ModelToolDefinition
from ..tools.registry import PreparedToolCall, ToolRegistry
from ..tools.result import ToolOutput
from .events import RuntimeEventEmitter, RuntimeEventHandler
from .hooks import (
    HookContext,
    ModelRequest,
    RuntimeHookManager,
    ToolPrepareRequest,
)
from .messages import AgentMessage, ModelMessage, ModelToolCall
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


def _latest_user_input(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _normalized_reply(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _tool_label(tool: Tool | None, fallback: str) -> str:
    if tool is None:
        return fallback
    return tool.display_name


def _make_local_run_id() -> str:
    now = datetime.now(UTC)
    return f"r_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


@dataclass(frozen=True)
class RunSpec:
    """One concrete agent execution prepared by the turn engine."""

    session_id: str
    model: str
    messages: list[ModelMessage]
    tool_definitions: list[ModelToolDefinition]
    max_iterations: int = 20
    run_id: str | None = None
    event_emitter: RuntimeEventEmitter | None = None
    cancel_event: threading.Event | None = None
    mode: str = "default"
    workspace: Path | None = None


@dataclass(frozen=True, init=False)
class RunOutcome:
    """Full outcome of one prepared agent run."""

    reply: str
    messages: list[AgentMessage]
    usage: TokenUsage | None = None

    def __init__(
        self,
        *,
        reply: str,
        messages: list[AgentMessage] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        object.__setattr__(self, "reply", reply)
        object.__setattr__(
            self,
            "messages",
            _coerce_agent_messages(messages or []),
        )
        object.__setattr__(self, "usage", usage)


@dataclass(frozen=True, init=False)
class PartialRunError(Exception):
    """Internal carrier for failures after partial agent progress."""

    cause: Exception
    messages: list[AgentMessage]
    usage: TokenUsage | None = None
    reply: str | None = None

    def __init__(
        self,
        *,
        cause: Exception,
        messages: list[AgentMessage] | None = None,
        usage: TokenUsage | None = None,
        reply: str | None = None,
    ) -> None:
        super().__init__(str(cause))
        object.__setattr__(self, "cause", cause)
        object.__setattr__(
            self,
            "messages",
            _coerce_agent_messages(messages or []),
        )
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "reply", reply)


@dataclass(frozen=True)
class _PlannedToolCall:
    """One model-requested tool call annotated with local execution metadata."""

    tool_call: ToolCall
    args: dict[str, Any]
    tool: Tool | None


def _coerce_agent_messages(messages: list[Any]) -> list[AgentMessage]:
    return [_coerce_agent_message(message) for message in messages]


def _coerce_agent_message(message: Any) -> AgentMessage:
    if isinstance(message, AgentMessage):
        return message
    if isinstance(message, ModelMessage):
        return _to_agent_message(message)
    if isinstance(message, dict):
        return _to_agent_message(message)
    return AgentMessage.create(
        role=str(getattr(message, "role")),
        content=str(getattr(message, "content", "")),
        tool_calls=getattr(message, "tool_calls", None),
        tool_call_id=getattr(message, "tool_call_id", None),
        name=getattr(message, "name", None),
        reasoning_content=getattr(message, "reasoning_content", None),
    )


class AgentRunner:
    """Execute the tool-calling LLM loop for one prepared request."""

    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        *,
        materializer: ToolOutputMaterializer,
        hook_manager: RuntimeHookManager | None = None,
        event_handler: RuntimeEventHandler | None = None,
        max_parallel_tools: int = 4,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.materializer = materializer
        self.hook_manager = hook_manager or RuntimeHookManager()
        self.event_handler = event_handler
        self.max_parallel_tools = max_parallel_tools

    def run(self, run_spec: RunSpec) -> RunOutcome:
        """Run one prepared request until the model returns a final answer."""
        messages = list(run_spec.messages)
        tool_context = ToolExecutionContext(session_id=run_spec.session_id)
        emitter = self._resolve_emitter(run_spec)
        hook_context = self._hook_context(run_spec, emitter)

        messages_out: list[AgentMessage] = []
        usage: TokenUsage | None = None

        def _check_cancel() -> None:
            if (
                run_spec.cancel_event is not None
                and run_spec.cancel_event.is_set()
            ):
                raise RunCancelled("run cancelled by user")

        for iteration in range(1, run_spec.max_iterations + 1):
            _check_cancel()
            started = time.perf_counter()
            self._emit(
                emitter,
                "model.request.started",
                {
                    "iteration": iteration,
                    "model": run_spec.model,
                    "input_preview": _preview(_latest_user_input(messages)),
                },
            )

            try:
                model_request = self.hook_manager.before_model_request(
                    hook_context,
                    ModelRequest(
                        model=run_spec.model,
                        messages=messages,
                        tool_definitions=run_spec.tool_definitions,
                    ),
                )
                resp = self.llm.chat(
                    model_request.messages,
                    model_request.tool_definitions,
                    model=model_request.model,
                )
                resp = self.hook_manager.after_model_response(
                    hook_context,
                    model_request,
                    resp,
                )
            except Exception as exc:
                if messages_out:
                    raise PartialRunError(
                        cause=exc,
                        messages=list(messages_out),
                        usage=usage,
                    ) from exc
                raise
            usage = _merge_usage(usage, resp.usage)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            model_completed_payload: dict[str, Any] = {
                "iteration": iteration,
                "elapsed_ms": elapsed_ms,
                "tool_call_count": len(resp.tool_calls),
                "usage": _usage_payload(resp.usage),
            }

            if not resp.tool_calls:
                reply = _normalized_reply(resp.content).strip()
                if reply:
                    self._emit(
                        emitter,
                        "model.request.completed",
                        model_completed_payload,
                    )
                    assistant_msg = ModelMessage.create(
                        role="assistant",
                        content=reply,
                        reasoning_content=resp.reasoning_content,
                    )
                    messages.append(assistant_msg)
                    messages_out.append(_to_agent_message(assistant_msg))
                    self._emit(
                        emitter,
                        "message.completed",
                        {"iteration": iteration, "content": reply},
                    )
                    return RunOutcome(
                        reply=reply,
                        messages=messages_out,
                        usage=usage,
                    )

                model_completed_payload.update(
                    {
                        "empty_reply": True,
                        "usage": _usage_payload(usage),
                        "response_debug": resp.debug,
                    }
                )
                self._emit(
                    emitter,
                    "model.request.completed",
                    model_completed_payload,
                )
                raise PartialRunError(
                    cause=RuntimeError("模型返回空回复，请重试；详见上一条空回答诊断日志。"),
                    messages=list(messages_out),
                    usage=usage,
                )

            self._emit(
                emitter,
                "model.request.completed",
                model_completed_payload,
            )
            assistant_msg = self._response_to_model_message(resp)
            messages.append(assistant_msg)
            messages_out.append(_to_agent_message(assistant_msg))

            planned_tool_calls = self._plan_tool_calls(resp.tool_calls, emitter)
            _check_cancel()
            tool_outputs = self._execute_tool_calls(
                planned_tool_calls,
                tool_context,
                hook_context,
                emitter,
            )

            for planned, output in zip(planned_tool_calls, tool_outputs):
                result = self.materializer.materialize(output, context=tool_context)
                self._emit(
                    emitter,
                    "tool_call.completed" if result.ok else "tool_call.failed",
                    {
                        "tool_call_id": planned.tool_call.id,
                        "tool": planned.tool_call.name,
                        "display_name": _tool_label(planned.tool, planned.tool_call.name),
                        "source": None if planned.tool is None else planned.tool.source,
                        "ok": result.ok,
                        "code": result.code,
                        "summary": result.summary,
                        "artifact": (
                            None if result.artifact is None else result.artifact.to_dict()
                        ),
                        "truncated": result.truncated,
                    },
                )

                tool_msg = ModelMessage.create(
                    role="tool",
                    tool_call_id=planned.tool_call.id,
                    tool_name=planned.tool_call.name,
                    content=result.to_model_content(),
                )
                messages.append(tool_msg)
                messages_out.append(_to_agent_message(tool_msg))

        fallback = "抱歉，工具调用轮次已达上限，请简化问题后重试。"
        self._emit(
            emitter,
            "message.completed",
            {
                "content": fallback,
                "reason": "max_iterations",
                "max_iterations": run_spec.max_iterations,
            },
        )
        messages_out.append(
            AgentMessage.create(role="assistant", content=fallback)
        )
        return RunOutcome(reply=fallback, messages=messages_out, usage=usage)

    @staticmethod
    def _response_to_model_message(resp: LLMResponse) -> ModelMessage:
        return ModelMessage.create(
            role="assistant",
            content=resp.content or "",
            reasoning_content=resp.reasoning_content,
            tool_calls=[
                ModelToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in resp.tool_calls
            ] or None,
        )

    def _plan_tool_calls(
        self,
        tool_calls: list[ToolCall],
        emitter: RuntimeEventEmitter | None,
    ) -> list[_PlannedToolCall]:
        planned: list[_PlannedToolCall] = []
        for tool_call in tool_calls:
            args = _parse_args(tool_call.arguments)
            tool = self.tool_registry.get(tool_call.name)
            self._emit(
                emitter,
                "tool_call.started",
                {
                    "tool_call_id": tool_call.id,
                    "tool": tool_call.name,
                    "display_name": _tool_label(tool, tool_call.name),
                    "source": None if tool is None else tool.source,
                    "args": args,
                    "requires_approval": bool(tool and tool.requires_approval),
                },
            )
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
        hook_context: HookContext,
        emitter: RuntimeEventEmitter | None,
    ) -> list[ToolOutput]:
        outputs: list[ToolOutput] = []
        for batch in self._partition_tool_batches(planned_tool_calls):
            if len(batch) > 1:
                outputs.extend(
                    self._execute_parallel_batch(batch, context, hook_context, emitter)
                )
                continue
            outputs.append(
                self._execute_planned_tool_call(batch[0], context, hook_context, emitter)
            )
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
        hook_context: HookContext,
        emitter: RuntimeEventEmitter | None,
    ) -> list[ToolOutput]:
        outputs: list[ToolOutput | None] = [None] * len(batch)
        submitted: list[tuple[int, _PlannedToolCall, Any]] = []

        for index, planned in enumerate(batch):
            prepared_or_output = self._prepare_tool_call(
                planned,
                context,
                hook_context,
            )
            if isinstance(prepared_or_output, ToolOutput):
                outputs[index] = prepared_or_output
                continue
            submitted.append((index, planned, prepared_or_output))

        if submitted:
            max_workers = min(self.max_parallel_tools, len(submitted))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    (
                        index,
                        planned,
                        prepared,
                        executor.submit(self.tool_registry.invoke, prepared),
                    )
                    for index, planned, prepared in submitted
                ]
                for index, planned, prepared, future in futures:
                    try:
                        raw_output = future.result()
                    except Exception as exc:
                        raw_output = ToolOutput.failure(
                            "error",
                            f"工具 {planned.tool_call.name} 执行失败: {exc}",
                            data={"tool": planned.tool_call.name},
                            meta={"exception": repr(exc)},
                        )
                    outputs[index] = self.hook_manager.after_tool_execute(
                        hook_context,
                        prepared,
                        raw_output,
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
        hook_context: HookContext,
        emitter: RuntimeEventEmitter | None,
    ) -> ToolOutput:
        del emitter
        prepared_or_output = self._prepare_tool_call(planned, context, hook_context)
        if isinstance(prepared_or_output, ToolOutput):
            return prepared_or_output
        output = self.tool_registry.invoke(prepared_or_output)
        return self.hook_manager.after_tool_execute(
            hook_context,
            prepared_or_output,
            output,
        )

    def _prepare_tool_call(
        self,
        planned: _PlannedToolCall,
        context: ToolExecutionContext,
        hook_context: HookContext,
    ) -> PreparedToolCall | ToolOutput:
        request_or_output = self.hook_manager.before_tool_prepare(
            hook_context,
            ToolPrepareRequest(
                tool_call_id=planned.tool_call.id,
                tool_name=planned.tool_call.name,
                args=planned.args,
                tool=planned.tool,
            ),
        )
        if isinstance(request_or_output, ToolOutput):
            return request_or_output

        prepared = self.tool_registry.prepare(
            request_or_output.tool_name,
            request_or_output.args,
            context=context,
        )
        if isinstance(prepared, ToolOutput):
            return prepared
        prepared = PreparedToolCall(
            tool=prepared.tool,
            args=prepared.args,
            context=prepared.context,
            tool_call_id=planned.tool_call.id,
        )
        decision = self.hook_manager.before_tool_execute(hook_context, prepared)
        if decision.blocked:
            assert decision.output is not None
            return decision.output
        return prepared

    def _resolve_emitter(self, run_spec: RunSpec) -> RuntimeEventEmitter | None:
        if run_spec.event_emitter is not None:
            return run_spec.event_emitter
        if self.event_handler is None:
            return None
        return RuntimeEventEmitter(
            run_id=run_spec.run_id or _make_local_run_id(),
            session_id=run_spec.session_id,
            handler=self.event_handler,
        )

    @staticmethod
    def _hook_context(
        run_spec: RunSpec,
        emitter: RuntimeEventEmitter | None,
    ) -> HookContext:
        return HookContext(
            run_id=run_spec.run_id or ("" if emitter is None else emitter.run_id),
            session_id=run_spec.session_id,
            workspace=run_spec.workspace or Path.cwd(),
            mode=run_spec.mode,
            emitter=emitter,
            cancel_event=run_spec.cancel_event,
        )

    @staticmethod
    def _emit(
        emitter: RuntimeEventEmitter | None,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if emitter is not None:
            emitter.emit(event_type, payload)


def _to_agent_message(message: ModelMessage | dict[str, Any]) -> AgentMessage:
    """Convert an internal model message to the runner output type."""
    if isinstance(message, ModelMessage):
        return AgentMessage.create(
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            name=message.tool_name,
            reasoning_content=message.reasoning_content,
        )
    return AgentMessage.create(
        role=str(message["role"]),
        content=str(message.get("content", "")),
        tool_calls=message.get("tool_calls"),
        tool_call_id=message.get("tool_call_id"),
        name=message.get("name"),
        reasoning_content=message.get("reasoning_content"),
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


def _usage_payload(usage: TokenUsage | None) -> dict[str, int | None] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
