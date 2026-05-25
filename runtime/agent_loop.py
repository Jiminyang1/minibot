"""Agent run spec and execution loop for MiniBot."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from ..run_log import make_run_id
from ..session import MessageEvent
from ..tools.base import Tool, ToolExecutionContext
from ..tools.registry import PreparedToolCall, ToolRegistry
from ..tools.result import ToolOutput
from .cancel import RunCancelled
from .context_manager import WorkingContext
from .events import RuntimeEventEmitter, RuntimeEventHandler
from .hooks import (
    HookContext,
    ModelRequest,
    RuntimeHookManager,
    ToolPrepareRequest,
)
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


def _tool_label(tool: Tool | None, default: str) -> str:
    if tool is None:
        return default
    return tool.display_name


@dataclass(frozen=True)
class RunSpec:
    """One concrete agent execution prepared by the turn engine."""

    session_id: str
    model: str
    user_input: str
    prepare_next_turn: Callable[[int | None], WorkingContext]
    on_message: Callable[[MessageEvent], MessageEvent]
    max_iterations: int = 20
    run_id: str | None = None
    event_emitter: RuntimeEventEmitter | None = None
    cancel_event: threading.Event | None = None
    mode: str = "default"
    workspace: Path | None = None


@dataclass(frozen=True)
class RunOutcome:
    """Full outcome of one prepared agent run."""

    reply: str
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class PartialRunError(Exception):
    """Internal carrier for failures after partial agent progress."""

    cause: Exception
    usage: TokenUsage | None = None
    reply: str | None = None

    def __init__(
        self,
        *,
        cause: Exception,
        usage: TokenUsage | None = None,
        reply: str | None = None,
    ) -> None:
        super().__init__(str(cause))
        object.__setattr__(self, "cause", cause)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "reply", reply)


@dataclass(frozen=True)
class _PlannedToolCall:
    """One model-requested tool call annotated with local execution metadata."""

    tool_call: ToolCall
    args: dict[str, Any]
    tool: Tool | None


class AgentLoop:
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
        tool_context = ToolExecutionContext(
            session_id=run_spec.session_id,
            run_id=run_spec.run_id,
            cancel_event=run_spec.cancel_event,
        )
        emitter = self._resolve_emitter(run_spec)
        hook_context = self._hook_context(run_spec, emitter)

        usage: TokenUsage | None = None
        observed_input_tokens: int | None = None

        def _check_cancel() -> None:
            self._check_cancel_event(run_spec.cancel_event)

        _check_cancel()
        run_spec.on_message(
            MessageEvent.create(role="user", content=run_spec.user_input)
        )

        for iteration in range(1, run_spec.max_iterations + 1):
            _check_cancel()
            started = time.perf_counter()

            try:
                working_context = run_spec.prepare_next_turn(observed_input_tokens)
                if working_context.did_compact:
                    self._emit(
                        emitter,
                        "context.compacted",
                        {
                            "iteration": iteration,
                            "message": working_context.compact_message,
                        },
                    )
                self._emit(
                    emitter,
                    "model.request.started",
                    {
                        "iteration": iteration,
                        "model": run_spec.model,
                        "input_preview": _preview(run_spec.user_input),
                    },
                )
                model_request = self.hook_manager.before_model_request(
                    hook_context,
                    ModelRequest(
                        model=run_spec.model,
                        messages=working_context.messages,
                        tool_definitions=working_context.tool_definitions,
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
                if usage is not None:
                    raise PartialRunError(cause=exc, usage=usage) from exc
                raise
            usage = _merge_usage(usage, resp.usage)
            observed_input_tokens = None if resp.usage is None else resp.usage.input_tokens
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
                    assistant_msg = run_spec.on_message(
                        MessageEvent.create(
                            role="assistant",
                            content=reply,
                            reasoning_content=resp.reasoning_content,
                        )
                    )
                    reply = assistant_msg.content
                    self._emit(
                        emitter,
                        "message.completed",
                        {"iteration": iteration, "content": reply},
                    )
                    return RunOutcome(
                        reply=reply,
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
                    usage=usage,
                )

            self._emit(
                emitter,
                "model.request.completed",
                model_completed_payload,
            )
            run_spec.on_message(self._response_to_message_event(resp))

            planned_tool_calls = self._plan_tool_calls(resp.tool_calls, emitter)
            _check_cancel()
            tool_outputs = self._execute_tool_calls(
                planned_tool_calls,
                tool_context,
                hook_context,
                emitter,
            )

            for planned, output in zip(planned_tool_calls, tool_outputs):
                _check_cancel()
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

                run_spec.on_message(
                    MessageEvent.create(
                        role="tool",
                        tool_call_id=planned.tool_call.id,
                        name=planned.tool_call.name,
                        content=result.to_model_content(),
                    )
                )

        max_iterations_reply = "抱歉，工具调用轮次已达上限，请简化问题后重试。"
        final_message = run_spec.on_message(
            MessageEvent.create(role="assistant", content=max_iterations_reply)
        )
        self._emit(
            emitter,
            "message.completed",
            {
                "content": final_message.content,
                "reason": "max_iterations",
                "max_iterations": run_spec.max_iterations,
            },
        )
        return RunOutcome(reply=final_message.content, usage=usage)

    @staticmethod
    def _response_to_message_event(resp: LLMResponse) -> MessageEvent:
        return MessageEvent.create(
            role="assistant",
            content=resp.content or "",
            reasoning_content=resp.reasoning_content,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
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
            self._check_cancel_event(hook_context.cancel_event)
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
        submitted: list[tuple[int, _PlannedToolCall, PreparedToolCall]] = []

        for index, planned in enumerate(batch):
            self._check_cancel_event(hook_context.cancel_event)
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
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
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
                        raw_output = self._wait_for_future(
                            future,
                            cancel_event=hook_context.cancel_event,
                        )
                    except Exception as exc:
                        if isinstance(exc, RunCancelled):
                            raise
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
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

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
        self._check_cancel_event(hook_context.cancel_event)
        prepared_or_output = self._prepare_tool_call(planned, context, hook_context)
        if isinstance(prepared_or_output, ToolOutput):
            return prepared_or_output
        self._check_cancel_event(hook_context.cancel_event)
        output = self.tool_registry.invoke(prepared_or_output)
        self._check_cancel_event(hook_context.cancel_event)
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
        self._check_cancel_event(hook_context.cancel_event)
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
        self._check_cancel_event(hook_context.cancel_event)
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
            run_id=run_spec.run_id or make_run_id(),
            session_id=run_spec.session_id,
            handler=self.event_handler,
        )

    def _wait_for_future(
        self,
        future: Future[ToolOutput],
        *,
        cancel_event: threading.Event | None,
    ) -> ToolOutput:
        while True:
            try:
                return future.result(timeout=0.05)
            except FutureTimeoutError:
                self._check_cancel_event(cancel_event)

    @staticmethod
    def _check_cancel_event(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("run cancelled by user")

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
