"""The agent loop: single owner of one user turn.

``run_turn`` reads top-to-bottom as the full lifecycle of a turn:

1. budget check — compact via ``Compactor`` when the projected request is
   too large (persisted immediately, surfaced as an event)
2. context assembly — a pure ``ContextBuilder.build`` call
3. model call
4. tool execution (approval injected as ``ToolApprovalGate``)
5. message append — session persistence plus a runtime event

Everything observable about a run leaves through the event emitter; the
returned ``TurnOutcome`` is a convenience for the caller, not a second
channel of truth.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
from dataclasses import dataclass
import threading
import time
from typing import TYPE_CHECKING, Any

from ..llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from ..session import MessageEvent, SessionEntry
from ..tools.base import Tool, ToolExecutionContext
from ..tools.registry import PreparedToolCall, ToolRegistry
from ..tools.result import ToolOutput
from .approval import ToolApprovalGate
from .budget import TokenBudget
from .cancel import RunCancelled
from .compactor import Compactor
from .context_builder import BuiltRequest, ContextBuilder
from .events import RuntimeEventEmitter
from .tool_output_materializer import ToolOutputMaterializer

if TYPE_CHECKING:
    from ..session import Session, SessionManager


@dataclass(frozen=True)
class TurnOutcome:
    """Result of one handled user turn, folded from the run's events."""

    reply: str
    usage: TokenUsage | None = None
    did_compact: bool = False
    compact_message: str | None = None


@dataclass(frozen=True)
class _PlannedToolCall:
    """One model-requested tool call annotated with local execution metadata."""

    tool_call: ToolCall
    args: dict[str, Any]
    tool: Tool | None


class AgentLoop:
    """Run user turns against one LLM, tool registry, and session store."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        context_builder: ContextBuilder,
        budget: TokenBudget,
        compactor: Compactor,
        materializer: ToolOutputMaterializer,
        model: str,
        approval_gate: ToolApprovalGate | None = None,
        max_iterations: int = 20,
        max_parallel_tools: int = 4,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.session_manager = session_manager
        self.context_builder = context_builder
        self.budget = budget
        self.compactor = compactor
        self.materializer = materializer
        self.model = model
        self.approval_gate = approval_gate
        self.max_iterations = max_iterations
        self.max_parallel_tools = max_parallel_tools

    def run_turn(
        self,
        session: Session,
        user_input: str,
        *,
        emitter: RuntimeEventEmitter,
        cancel_event: threading.Event | None = None,
    ) -> TurnOutcome:
        """Run one user turn to completion; the loop owns all turn state."""
        tool_context = ToolExecutionContext(
            session_id=session.session_id,
            run_id=emitter.run_id,
            cancel_event=cancel_event,
        )
        usage: TokenUsage | None = None
        observed_input_tokens: int | None = None
        compact_messages: list[str] = []
        executor: ThreadPoolExecutor | None = None

        self._check_cancel(cancel_event)
        self._emit_context_usage(session, emitter)
        self._append(session, MessageEvent.create(role="user", content=user_input))

        try:
            for iteration in range(1, self.max_iterations + 1):
                self._check_cancel(cancel_event)
                started = time.perf_counter()

                built, compact_message = self._prepare_context(
                    session,
                    observed_input_tokens=observed_input_tokens,
                    cancel_event=cancel_event,
                )
                if compact_message is not None:
                    compact_messages.append(compact_message)
                    emitter.emit(
                        "context.compacted",
                        {"iteration": iteration, "message": compact_message},
                    )

                emitter.emit(
                    "model.request.started",
                    {
                        "iteration": iteration,
                        "model": self.model,
                        "input_preview": _preview(user_input),
                    },
                )
                resp = self.llm.chat(
                    built.messages,
                    built.tool_definitions,
                    model=self.model,
                )
                usage = _merge_usage(usage, resp.usage)
                observed_input_tokens = (
                    None if resp.usage is None else resp.usage.input_tokens
                )
                completed_payload: dict[str, Any] = {
                    "iteration": iteration,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "tool_call_count": len(resp.tool_calls),
                    "usage": _usage_payload(resp.usage),
                }

                if not resp.tool_calls:
                    reply = _normalized_reply(resp.content).strip()
                    if not reply:
                        completed_payload.update(
                            {"empty_reply": True, "response_debug": resp.debug}
                        )
                        emitter.emit("model.request.completed", completed_payload)
                        raise RuntimeError(
                            "模型返回空回复，请重试；详见上一条空回答诊断日志。"
                        )
                    emitter.emit("model.request.completed", completed_payload)
                    self._append(
                        session,
                        MessageEvent.create(
                            role="assistant",
                            content=reply,
                            reasoning_content=resp.reasoning_content,
                        ),
                    )
                    emitter.emit(
                        "message.completed",
                        {"iteration": iteration, "content": reply},
                    )
                    return self._outcome(reply, usage, compact_messages)

                emitter.emit("model.request.completed", completed_payload)
                self._append(session, _response_to_message_event(resp))

                planned_tool_calls = self._plan_tool_calls(resp.tool_calls, emitter)
                self._check_cancel(cancel_event)
                if executor is None and self._wants_executor(planned_tool_calls):
                    executor = ThreadPoolExecutor(
                        max_workers=max(2, self.max_parallel_tools)
                    )
                tool_outputs = self._execute_tool_calls(
                    planned_tool_calls,
                    tool_context,
                    emitter,
                    cancel_event,
                    executor,
                )

                for planned, output in zip(planned_tool_calls, tool_outputs):
                    self._check_cancel(cancel_event)
                    result = self.materializer.materialize(output, context=tool_context)
                    emitter.emit(
                        "tool_call.completed" if result.ok else "tool_call.failed",
                        {
                            "tool_call_id": planned.tool_call.id,
                            "tool": planned.tool_call.name,
                            "display_name": _tool_label(
                                planned.tool, planned.tool_call.name
                            ),
                            "source": None if planned.tool is None else planned.tool.source,
                            "ok": result.ok,
                            "code": result.code,
                            "summary": result.summary,
                            "artifact": (
                                None
                                if result.artifact is None
                                else result.artifact.to_dict()
                            ),
                            "truncated": result.truncated,
                        },
                    )
                    self._append(
                        session,
                        MessageEvent.create(
                            role="tool",
                            tool_call_id=planned.tool_call.id,
                            name=planned.tool_call.name,
                            content=result.to_model_content(),
                        ),
                    )

            reply = "抱歉，工具调用轮次已达上限，请简化问题后重试。"
            final_message = self._append(
                session, MessageEvent.create(role="assistant", content=reply)
            )
            emitter.emit(
                "message.completed",
                {
                    "content": final_message.content,
                    "reason": "max_iterations",
                    "max_iterations": self.max_iterations,
                },
            )
            return self._outcome(final_message.content, usage, compact_messages)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def estimate_visible_tokens(self, session: Session) -> int:
        return self.budget.estimate(self.context_builder.build(session.messages))

    # ── turn steps ────────────────────────────────────────────────

    def _prepare_context(
        self,
        session: Session,
        *,
        observed_input_tokens: int | None,
        cancel_event: threading.Event | None,
    ) -> tuple[BuiltRequest, str | None]:
        """Step ①+②: check the budget, reduce if needed, assemble the request."""
        built = self.context_builder.build(session.messages)
        tokens = self.budget.request_tokens(
            built,
            session=session,
            observed_input_tokens=observed_input_tokens,
        )
        if tokens <= self.budget.input_budget:
            self.budget.remember(session)
            return built, None

        message = self.compactor.reduce(
            session,
            tokens_before=tokens,
            cancel_event=cancel_event,
        )
        built = self.context_builder.build(session.messages)
        self.budget.remember(session)
        return built, message

    def _append(self, session: Session, message: MessageEvent) -> MessageEvent:
        """Step ⑤: the loop's own state mutation, persisted in the same call."""
        session.add_message(message)
        self.session_manager.append_entries(
            session.session_id,
            [SessionEntry.from_message(message)],
        )
        self.session_manager.update_metadata(session)
        return message

    def _emit_context_usage(
        self,
        session: Session,
        emitter: RuntimeEventEmitter,
    ) -> None:
        emitter.emit(
            "context.usage",
            {
                "current_tokens": self.estimate_visible_tokens(session),
                "budget": self.budget.input_budget,
            },
        )

    def _outcome(
        self,
        reply: str,
        usage: TokenUsage | None,
        compact_messages: list[str],
    ) -> TurnOutcome:
        return TurnOutcome(
            reply=reply,
            usage=usage,
            did_compact=bool(compact_messages),
            compact_message="\n".join(compact_messages) or None,
        )

    # ── tool execution ────────────────────────────────────────────

    def _plan_tool_calls(
        self,
        tool_calls: list[ToolCall],
        emitter: RuntimeEventEmitter,
    ) -> list[_PlannedToolCall]:
        planned: list[_PlannedToolCall] = []
        for tool_call in tool_calls:
            args = _parse_args(tool_call.arguments)
            tool = self.tool_registry.get(tool_call.name)
            emitter.emit(
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
            planned.append(_PlannedToolCall(tool_call=tool_call, args=args, tool=tool))
        return planned

    def _wants_executor(self, planned_tool_calls: list[_PlannedToolCall]) -> bool:
        return any(
            len(batch) > 1 for batch in self._partition_tool_batches(planned_tool_calls)
        )

    def _execute_tool_calls(
        self,
        planned_tool_calls: list[_PlannedToolCall],
        context: ToolExecutionContext,
        emitter: RuntimeEventEmitter,
        cancel_event: threading.Event | None,
        executor: ThreadPoolExecutor | None,
    ) -> list[ToolOutput]:
        outputs: list[ToolOutput] = []
        for batch in self._partition_tool_batches(planned_tool_calls):
            self._check_cancel(cancel_event)
            if len(batch) > 1 and executor is not None:
                outputs.extend(
                    self._execute_parallel_batch(
                        batch, context, emitter, cancel_event, executor
                    )
                )
                continue
            for planned in batch:
                outputs.append(
                    self._execute_single(planned, context, emitter, cancel_event)
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
        emitter: RuntimeEventEmitter,
        cancel_event: threading.Event | None,
        executor: ThreadPoolExecutor,
    ) -> list[ToolOutput]:
        outputs: list[ToolOutput | None] = [None] * len(batch)
        submitted: list[tuple[int, _PlannedToolCall, Future[ToolOutput]]] = []

        for index, planned in enumerate(batch):
            self._check_cancel(cancel_event)
            prepared_or_output = self._prepare_tool_call(
                planned, context, emitter, cancel_event
            )
            if isinstance(prepared_or_output, ToolOutput):
                outputs[index] = prepared_or_output
                continue
            submitted.append(
                (
                    index,
                    planned,
                    executor.submit(self.tool_registry.invoke, prepared_or_output),
                )
            )

        for index, planned, future in submitted:
            try:
                outputs[index] = self._wait_for_future(future, cancel_event=cancel_event)
            except RunCancelled:
                raise
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

    def _execute_single(
        self,
        planned: _PlannedToolCall,
        context: ToolExecutionContext,
        emitter: RuntimeEventEmitter,
        cancel_event: threading.Event | None,
    ) -> ToolOutput:
        self._check_cancel(cancel_event)
        prepared_or_output = self._prepare_tool_call(
            planned, context, emitter, cancel_event
        )
        if isinstance(prepared_or_output, ToolOutput):
            return prepared_or_output
        self._check_cancel(cancel_event)
        return self.tool_registry.invoke(prepared_or_output)

    def _prepare_tool_call(
        self,
        planned: _PlannedToolCall,
        context: ToolExecutionContext,
        emitter: RuntimeEventEmitter,
        cancel_event: threading.Event | None,
    ) -> PreparedToolCall | ToolOutput:
        self._check_cancel(cancel_event)
        prepared = self.tool_registry.prepare(
            planned.tool_call.name,
            planned.args,
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
        if self.approval_gate is None:
            return prepared
        self._check_cancel(cancel_event)
        try:
            denial = self.approval_gate.check(
                prepared,
                run_id=context.run_id or emitter.run_id,
                session_id=context.session_id,
                emitter=emitter,
                cancel_event=cancel_event,
            )
        except RunCancelled:
            raise
        except Exception as exc:
            return ToolOutput.failure(
                "error",
                f"工具 {prepared.tool.name} 审批流程失败: {exc}",
                data={"tool": prepared.tool.name},
                meta={"exception": repr(exc)},
            )
        if denial is not None:
            return denial
        return prepared

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
                self._check_cancel(cancel_event)

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("run cancelled by user")


def _response_to_message_event(resp: LLMResponse) -> MessageEvent:
    return MessageEvent.create(
        role="assistant",
        content=resp.content or "",
        reasoning_content=resp.reasoning_content,
        tool_calls=[
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]
        or None,
    )


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
