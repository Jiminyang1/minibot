"""Internal runtime hook pipeline for MiniBot.

Hooks are intentionally narrow: they can inspect or transform boundary
objects, emit runtime events, and block tool execution by returning a
``ToolOutput``. They do not receive core runtime objects such as sessions,
engines, registries, or UI adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from ..llm import LLMResponse
from ..tools.base import Tool
from ..tools.definitions import ModelToolDefinition
from ..tools.registry import PreparedToolCall
from ..tools.result import ToolOutput
from .cancel import RunCancelled
from .events import RuntimeEventEmitter
from .messages import ModelMessage

if TYPE_CHECKING:
    from .context_manager import WorkingContext
    from .turn_engine import TurnResult


@dataclass(frozen=True)
class HookContext:
    """Run-scoped context available to hooks."""

    run_id: str
    session_id: str
    workspace: Path
    mode: str = "default"
    emitter: RuntimeEventEmitter | None = None
    cancel_event: threading.Event | None = None


@dataclass(frozen=True)
class ModelRequest:
    """One concrete model request before it reaches the LLM client."""

    model: str
    messages: list[ModelMessage]
    tool_definitions: list[ModelToolDefinition]


@dataclass(frozen=True)
class ToolPrepareRequest:
    """Parsed model tool call before registry validation."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    tool: Tool | None


@dataclass(frozen=True)
class ToolExecuteDecision:
    """Decision returned before a prepared tool is invoked."""

    output: ToolOutput | None = None

    @property
    def blocked(self) -> bool:
        return self.output is not None


class RuntimeHook:
    """Base class for internal runtime hooks."""

    priority = 100

    def after_context(
        self,
        context: HookContext,
        prepared: WorkingContext,
    ) -> WorkingContext:
        del context
        return prepared

    def before_model_request(
        self,
        context: HookContext,
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        return request

    def after_model_response(
        self,
        context: HookContext,
        request: ModelRequest,
        response: LLMResponse,
    ) -> LLMResponse:
        del context, request
        return response

    def before_tool_prepare(
        self,
        context: HookContext,
        request: ToolPrepareRequest,
    ) -> ToolPrepareRequest:
        del context
        return request

    def before_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
    ) -> ToolExecuteDecision:
        del context, call
        return ToolExecuteDecision()

    def after_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
        output: ToolOutput,
    ) -> ToolOutput:
        del context, call
        return output

    def after_turn(
        self,
        context: HookContext,
        result: TurnResult,
    ) -> TurnResult:
        del context
        return result

    def on_error(
        self,
        context: HookContext,
        error: Exception,
    ) -> None:
        del context, error


class RuntimeHookManager:
    """Ordered hook pipeline used by the runtime orchestration layer."""

    def __init__(self, hooks: list[RuntimeHook] | None = None) -> None:
        self._hooks: list[RuntimeHook] = []
        for hook in hooks or []:
            self.register(hook)

    @property
    def hooks(self) -> tuple[RuntimeHook, ...]:
        return tuple(self._hooks)

    def register(self, hook: RuntimeHook) -> None:
        self._hooks.append(hook)
        self._hooks.sort(key=lambda item: item.priority)

    def after_context(
        self,
        context: HookContext,
        prepared: WorkingContext,
    ) -> WorkingContext:
        current = prepared
        for hook in self._hooks:
            current = hook.after_context(context, current)
        return current

    def before_model_request(
        self,
        context: HookContext,
        request: ModelRequest,
    ) -> ModelRequest:
        current = request
        for hook in self._hooks:
            current = hook.before_model_request(context, current)
        return current

    def after_model_response(
        self,
        context: HookContext,
        request: ModelRequest,
        response: LLMResponse,
    ) -> LLMResponse:
        current = response
        for hook in self._hooks:
            current = hook.after_model_response(context, request, current)
        return current

    def before_tool_prepare(
        self,
        context: HookContext,
        request: ToolPrepareRequest,
    ) -> ToolPrepareRequest | ToolOutput:
        current = request
        for hook in self._hooks:
            try:
                current = hook.before_tool_prepare(context, current)
            except Exception as exc:
                if isinstance(exc, RunCancelled):
                    raise
                return _hook_failure(
                    "before_tool_prepare",
                    exc,
                    tool_name=current.tool_name,
                )
        return current

    def before_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
    ) -> ToolExecuteDecision:
        for hook in self._hooks:
            try:
                decision = hook.before_tool_execute(context, call)
            except Exception as exc:
                if isinstance(exc, RunCancelled):
                    raise
                return ToolExecuteDecision(
                    _hook_failure(
                        "before_tool_execute",
                        exc,
                        tool_name=call.tool.name,
                    )
                )
            if decision.blocked:
                return decision
        return ToolExecuteDecision()

    def after_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
        output: ToolOutput,
    ) -> ToolOutput:
        current = output
        for hook in self._hooks:
            try:
                current = hook.after_tool_execute(context, call, current)
            except Exception as exc:
                if isinstance(exc, RunCancelled):
                    raise
                return _hook_failure(
                    "after_tool_execute",
                    exc,
                    tool_name=call.tool.name,
                )
        return current

    def after_turn(
        self,
        context: HookContext,
        result: TurnResult,
    ) -> TurnResult:
        current = result
        for hook in self._hooks:
            current = hook.after_turn(context, current)
        return current

    def on_error(self, context: HookContext, error: Exception) -> None:
        for hook in self._hooks:
            try:
                hook.on_error(context, error)
            except Exception:
                continue


def _hook_failure(phase: str, exc: Exception, *, tool_name: str) -> ToolOutput:
    return ToolOutput.failure(
        "error",
        f"工具 hook {phase} 失败: {exc}",
        data={"tool": tool_name, "hook_phase": phase},
        meta={"exception": repr(exc)},
    )
