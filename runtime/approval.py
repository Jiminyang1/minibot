"""Tool approval as an explicit runtime dependency.

Approval is the one policy that must interrupt the loop and wait for a human
decision, so it is injected into ``AgentLoop`` as a named collaborator rather
than hidden inside a hook pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
from typing import Any
import uuid

from ..config import ApprovalMode
from ..tools.registry import PreparedToolCall
from ..tools.result import ToolOutput
from .cancel import RunCancelled
from .events import RuntimeEventEmitter


@dataclass(frozen=True)
class ApprovalRequest:
    """One pending approval decision for a sensitive tool call."""

    run_id: str
    session_id: str
    approval_id: str
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class ApprovalPolicy:
    """Thread-safe approval mode and user-decision callback."""

    def __init__(
        self,
        *,
        handler: Callable[[ApprovalRequest, threading.Event | None], bool] | None = None,
        mode: ApprovalMode = "ask",
    ) -> None:
        self.handler = handler
        self._mode: ApprovalMode = mode
        self._lock = threading.Lock()

    @property
    def mode(self) -> ApprovalMode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: ApprovalMode) -> None:
        if mode not in {"ask", "always"}:
            raise ValueError("approval mode must be ask or always")
        with self._lock:
            self._mode = mode

    def request(
        self,
        request: ApprovalRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("run cancelled while waiting for approval")
        if self.handler is None:
            return True
        return bool(self.handler(request, cancel_event))


class ToolApprovalGate:
    """Gate tools that declare ``requires_approval`` before execution."""

    def __init__(self, policy: ApprovalPolicy) -> None:
        self.policy = policy

    def check(
        self,
        call: PreparedToolCall,
        *,
        run_id: str,
        session_id: str,
        emitter: RuntimeEventEmitter | None,
        cancel_event: threading.Event | None,
    ) -> ToolOutput | None:
        """Return ``None`` when the call may run, or a denial ``ToolOutput``."""
        if not call.tool.requires_approval:
            return None

        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("run cancelled before approval")

        if self.policy.mode == "always":
            _emit(
                emitter,
                "approval.resolved",
                {
                    "tool_call_id": call.tool_call_id,
                    "tool": call.tool.name,
                    "approved": True,
                    "auto": True,
                },
            )
            return None

        if self.policy.handler is None:
            return None

        approval_id = "ap_" + uuid.uuid4().hex[:12]
        request = ApprovalRequest(
            run_id=run_id,
            session_id=session_id,
            approval_id=approval_id,
            tool_call_id=call.tool_call_id or "",
            tool_name=call.tool.name,
            args=call.args,
        )
        _emit(
            emitter,
            "approval.required",
            {
                "approval_id": approval_id,
                "tool_call_id": call.tool_call_id,
                "tool": call.tool.name,
                "args": call.args,
            },
        )
        approved = self.policy.request(request, cancel_event=cancel_event)
        _emit(
            emitter,
            "approval.resolved",
            {
                "approval_id": approval_id,
                "tool_call_id": call.tool_call_id,
                "tool": call.tool.name,
                "approved": approved,
            },
        )
        if approved:
            return None
        return ToolOutput.failure(
            "denied",
            f"工具 {call.tool.name} 未获批准执行。",
            data={"tool": call.tool.name, "args": call.args},
        )


def _emit(
    emitter: RuntimeEventEmitter | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if emitter is not None:
        emitter.emit(event_type, payload)
