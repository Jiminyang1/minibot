"""Tool approval as an explicit runtime dependency.

Approval is the one policy that must interrupt the loop and wait for a human
decision, so it is injected into ``AgentLoop`` as a named collaborator rather
than hidden inside a hook pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import threading
from typing import TYPE_CHECKING, Any
import uuid

from ..config import ApprovalMode
from ..tools.base import ShownText, UntrustedExposure, written_forms
from ..tools.registry import PreparedToolCall, ToolRegistry
from ..tools.result import ToolOutput
from .cancel import RunCancelled
from .events import RuntimeEventEmitter

if TYPE_CHECKING:
    from ..session import Session


@dataclass(frozen=True)
class ApprovalRequest:
    """One pending approval decision for a sensitive tool call."""

    run_id: str
    session_id: str
    approval_id: str
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    reason: str | None = None


def untrusted_exposure(session: Session, registry: ToolRegistry) -> UntrustedExposure:
    """Fold the whole session (compacted entries included) into an exposure.

    Sticky by design: injected text may survive in a summary or act turns
    later, so once an untrusted tool has answered, the session stays tainted.

    The exposure vouches for URLs, so it separates who wrote what. User
    messages always vouch. A successful result of an untrusted tool vouches
    only for text the model had not written before it arrived: a URL the
    model put into an earlier reply, tool call or summary and then got back
    — a search query echoed with its whitespace collapsed, a value stored in
    one call and read in the next — is still a URL the model built (the same
    idea as the ``url_sources`` of Anthropic's web_fetch tool). Failed
    results never vouch: errors and denials repeat the call's arguments.

    Residual: a tool that computes its output (joins non-adjacent strings,
    decodes base64) can return a URL the model never wrote in one piece.
    """
    tainted = False
    seen: list[str] = []
    shown: list[ShownText] = []
    written: list[str] = []
    written_length = 0

    def wrote(text: str | None) -> None:
        nonlocal written_length
        for form in written_forms(text or ""):
            written.append(form)
            written_length += len(form)

    for entry in session.entries:
        # A summary is the summariser model's text.
        wrote(entry.summary)
        message = entry.message
        if message is None:
            continue
        if message.role == "user":
            seen.append(message.content)
        elif message.role == "assistant":
            wrote(message.content)
            for call in message.tool_calls or []:
                for text in _argument_strings(call.arguments):
                    wrote(text)
        elif message.role == "tool":
            tool = registry.get(message.name or "")
            if tool is None:
                # An MCP server that is gone now still answered back then.
                untrusted = (message.name or "").startswith("mcp__")
            else:
                untrusted = tool.untrusted_output
            tainted = tainted or untrusted
            if untrusted and _succeeded(message.content):
                shown.append(ShownText(message.content, written_length))
    return UntrustedExposure(
        tainted=tainted,
        seen_text="\n".join(seen),
        shown=tuple(shown),
        # No separator: pieces the model wrote side by side also match a
        # tool that joins them.
        written="".join(written),
    )


def _argument_strings(arguments: str) -> list[str]:
    """Every key and string value in a call's JSON arguments, in order."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return [arguments] if arguments else []
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                found.append(str(key))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(parsed)
    return found


def _succeeded(content: str) -> bool:
    try:
        return bool(json.loads(content).get("ok"))
    except (json.JSONDecodeError, AttributeError):
        # Not a tool-result envelope (e.g. raw MCP text): it did not fail.
        return True


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
            return False
        return bool(self.handler(request, cancel_event))


class ToolApprovalGate:
    """Gate calls whose tool gives an ``approval_reason`` before execution."""

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
        exposure: UntrustedExposure | None = None,
    ) -> ToolOutput | None:
        """Return ``None`` when the call may run, or a denial ``ToolOutput``."""
        reason = call.tool.approval_reason(call.args, exposure or UntrustedExposure())
        if reason is None:
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
            # Fail closed: a frontend that never wired an approval channel
            # must not let sensitive tools run unasked.
            _emit(
                emitter,
                "approval.resolved",
                {
                    "tool_call_id": call.tool_call_id,
                    "tool": call.tool.name,
                    "approved": False,
                    "auto": True,
                },
            )
            return ToolOutput.failure(
                "denied",
                f"工具 {call.tool.name} 需要审批，但当前没有可用的审批渠道，已拒绝执行。",
                # No args: a denial must not put the refused URL in front of the
                # model as if a tool had shown it.
                data={"tool": call.tool.name, "reason": reason},
            )

        approval_id = "ap_" + uuid.uuid4().hex[:12]
        request = ApprovalRequest(
            run_id=run_id,
            session_id=session_id,
            approval_id=approval_id,
            tool_call_id=call.tool_call_id or "",
            tool_name=call.tool.name,
            args=call.args,
            reason=reason,
        )
        _emit(
            emitter,
            "approval.required",
            {
                "approval_id": approval_id,
                "tool_call_id": call.tool_call_id,
                "tool": call.tool.name,
                "args": call.args,
                "reason": reason,
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
            data={"tool": call.tool.name},
        )


def _emit(
    emitter: RuntimeEventEmitter | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if emitter is not None:
        emitter.emit(event_type, payload)
