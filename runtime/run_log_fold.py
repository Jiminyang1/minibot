"""runs.jsonl as a fold over the runtime event stream.

The fold subscribes to the same events the CLI and SSE clients see and
reduces each run into one ``RunLogRecord`` when its terminal event arrives.
Nothing in the loop knows run logging exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import TYPE_CHECKING, Any

from ..run_log import RunLogRecord, RunLogStore, preview_text, utc_now
from .events import RuntimeEvent

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


@dataclass
class _RunAccumulator:
    session_id: str
    timestamp: str
    started_perf: float
    model: str = ""
    turn_index: int = 0
    user_input_preview: str = ""
    llm_call_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    saw_usage: bool = False
    tool_call_count: int = 0
    tools_used: list[str] = field(default_factory=list)
    mcp_tool_call_count: int = 0
    mcp_servers_used: set[str] = field(default_factory=set)
    mcp_transports_used: set[str] = field(default_factory=set)
    mcp_error_count: int = 0
    did_compact: bool = False
    compact_messages: list[str] = field(default_factory=list)


class RunLogFold:
    """Reduce one run's events into a persisted run summary record."""

    _MAX_ACTIVE_RUNS = 256

    def __init__(
        self,
        store: RunLogStore,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.store = store
        self.tool_registry = tool_registry
        self._active: dict[str, _RunAccumulator] = {}
        self._lock = threading.Lock()

    def __call__(self, event: RuntimeEvent) -> None:
        try:
            self._fold(event)
        except Exception:
            # Run logging is observability bookkeeping; never fail a run
            # because a summary row could not be reduced or written.
            return

    def _fold(self, event: RuntimeEvent) -> None:
        payload = event.payload
        if event.type == "run.started":
            self._start(event)
            return

        with self._lock:
            acc = self._active.get(event.run_id)
        if acc is None:
            return

        if event.type == "model.request.completed":
            acc.llm_call_count += 1
            self._add_usage(acc, payload.get("usage"))
        elif event.type in {"tool_call.completed", "tool_call.failed"}:
            self._add_tool_call(acc, payload)
        elif event.type == "context.compacted":
            acc.did_compact = True
            message = payload.get("message")
            if message:
                acc.compact_messages.append(str(message))
        elif event.type == "run.completed":
            self._finish(
                event,
                acc,
                status="success",
                reply=payload.get("reply"),
            )
        elif event.type == "run.failed":
            self._finish(
                event,
                acc,
                status="failed",
                error_type=str(payload.get("error_type") or "Exception"),
                error_message=str(payload.get("message") or ""),
            )
        elif event.type == "run.cancelled":
            self._finish(
                event,
                acc,
                status="failed",
                error_type="RunCancelled",
                error_message="run cancelled by user",
            )

    def _start(self, event: RuntimeEvent) -> None:
        payload = event.payload
        acc = _RunAccumulator(
            session_id=event.session_id,
            timestamp=event.created_at,
            started_perf=time.perf_counter(),
            model=str(payload.get("model") or ""),
            turn_index=int(payload.get("turn_index") or 0),
            user_input_preview=str(payload.get("input_preview") or ""),
        )
        with self._lock:
            self._active[event.run_id] = acc
            overflow = len(self._active) - self._MAX_ACTIVE_RUNS
            if overflow > 0:
                for stale_id in list(self._active.keys())[:overflow]:
                    self._active.pop(stale_id, None)

    def _finish(
        self,
        event: RuntimeEvent,
        acc: _RunAccumulator,
        *,
        status: str,
        reply: object = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            self._active.pop(event.run_id, None)
        record = RunLogRecord(
            run_id=event.run_id,
            session_id=acc.session_id,
            turn_index=acc.turn_index,
            timestamp=acc.timestamp,
            ended_at=utc_now(),
            status=status,  # type: ignore[arg-type]
            model=acc.model,
            user_input_preview=acc.user_input_preview,
            duration_ms=int((time.perf_counter() - acc.started_perf) * 1000),
            did_compact=acc.did_compact,
            compact_message="\n".join(acc.compact_messages) or None,
            input_tokens=acc.input_tokens if acc.saw_usage else None,
            output_tokens=acc.output_tokens if acc.saw_usage else None,
            total_tokens=acc.total_tokens if acc.saw_usage else None,
            llm_call_count=acc.llm_call_count,
            tool_call_count=acc.tool_call_count,
            tools_used=list(acc.tools_used),
            mcp_tool_call_count=acc.mcp_tool_call_count,
            mcp_servers_used=sorted(acc.mcp_servers_used),
            mcp_transports_used=sorted(acc.mcp_transports_used),
            mcp_error_count=acc.mcp_error_count,
            final_reply_preview=(
                None if reply is None else preview_text(str(reply), 200)
            ),
            error_type=error_type,
            error_message_preview=(
                None if error_message is None else preview_text(error_message, 200)
            ),
        )
        self.store.append(record)

    @staticmethod
    def _add_usage(acc: _RunAccumulator, usage: Any) -> None:
        if not isinstance(usage, dict):
            # A call without provider usage does not poison the sum.
            return
        first = not acc.saw_usage
        acc.saw_usage = True
        acc.input_tokens = _sum_optional(
            acc.input_tokens, usage.get("input_tokens"), first=first
        )
        acc.output_tokens = _sum_optional(
            acc.output_tokens, usage.get("output_tokens"), first=first
        )
        acc.total_tokens = _sum_optional(
            acc.total_tokens, usage.get("total_tokens"), first=first
        )

    def _add_tool_call(self, acc: _RunAccumulator, payload: dict[str, Any]) -> None:
        acc.tool_call_count += 1
        name = str(payload.get("tool") or "")
        if name:
            acc.tools_used.append(name)
        if payload.get("source") != "mcp":
            return
        acc.mcp_tool_call_count += 1
        if payload.get("ok") is False:
            acc.mcp_error_count += 1
        if self.tool_registry is None:
            return
        tool = self.tool_registry.get(name)
        if tool is None:
            return
        server_name = getattr(tool, "server_name", None)
        if isinstance(server_name, str) and server_name:
            acc.mcp_servers_used.add(server_name)
        transport_type = getattr(tool, "transport_type", None)
        if isinstance(transport_type, str) and transport_type:
            acc.mcp_transports_used.add(transport_type)


def _sum_optional(
    accumulated: int | None,
    current: object,
    *,
    first: bool,
) -> int | None:
    """Sum usage fields the way the loop does: any missing value poisons the sum."""
    value = current if isinstance(current, int) else None
    if first:
        return value
    if accumulated is None or value is None:
        return None
    return accumulated + value
