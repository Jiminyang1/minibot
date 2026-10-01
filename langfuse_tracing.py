"""Langfuse tracing as one more subscriber of the runtime event stream.

Like ``RunLogFold``, this reduces a run's events — into Langfuse observations
instead of a ``runs.jsonl`` row:

    run.started … run.completed/failed/cancelled → one trace, rooted at an
                                                   ``agent`` observation
    model.request.started → .completed           → ``generation`` (request,
                                                   output, usage, first-token time)
    compaction.request.started → .completed      → ``generation`` for the summary
    tool_call.started → .completed/.failed       → ``tool``
    approvals, compaction, retries               → ``event`` markers

Langfuse v4 reads trace attributes (session id, trace name, tags) from the
OpenTelemetry context when a span starts, and a child created outside that
context does not inherit them. So every observation is started inside its own
short ``propagate_attributes`` block rather than one block spanning the run.

Tracing is best effort: a failure here is swallowed and never reaches the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
import threading
from typing import Any

from .runtime.events import RuntimeEvent

_TRACE_NAME = "minibot.turn"
_DEFAULT_MAX_INPUT_MESSAGES = 50
_DISABLED_VALUES = {"0", "false", "no", "off"}


@dataclass
class _RunTrace:
    root: Any
    attributes: dict[str, Any]
    # "model:<iteration>" / "compaction:<n>" -> open generation
    generations: dict[str, Any] = field(default_factory=dict)
    first_token_seen: set[str] = field(default_factory=set)
    # tool_call_id -> open tool observation
    tools: dict[str, Any] = field(default_factory=dict)
    compaction_count: int = 0


class LangfuseFold:
    """Reduce each run's events into one Langfuse trace."""

    def __init__(
        self,
        client: Any,
        *,
        propagate_attributes: Callable[..., Any],
        max_input_messages: int = _DEFAULT_MAX_INPUT_MESSAGES,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.max_input_messages = max_input_messages
        self._propagate_attributes = propagate_attributes
        self._log = log
        self._runs: dict[str, _RunTrace] = {}
        self._lock = threading.Lock()
        self._warned = False

    def __call__(self, event: RuntimeEvent) -> None:
        try:
            self._handle(event)
        except Exception as exc:  # Observability must never fail a run.
            if not self._warned and self._log is not None:
                self._warned = True
                self._log(f"Langfuse 追踪出错，已忽略（后续不再提示）: {exc!r}")

    def shutdown(self) -> None:
        """Flush buffered observations; call once at process exit."""
        try:
            self.client.shutdown()
        except Exception:
            pass

    # ── event handling ────────────────────────────────────────────

    def _handle(self, event: RuntimeEvent) -> None:
        payload = event.payload
        if event.type == "run.started":
            self._start_run(event)
            return

        with self._lock:
            run = self._runs.get(event.run_id)
        if run is None:
            return

        kind = event.type
        if kind == "context.usage":
            run.root.update(metadata={"context": payload})
        elif kind == "model.request.started":
            key = f"model:{payload.get('iteration')}"
            run.generations[key] = self._start(
                run,
                run.root,
                name="model",
                as_type="generation",
                model=payload.get("model"),
                input=self._trim_messages(payload.get("messages")),
                metadata={
                    "iteration": payload.get("iteration"),
                    "tools": payload.get("tools"),
                },
            )
        elif kind == "message.delta":
            key = f"model:{payload.get('iteration')}"
            generation = run.generations.get(key)
            if generation is not None and key not in run.first_token_seen:
                run.first_token_seen.add(key)
                generation.update(completion_start_time=datetime.now(UTC))
        elif kind == "model.request.retrying":
            parent = run.generations.get(f"model:{payload.get('iteration')}", run.root)
            self._event(run, parent, "model.retrying", payload, level="WARNING")
        elif kind == "model.request.completed":
            generation = run.generations.pop(f"model:{payload.get('iteration')}", None)
            if generation is not None:
                empty = bool(payload.get("empty_reply"))
                generation.update(
                    output=payload.get("output"),
                    usage_details=_usage_details(payload.get("usage")),
                    metadata={
                        "elapsed_ms": payload.get("elapsed_ms"),
                        "tool_call_count": payload.get("tool_call_count"),
                    },
                    level="ERROR" if empty else None,
                    status_message="模型返回空回复" if empty else None,
                )
                generation.end()
        elif kind == "compaction.request.started":
            run.compaction_count += 1
            run.generations[f"compaction:{run.compaction_count}"] = self._start(
                run,
                run.root,
                name="compaction.summary",
                as_type="generation",
                model=payload.get("model"),
                input=self._trim_messages(payload.get("messages")),
            )
        elif kind == "compaction.request.completed":
            generation = run.generations.pop(
                f"compaction:{run.compaction_count}", None
            )
            if generation is not None:
                error = payload.get("error_type")
                generation.update(
                    output=payload.get("output"),
                    usage_details=_usage_details(payload.get("usage")),
                    metadata={"elapsed_ms": payload.get("elapsed_ms")},
                    level="ERROR" if error else None,
                    status_message=(
                        f"{error}: {payload.get('message')}" if error else None
                    ),
                )
                generation.end()
        elif kind == "context.compacted":
            self._event(run, run.root, "context.compacted", payload)
        elif kind == "tool_call.started":
            run.tools[str(payload.get("tool_call_id"))] = self._start(
                run,
                run.root,
                name=str(payload.get("tool") or "tool"),
                as_type="tool",
                input=payload.get("args"),
                metadata={
                    "tool_call_id": payload.get("tool_call_id"),
                    "source": payload.get("source"),
                    "requires_approval": payload.get("requires_approval"),
                },
            )
        elif kind in {"approval.required", "approval.resolved"}:
            parent = run.tools.get(str(payload.get("tool_call_id")), run.root)
            denied = kind == "approval.resolved" and not payload.get("approved")
            self._event(
                run, parent, kind, payload, level="WARNING" if denied else None
            )
        elif kind in {"tool_call.completed", "tool_call.failed"}:
            tool = run.tools.pop(str(payload.get("tool_call_id")), None)
            if tool is not None:
                failed = kind == "tool_call.failed"
                tool.update(
                    output={
                        "ok": payload.get("ok"),
                        "code": payload.get("code"),
                        "summary": payload.get("summary"),
                        "artifact": payload.get("artifact"),
                        "truncated": payload.get("truncated"),
                    },
                    level="ERROR" if failed else None,
                    status_message=payload.get("summary") if failed else None,
                )
                tool.end()
        elif kind == "message.completed":
            hit_limit = payload.get("reason") == "max_iterations"
            run.root.update(
                output=payload.get("content"),
                level="WARNING" if hit_limit else None,
                status_message="达到工具调用轮次上限" if hit_limit else None,
            )
        elif kind == "run.completed":
            run.root.update(
                output=payload.get("reply"),
                metadata={"did_compact": payload.get("did_compact")},
            )
            self._finish(event.run_id)
        elif kind == "run.failed":
            run.root.update(
                level="ERROR",
                status_message=f"{payload.get('error_type')}: {payload.get('message')}",
            )
            self._finish(event.run_id)
        elif kind == "run.cancelled":
            run.root.update(level="WARNING", status_message="用户取消")
            self._finish(event.run_id)

    def _start_run(self, event: RuntimeEvent) -> None:
        payload = event.payload
        source = payload.get("source")
        attributes: dict[str, Any] = {
            "session_id": event.session_id,
            "trace_name": _TRACE_NAME,
        }
        if source:
            attributes["tags"] = [str(source)]
        run = _RunTrace(root=None, attributes=attributes)
        run.root = self._start(
            run,
            None,
            name=_TRACE_NAME,
            as_type="agent",
            input=payload.get("input") or payload.get("input_preview"),
            metadata={
                "run_id": event.run_id,
                "turn_index": payload.get("turn_index"),
                "model": payload.get("model"),
                "source": source,
            },
        )
        with self._lock:
            self._runs[event.run_id] = run

    def _finish(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return
        # Anything still open was cut off by the run's end (error/cancel).
        for observation in [*run.generations.values(), *run.tools.values()]:
            observation.update(level="WARNING", status_message="运行结束时仍未完成")
            observation.end()
        run.root.end()

    # ── Langfuse helpers ──────────────────────────────────────────

    def _start(self, run: _RunTrace, parent: Any, **kwargs: Any) -> Any:
        with self._propagate_attributes(**run.attributes):
            if parent is None:
                return self.client.start_observation(**kwargs)
            return parent.start_observation(**kwargs)

    def _event(
        self,
        run: _RunTrace,
        parent: Any,
        name: str,
        payload: dict[str, Any],
        *,
        level: str | None = None,
    ) -> None:
        # create_event parents the event under *parent*; start_observation
        # with as_type="event" does not (it would start a separate trace).
        with self._propagate_attributes(**run.attributes):
            parent.create_event(name=name, metadata=payload, level=level)

    def _trim_messages(self, messages: Any) -> Any:
        """Keep leading system messages plus the latest N; mark the gap."""
        if not isinstance(messages, list) or self.max_input_messages <= 0:
            return messages
        head = 0
        while head < len(messages) and _role(messages[head]) == "system":
            head += 1
        rest = messages[head:]
        omitted = len(rest) - self.max_input_messages
        if omitted <= 0:
            return messages
        marker = {
            "role": "system",
            "content": f"[minibot: 为控制上报体积，省略了 {omitted} 条较早的消息]",
        }
        return [*messages[:head], marker, *rest[omitted:]]


def build_langfuse_fold(
    *,
    log: Callable[[str], None] | None = None,
) -> LangfuseFold | None:
    """Enable tracing when Langfuse keys are configured; otherwise ``None``.

    Credentials and endpoint come from the SDK's own ``LANGFUSE_PUBLIC_KEY``,
    ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_BASE_URL``. ``MINIBOT_LANGFUSE=0``
    turns tracing off; ``MINIBOT_LANGFUSE_MAX_INPUT_MESSAGES`` caps how many
    history messages each generation uploads (0 = all).
    """
    if os.environ.get("MINIBOT_LANGFUSE", "").strip().lower() in _DISABLED_VALUES:
        return None
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    ):
        return None
    raw_limit = os.environ.get("MINIBOT_LANGFUSE_MAX_INPUT_MESSAGES", "").strip()
    try:
        max_input_messages = (
            int(raw_limit) if raw_limit else _DEFAULT_MAX_INPUT_MESSAGES
        )
    except ValueError as exc:
        raise ValueError("MINIBOT_LANGFUSE_MAX_INPUT_MESSAGES 必须是整数。") from exc
    try:
        from langfuse import Langfuse, propagate_attributes
    except ImportError:
        if log is not None:
            log("已配置 Langfuse key，但未安装 langfuse 包，追踪未启用。")
        return None
    return LangfuseFold(
        Langfuse(),
        propagate_attributes=propagate_attributes,
        max_input_messages=max_input_messages,
        log=log,
    )


def _role(message: Any) -> Any:
    return message.get("role") if isinstance(message, dict) else None


def _usage_details(usage: Any) -> dict[str, int] | None:
    """Map minibot usage onto Langfuse usage types.

    Cached prompt tokens are split out of ``input`` so a model price with a
    separate ``input_cached_tokens`` rate is not charged twice.
    """
    if not isinstance(usage, dict):
        return None
    details: dict[str, int] = {}
    input_tokens = usage.get("input_tokens")
    cached = usage.get("cached_input_tokens")
    if isinstance(input_tokens, int):
        if isinstance(cached, int) and cached > 0:
            details["input"] = max(input_tokens - cached, 0)
            details["input_cached_tokens"] = cached
        else:
            details["input"] = input_tokens
    if isinstance(usage.get("output_tokens"), int):
        details["output"] = usage["output_tokens"]
    if isinstance(usage.get("total_tokens"), int):
        details["total"] = usage["total_tokens"]
    return details or None
