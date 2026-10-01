"""Run one real minibot turn inside an isolated sandbox.

Evals drive the production runtime (same system prompt, skills catalog,
loop, compaction and tool schemas) against a real model, so anything with
side effects is swapped for a stand-in first:

- state (sessions, memory, schedule) lives in a temp ``MINIBOT_HOME``;
- file tools are real but rooted at a temp workspace seeded per case;
- macOS calendar/reminders/notes/mail tools are stand-ins built from the real
  server's own tool schemas, returning canned results instead of touching apps.
  A case can script ``{"sequence": [...]}`` (later calls reuse the last entry;
  ``{"error": "..."}`` fails that call) to exercise error recovery;
- ``exec``, ``web_search`` and ``fetch_url`` never run; they return canned
  results, or look like a silent success / ordinary network failure. Stand-ins
  never mention the sandbox: a model that knows it is being tested behaves
  differently.

What happened is read back from the run's event stream, the same stream the
UIs and tracing subscribe to.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from mcp.types import CallToolResult, TextContent

from ..bootstrap import build_runtime
from ..config import Config
from ..llm import LLMClient
from ..mcp_host.models import MCPToolSpec
from ..mcp_host.provider import MCPToolProxy, _result_to_tool_output
from ..runtime.events import RuntimeEvent
from ..tools.base import ToolExecutionContext
from ..tools.exec_cmd import ExecTool
from ..tools.fetch_url import FetchUrlTool
from ..tools.registry import ToolRegistry
from ..tools.result import ToolOutput
from ..tools.web_search import WebSearchTool
from ..user_memory import UserMemoryStore

_MACOS_SERVER = "macos_system"
_MAX_SNAPSHOT_FILE_CHARS = 20_000

ToolResults = dict[str, Any]


def run_case(
    case_input: dict[str, Any],
    *,
    config: Config,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Run one case's prompt as a single turn; return a compact transcript."""
    with tempfile.TemporaryDirectory(prefix="minibot-eval-") as tmp:
        home = Path(tmp) / "home"
        workspace = Path(tmp) / "workspace"
        workspace.mkdir(parents=True)
        _seed_files(workspace, case_input.get("files") or {})
        memory = UserMemoryStore(home)
        for fact in case_input.get("memory") or []:
            memory.add(fact)

        approve = case_input.get("approval", "approve") == "approve"
        runtime = build_runtime(
            # Evals decide approvals themselves; never inherit "always".
            config=replace(config, approval_mode="ask"),
            workspace=workspace,
            state_home=home,
            enable_mcp=False,
            approval_handler=lambda request, cancel_event: approve,
        )
        if llm is not None:
            runtime.agent_loop.llm = llm
        install_sandbox_tools(
            runtime.tool_registry,
            tool_results=case_input.get("tool_results") or {},
        )

        events: list[RuntimeEvent] = []
        started = time.perf_counter()
        reply: str | None = None
        error: str | None = None
        try:
            outcome = runtime.agent_session.prompt(
                None,
                str(case_input["prompt"]),
                event_handler=events.append,
                source="eval",
            )
            reply = outcome.reply
        except Exception as exc:  # A failed run is a result to score, not a crash.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Not runtime.close(): that shuts down the process-wide Langfuse
            # client, which the experiment runner still needs.
            runtime.mcp_host.close()

        return {
            "reply": reply,
            "error": error,
            "elapsed_s": round(time.perf_counter() - started, 2),
            **summarize_events(events),
            "files": _snapshot_files(workspace),
        }


def summarize_events(events: list[RuntimeEvent]) -> dict[str, Any]:
    """Reduce a run's events to what the scorers need."""
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    model_calls = 0
    input_tokens = output_tokens = 0
    for event in events:
        payload = event.payload
        call_id = str(payload.get("tool_call_id") or "")
        if event.type == "tool_call.started":
            calls[call_id] = {
                "name": payload.get("tool"),
                "args": payload.get("args") or {},
                "ok": None,
                "code": None,
                "approved": None,
            }
            order.append(call_id)
        elif event.type in {"tool_call.completed", "tool_call.failed"} and call_id in calls:
            calls[call_id].update(ok=payload.get("ok"), code=payload.get("code"))
        elif event.type == "approval.resolved" and call_id in calls:
            calls[call_id]["approved"] = payload.get("approved")
        elif event.type in {"model.request.completed", "compaction.request.completed"}:
            model_calls += 1
            usage = payload.get("usage") or {}
            input_tokens += usage.get("input_tokens") or 0
            output_tokens += usage.get("output_tokens") or 0
    tool_calls = [calls[call_id] for call_id in order]
    return {
        "tool_calls": tool_calls,
        "skills_read": [
            str(call["args"].get("name"))
            for call in tool_calls
            if call["name"] == "read_skill"
        ],
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ── sandbox tools ─────────────────────────────────────────────────


def install_sandbox_tools(registry: ToolRegistry, *, tool_results: ToolResults) -> None:
    """Swap side-effecting tools for stand-ins with identical model-facing shape."""
    registry.register(_SandboxExec(workspace=None, results=tool_results))
    registry.register(_SandboxWebSearch(results=tool_results))
    registry.register(_SandboxFetchUrl(results=tool_results))
    for spec in macos_tool_specs():
        registry.register(_SandboxMCPTool(spec, results=tool_results))


@lru_cache(maxsize=1)
def macos_tool_specs() -> tuple[MCPToolSpec, ...]:
    """The real macOS server's tool schemas, read from its own definitions."""
    from ..mcp_servers.macos_system.server import build_macos_server

    # The bridge is only touched when a tool runs; listing needs none.
    app = build_macos_server(bridge=object())  # type: ignore[arg-type]
    # Own thread: the experiment runner calls tasks inside a running loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        tools = pool.submit(asyncio.run, app.list_tools()).result()
    return tuple(
        MCPToolSpec(
            server_name=_MACOS_SERVER,
            remote_name=tool.name,
            title=tool.title,
            # Same fallback chain as the real transport's _convert_tool.
            description=tool.description or tool.title or tool.name,
            input_schema=dict(tool.inputSchema),
        )
        for tool in tools
    )


class _SandboxMCPTool(MCPToolProxy):
    """Same name, schema and approval rule as the real proxy; canned results."""

    def __init__(self, spec: MCPToolSpec, *, results: ToolResults) -> None:
        super().__init__(client=None, tool_spec=spec, trusted=False)  # type: ignore[arg-type]
        self._results = results
        self._calls = 0

    @property
    def transport_type(self) -> str:
        return "sandbox"

    def execute(self, *, context: ToolExecutionContext, **kwargs: Any) -> ToolOutput:
        del context
        payload = self._results.get(self.name)
        if isinstance(payload, dict) and "sequence" in payload:
            sequence = payload["sequence"]
            payload = sequence[min(self._calls, len(sequence) - 1)]
        self._calls += 1
        if isinstance(payload, dict) and "error" in payload:
            result = CallToolResult(
                content=[TextContent(type="text", text=str(payload["error"]))],
                isError=True,
            )
            return _result_to_tool_output(self._tool_spec, result)
        if payload is None:
            payload = {"ok": True, **kwargs}
        result = CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structuredContent=payload,
            isError=False,
        )
        # Reuse the real conversion so the model sees the real result shape.
        return _result_to_tool_output(self._tool_spec, result)


class _SandboxExec(ExecTool):
    def __init__(self, *, workspace: Path | None, results: ToolResults) -> None:
        super().__init__(workspace=workspace)
        self._results = results

    def execute(self, *, context: ToolExecutionContext, command: str, **kwargs: Any) -> ToolOutput:
        del context, kwargs
        canned = self._results.get(self.name)
        if isinstance(canned, dict):
            return ToolOutput.success(
                "命令已执行，退出码 0。",
                data={"command": command, "exit_code": 0, **canned},
            )
        # Like a command that succeeds silently (e.g. `open -a Reminders`).
        return ToolOutput.success(
            "命令已执行，退出码 0。",
            data={"command": command, "exit_code": 0, "stdout": "", "stderr": ""},
        )


class _SandboxWebSearch(WebSearchTool):
    def __init__(self, *, results: ToolResults) -> None:
        super().__init__(workspace=None)
        self._results = results

    def execute(self, *, context: ToolExecutionContext, query: str = "", **kwargs: Any) -> ToolOutput:
        del context
        items = self._results.get(self.name) or []
        return ToolOutput.success(
            f"找到 {len(items)} 条网页结果。",
            data={"query": query, "allowed_domains": [], "results": items},
        )


class _SandboxFetchUrl(FetchUrlTool):
    def __init__(self, *, results: ToolResults) -> None:
        super().__init__()
        self._results = results

    def execute(self, *, context: ToolExecutionContext, url: str, **kwargs: Any) -> ToolOutput:
        del context, kwargs
        pages = self._results.get(self.name) or {}
        page = pages.get(url) if isinstance(pages, dict) else None
        if page is None:
            # Look like an ordinary fetch failure: a message that reveals the
            # sandbox changes how the model behaves.
            return ToolOutput.failure(
                "error", "获取页面失败: HTTP 503 Service Unavailable", data={"url": url}
            )
        return ToolOutput.success("已获取页面。", data={"url": url, "content": page})


# ── workspace helpers ─────────────────────────────────────────────


def _seed_files(workspace: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace.resolve()):
            raise ValueError(f"用例文件路径越界: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _snapshot_files(workspace: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        snapshot[str(path.relative_to(workspace))] = text[:_MAX_SNAPSHOT_FILE_CHARS]
    return snapshot
