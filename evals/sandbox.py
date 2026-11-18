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
  results, or look like a silent success / ordinary network failure. ``exec``
  takes ``{"stdout", "stderr", "exit_code"}`` (a non-zero code fails the call)
  or a ``{"sequence": [...]}`` of them, like the macOS tools. Stand-ins
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

from ..bootstrap import MiniBotRuntime, build_runtime
from ..config import Config
from ..llm import LLMClient, LLMResponse
from ..mcp_host.models import MCPToolSpec
from ..mcp_host.provider import MCPToolProxy, _result_to_tool_output
from ..prompts import MEMORY_INSTRUCTIONS, SUMMARY_SYSTEM_PROMPT
from ..runtime.events import RuntimeEvent
from ..runtime.messages import ModelMessage
from ..tools.definitions import ModelToolDefinition
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

        runtime = _sandbox_runtime(
            config,
            workspace=workspace,
            home=home,
            approve=case_input.get("approval", "approve") == "approve",
            llm=llm,
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

        run_id = events[0].run_id if events else None
        trace_id = None
        if run_id and runtime.langfuse is not None:
            # The real trace: inside an experiment or a pytest span the turn
            # joins the trace that is already open.
            trace_id = runtime.langfuse.trace_id_of(run_id)
        return {
            "reply": reply,
            "error": error,
            "elapsed_s": round(time.perf_counter() - started, 2),
            **summarize_events(events),
            "files": _snapshot_files(workspace),
            "run_id": run_id,
            "trace_id": trace_id,
        }


def _sandbox_runtime(
    config: Config,
    *,
    workspace: Path,
    home: Path,
    approve: bool,
    llm: LLMClient | None,
    tool_results: ToolResults,
) -> MiniBotRuntime:
    runtime = build_runtime(
        # Evals decide approvals themselves; never inherit "always".
        config=replace(config, approval_mode="ask"),
        workspace=workspace,
        state_home=home,
        enable_mcp=False,
        approval_handler=lambda request, cancel_event: approve,
        llm=llm,
    )
    install_sandbox_tools(runtime.tool_registry, tool_results=tool_results)
    return runtime


# ── what the model is shown, part by part ─────────────────────────

# Sent with every request; the summary prompt only when the history is compacted.
ALWAYS_SENT = ("prompt:system", "prompt:memory")
SUMMARY_PROMPT = "prompt:summary"


def model_facing_parts(config: Config) -> dict[str, str]:
    """Part name -> the text the model is shown for it.

    Parts are the prompts (``prompt:system``, ``prompt:memory``,
    ``prompt:summary``), every skill (``skill:<name>``: description, tools
    and body) and every tool definition (``tool:<name>``: description and
    parameters). They are read from a sandbox runtime built the way
    ``run_case`` builds one, so the tools are the ones a case sees. No model
    is called, and no API key is needed.

    Not a part: the section headings ``ContextBuilder`` writes around them.
    A change there shows only as a different git revision.
    """
    with tempfile.TemporaryDirectory(prefix="minibot-eval-") as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        runtime = _sandbox_runtime(
            # Lessons are recorded per run already, and loading them is slow.
            replace(config, lessons_mode="off"),
            workspace=workspace,
            home=Path(tmp) / "home",
            approve=False,
            llm=_NoModel(),
            tool_results={},
        )
        try:
            parts = {
                "prompt:system": runtime.context_builder.base_system_prompt,
                "prompt:memory": MEMORY_INSTRUCTIONS,
                SUMMARY_PROMPT: SUMMARY_SYSTEM_PROMPT,
            }
            for skill in runtime.skill_registry.list():
                parts[f"skill:{skill.name}"] = _canonical(
                    {"description": skill.description, "tools": list(skill.tools), "body": skill.body}
                )
            for definition in runtime.tool_registry.get_definitions():
                parts[f"tool:{definition.name}"] = _canonical(
                    {"description": definition.description, "parameters": definition.parameters}
                )
        finally:
            runtime.mcp_host.close()
    return parts


def parts_touched(output: dict[str, Any]) -> set[str]:
    """The parts one run's output shows the model used.

    The prompts that go with every request, plus the skills it read, the
    tools it called, and the summary prompt if the history was compacted.
    Every skill description and tool definition is in every request, so
    this is where to look first, not everything that could matter.
    """
    touched = set(ALWAYS_SENT)
    if output.get("compactions"):
        touched.add(SUMMARY_PROMPT)
    touched.update(f"skill:{name}" for name in output.get("skills_read") or [])
    touched.update(
        f"tool:{call.get('name')}" for call in output.get("tool_calls") or [] if call.get("name")
    )
    return touched


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class _NoModel(LLMClient):
    """Fills the runtime's model slot when only its text is read."""

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        raise RuntimeError("这个 runtime 只用来读提示和工具定义，不调用模型。")


def summarize_events(events: list[RuntimeEvent]) -> dict[str, Any]:
    """Reduce a run's events to what the scorers need."""
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    model_calls = 0
    compactions = 0
    input_tokens = output_tokens = 0
    lessons_used: list[str] = []
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
        elif event.type == "model.request.started":
            for lesson_id in payload.get("lessons") or []:
                if lesson_id not in lessons_used:
                    lessons_used.append(str(lesson_id))
        elif event.type in {"model.request.completed", "compaction.request.completed"}:
            model_calls += 1
            if event.type == "compaction.request.completed":
                compactions += 1
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
        "compactions": compactions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "lessons_used": lessons_used,
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
        self._calls = 0

    def execute(self, *, context: ToolExecutionContext, command: str, **kwargs: Any) -> ToolOutput:
        del context, kwargs
        canned = self._results.get(self.name)
        if isinstance(canned, dict) and "sequence" in canned:
            sequence = canned["sequence"]
            canned = sequence[min(self._calls, len(sequence) - 1)]
        self._calls += 1
        if isinstance(canned, dict):
            data = {"command": command, "exit_code": 0, **canned}
            code = data["exit_code"]
            if code != 0:
                # Same summary as the real tool, so a replayed failure reads the same.
                return ToolOutput.failure("error", f"命令执行失败，退出码 {code}。", data=data)
            return ToolOutput.success(f"命令已执行，退出码 {code}。", data=data)
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
