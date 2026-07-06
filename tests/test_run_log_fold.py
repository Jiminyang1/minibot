from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from minibot.mcp_host.models import MCPToolSpec
from minibot.mcp_host.provider import MCPToolProxy
from minibot.run_log import RunLogStore
from minibot.runtime.agent_session import AgentSession
from minibot.runtime.events import RuntimeEvent, fanout
from minibot.runtime.messages import ModelMessage
from minibot.runtime.run_log_fold import RunLogFold
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.registry import ToolRegistry

from loop_harness import build_loop

from mcp.types import CallToolResult, TextContent


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        del messages, tools, model
        if not self._responses:
            raise RuntimeError("llm unavailable")
        return self._responses.pop(0)


class _FakeMCPClient:
    def __init__(self, transport_type: str = "streamable_http") -> None:
        self.config = type(
            "_Config",
            (),
            {"transport": type("_Transport", (), {"type": transport_type})()},
        )()

    def call_tool(self, remote_name: str, arguments: dict[str, object]) -> CallToolResult:
        del remote_name, arguments
        return CallToolResult(
            content=[TextContent(type="text", text="boom")],
            isError=True,
        )


def _agent_session(loop, manager, workspace: Path, registry: ToolRegistry) -> AgentSession:
    fold = RunLogFold(RunLogStore(workspace), tool_registry=registry)
    return AgentSession(
        agent_loop=loop,
        session_manager=manager,
        base_event_handler=fold,
    )


def _read_run_logs(workspace: Path) -> list[dict[str, object]]:
    path = workspace / "runs.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RunLogFoldTests(unittest.TestCase):
    def test_successful_turn_folds_events_into_summary_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            registry = ToolRegistry()
            from minibot.tools.base import Tool, ToolExecutionContext
            from minibot.tools.result import ToolOutput

            class _ReadFileLike(Tool):
                @property
                def name(self) -> str:
                    return "read_file"

                @property
                def description(self) -> str:
                    return "read"

                @property
                def parameters(self) -> dict[str, object]:
                    return {"type": "object", "properties": {}}

                def execute(self, *, context: ToolExecutionContext, **kwargs) -> ToolOutput:
                    del context, kwargs
                    return ToolOutput.success("ok")

            registry.register(_ReadFileLike())
            user_input = ("hello   world " * 20).strip()
            reply = ("final   answer " * 30).strip()
            llm = _ScriptedLLM(
                [
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="call_1", name="read_file", arguments="{}")
                        ],
                        usage=TokenUsage(
                            input_tokens=100, output_tokens=9, total_tokens=109
                        ),
                    ),
                    LLMResponse(
                        content=reply,
                        usage=TokenUsage(
                            input_tokens=11, output_tokens=20, total_tokens=31
                        ),
                    ),
                ]
            )
            loop, manager = build_loop(llm, registry, workspace)
            manager.create_session("s_test")
            agent_session = _agent_session(loop, manager, workspace, registry)

            result = agent_session.prompt("s_test", user_input)

            self.assertFalse(result.did_compact)
            logs = _read_run_logs(workspace)
            self.assertEqual(len(logs), 1)
            log = logs[0]
            self.assertEqual(log["session_id"], "s_test")
            self.assertEqual(log["turn_index"], 1)
            self.assertEqual(log["status"], "success")
            self.assertEqual(log["did_compact"], False)
            self.assertEqual(log["tool_call_count"], 1)
            self.assertEqual(log["llm_call_count"], 2)
            self.assertEqual(log["tools_used"], ["read_file"])
            self.assertEqual(log["mcp_tool_call_count"], 0)
            self.assertEqual(log["mcp_servers_used"], [])
            self.assertEqual(log["mcp_transports_used"], [])
            self.assertEqual(log["mcp_error_count"], 0)
            self.assertIsNone(log["compact_message"])
            self.assertEqual(log["model"], "gpt-5.4-mini")
            self.assertEqual(log["input_tokens"], 111)
            self.assertEqual(log["output_tokens"], 29)
            self.assertEqual(log["total_tokens"], 140)
            self.assertIsNone(log["error_type"])
            self.assertIsNone(log["error_message_preview"])
            self.assertLessEqual(len(str(log["user_input_preview"])), 120)
            self.assertLessEqual(len(str(log["final_reply_preview"])), 200)
            self.assertNotEqual(log["user_input_preview"], user_input)
            self.assertNotEqual(log["final_reply_preview"], reply)
            self.assertGreaterEqual(int(log["duration_ms"]), 0)

            session_dir = workspace / "sessions" / "s_test"
            self.assertTrue((session_dir / "messages.jsonl").exists())
            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 4)

    def test_run_log_records_mcp_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            registry = ToolRegistry()
            registry.register(
                MCPToolProxy(
                    client=_FakeMCPClient("stdio"),
                    tool_spec=MCPToolSpec(
                        server_name="sqlite",
                        remote_name="query",
                        title=None,
                        description="query",
                        input_schema={"type": "object", "properties": {}},
                    ),
                    trusted=True,
                )
            )
            llm = _ScriptedLLM(
                [
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="mcp__sqlite__query",
                                arguments="{}",
                            )
                        ],
                    ),
                    LLMResponse(content="done"),
                ]
            )
            loop, manager = build_loop(llm, registry, workspace)
            manager.create_session("s_test")
            agent_session = _agent_session(loop, manager, workspace, registry)

            agent_session.prompt("s_test", "query db")

            log = _read_run_logs(workspace)[0]
            self.assertEqual(log["mcp_tool_call_count"], 1)
            self.assertEqual(log["mcp_servers_used"], ["sqlite"])
            self.assertEqual(log["mcp_transports_used"], ["stdio"])
            self.assertEqual(log["mcp_error_count"], 1)

    def test_failed_turn_before_any_model_call_writes_failed_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            registry = ToolRegistry()
            llm = _ScriptedLLM([])
            loop, manager = build_loop(llm, registry, workspace)
            manager.create_session("s_test")
            agent_session = _agent_session(loop, manager, workspace, registry)

            with self.assertRaisesRegex(RuntimeError, "llm unavailable"):
                agent_session.prompt("s_test", "hello there")

            logs = _read_run_logs(workspace)
            self.assertEqual(len(logs), 1)
            log = logs[0]
            self.assertEqual(log["status"], "failed")
            self.assertEqual(log["error_type"], "RuntimeError")
            self.assertEqual(log["error_message_preview"], "llm unavailable")
            self.assertIsNone(log["input_tokens"])
            self.assertIsNone(log["output_tokens"])
            self.assertIsNone(log["total_tokens"])
            self.assertEqual(log["llm_call_count"], 0)
            self.assertEqual(log["tool_call_count"], 0)
            self.assertEqual(log["tools_used"], [])
            self.assertIsNone(log["final_reply_preview"])

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 1)
            self.assertEqual(reloaded.messages[0].role, "user")

    def test_partial_failure_keeps_usage_from_completed_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            registry = ToolRegistry()
            from minibot.tools.base import Tool, ToolExecutionContext
            from minibot.tools.result import ToolOutput

            class _ReadFileLike(Tool):
                @property
                def name(self) -> str:
                    return "read_file"

                @property
                def description(self) -> str:
                    return "read"

                @property
                def parameters(self) -> dict[str, object]:
                    return {"type": "object", "properties": {}}

                def execute(self, *, context: ToolExecutionContext, **kwargs) -> ToolOutput:
                    del context, kwargs
                    return ToolOutput.success("ok")

            registry.register(_ReadFileLike())
            llm = _ScriptedLLM(
                [
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="call_1", name="read_file", arguments="{}")
                        ],
                        usage=TokenUsage(
                            input_tokens=100, output_tokens=10, total_tokens=110
                        ),
                    ),
                ]
            )
            loop, manager = build_loop(llm, registry, workspace)
            manager.create_session("s_test")
            agent_session = _agent_session(loop, manager, workspace, registry)

            with self.assertRaisesRegex(RuntimeError, "llm unavailable"):
                agent_session.prompt("s_test", "trigger partial failure")

            log = _read_run_logs(workspace)[0]
            self.assertEqual(log["status"], "failed")
            self.assertEqual(log["error_type"], "RuntimeError")
            self.assertEqual(log["input_tokens"], 100)
            self.assertEqual(log["output_tokens"], 10)
            self.assertEqual(log["total_tokens"], 110)
            self.assertEqual(log["llm_call_count"], 1)
            self.assertEqual(log["tool_call_count"], 1)
            self.assertEqual(log["tools_used"], ["read_file"])

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(
                [message.role for message in reloaded.messages],
                ["user", "assistant", "tool"],
            )
            self.assertEqual(reloaded.messages[2].name, "read_file")

    def test_fold_and_caller_handler_see_the_same_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            registry = ToolRegistry()
            llm = _ScriptedLLM([LLMResponse(content="ok")])
            loop, manager = build_loop(llm, registry, workspace)
            manager.create_session("s_test")
            fold = RunLogFold(RunLogStore(workspace), tool_registry=registry)
            caller_events: list[RuntimeEvent] = []
            agent_session = AgentSession(
                agent_loop=loop,
                session_manager=manager,
                base_event_handler=fanout(fold, None),
            )

            agent_session.prompt(
                "s_test", "hello", event_handler=caller_events.append
            )

            self.assertEqual(caller_events[0].type, "run.started")
            self.assertEqual(caller_events[0].payload["model"], "gpt-5.4-mini")
            self.assertEqual(caller_events[0].payload["turn_index"], 1)
            self.assertEqual(caller_events[-1].type, "run.completed")
            log = _read_run_logs(workspace)[0]
            self.assertEqual(log["status"], "success")
            self.assertEqual(log["final_reply_preview"], "ok")

    def test_context_usage_event_reports_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            registry = ToolRegistry()
            llm = _ScriptedLLM([LLMResponse(content="ok")])
            loop, manager = build_loop(
                llm,
                registry,
                workspace,
                compact_token_threshold=50_000,
                reserved_completion_tokens=4_096,
            )
            manager.create_session("s_test")
            events: list[RuntimeEvent] = []
            agent_session = AgentSession(agent_loop=loop, session_manager=manager)

            agent_session.prompt("s_test", "hello", event_handler=events.append)

            usage_events = [e for e in events if e.type == "context.usage"]
            self.assertEqual(len(usage_events), 1)
            self.assertEqual(usage_events[0].payload["budget"], 50_000 - 4_096)
            self.assertGreater(usage_events[0].payload["current_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
