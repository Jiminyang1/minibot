from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.llm import TokenUsage
from minibot.mcp_host.models import MCPToolSpec
from minibot.mcp_host.provider import MCPToolProxy
from minibot.run_log import RunLogStore
from minibot.runtime.agent_runner import PartialRunError, RunOutcome
from minibot.runtime.context_manager import PreparedContext
from minibot.runtime.events import RuntimeEvent
from minibot.runtime.hooks import HookContext, RuntimeHook, RuntimeHookManager
from minibot.runtime.messages import AgentMessage, ModelMessage
from minibot.runtime.turn_engine import TurnEngine
from minibot.runtime.turn_engine import TurnResult
from minibot.session import MessageEvent, SessionManager
from minibot.tools.registry import ToolRegistry
from mcp.types import CallToolResult, TextContent


class _FakeMCPClient:
    def __init__(self, transport_type: str = "streamable_http") -> None:
        self.config = type(
            "_Config",
            (),
            {"transport": type("_Transport", (), {"type": transport_type})()},
        )()

    def call_tool(self, remote_name: str, arguments: dict[str, object]) -> CallToolResult:
        del remote_name, arguments
        return CallToolResult(content=[TextContent(type="text", text="ok")])


class _StubContextManager:
    def __init__(
        self,
        *,
        prepared: PreparedContext | None = None,
        prepare_exc: Exception | None = None,
        visible_tokens: int = 123,
        on_prepare: Callable[[object], None] | None = None,
    ) -> None:
        self._prepared = prepared
        self._prepare_exc = prepare_exc
        self._visible_tokens = visible_tokens
        self._on_prepare = on_prepare

    def prepare_for_turn(self, *, session: object, user_input: str | None = None) -> PreparedContext:
        del user_input
        if self._prepare_exc is not None:
            raise self._prepare_exc
        if self._on_prepare is not None:
            self._on_prepare(session)
        assert self._prepared is not None
        return self._prepared

    def compact_session(self, *, session: object) -> tuple[bool, str]:
        del session
        return False, "noop"

    def estimate_visible_tokens(self, *, session: object) -> int:
        del session
        return self._visible_tokens

    def list_available_skills(self) -> list[tuple[str, str, tuple[str, ...]]]:
        return []

    @property
    def effective_input_budget(self) -> int:
        return 456


class _StubRunner:
    def __init__(
        self,
        *,
        reply: str = "",
        messages: list[AgentMessage] | None = None,
        usage: TokenUsage | None = None,
        exc: Exception | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._reply = reply
        self._messages = list(messages or [])
        self._usage = usage
        self._exc = exc
        self.tool_registry = tool_registry or ToolRegistry()
        self.seen_run_spec = None

    def run(self, run_spec: object) -> RunOutcome:
        self.seen_run_spec = run_spec
        if self._exc is not None:
            raise self._exc
        return RunOutcome(
            reply=self._reply,
            messages=list(self._messages),
            usage=self._usage,
        )


class _RewriteTurnHook(RuntimeHook):
    def after_turn(self, context: HookContext, result: TurnResult) -> TurnResult:
        del context
        return TurnResult(
            reply="hooked reply",
            did_compact=result.did_compact,
            compact_message=result.compact_message,
        )


def _read_run_logs(workspace: Path) -> list[dict[str, object]]:
    path = workspace / ".minibot" / "runs.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TurnEngineRunLogTests(unittest.TestCase):
    def test_emits_current_context_usage_against_effective_input_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=False,
                compact_message=None,
            )
            emitted: list[RuntimeEvent] = []
            engine = TurnEngine(
                _StubRunner(reply="ok"),
                manager,
                Config(),
                context_manager=_StubContextManager(
                    prepared=prepared,
                    visible_tokens=123,
                ),
                event_handler=emitted.append,
                run_log_store=RunLogStore(workspace),
            )

            engine.handle_turn(session, "hello")

            self.assertEqual(emitted[0].type, "context.usage")
            self.assertEqual(emitted[0].payload["current_tokens"], 123)
            self.assertEqual(emitted[0].payload["budget"], 456)

    def test_successful_turn_appends_summary_log_and_preserves_session_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            user_input = ("hello   world " * 20).strip()
            reply = ("final   answer " * 30).strip()
            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=True,
                compact_message="已压缩: 20 -> 8 条消息",
            )
            messages = [
                AgentMessage.create(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"README.md\"}",
                            },
                        }
                    ],
                ),
                AgentMessage.create(
                    role="tool",
                    content="{\"ok\":true}",
                    tool_call_id="call_1",
                    name="read_file",
                ),
                AgentMessage.create(role="assistant", content=reply),
            ]
            engine = TurnEngine(
                _StubRunner(
                    reply=reply,
                    messages=messages,
                    usage=TokenUsage(
                        input_tokens=111,
                        output_tokens=29,
                        total_tokens=140,
                    ),
                ),
                manager,
                Config(),
                context_manager=_StubContextManager(prepared=prepared),
                run_log_store=RunLogStore(workspace),
            )

            result = engine.handle_turn(session, user_input)

            self.assertTrue(result.did_compact)
            logs = _read_run_logs(workspace)
            self.assertEqual(len(logs), 1)
            log = logs[0]
            self.assertEqual(log["session_id"], "s_test")
            self.assertEqual(log["turn_index"], 1)
            self.assertEqual(log["status"], "success")
            self.assertEqual(log["did_compact"], True)
            self.assertEqual(log["tool_call_count"], 1)
            self.assertEqual(log["llm_call_count"], 2)
            self.assertEqual(log["tools_used"], ["read_file"])
            self.assertEqual(log["mcp_tool_call_count"], 0)
            self.assertEqual(log["mcp_servers_used"], [])
            self.assertEqual(log["mcp_transports_used"], [])
            self.assertEqual(log["mcp_error_count"], 0)
            self.assertEqual(log["compact_message"], "已压缩: 20 -> 8 条消息")
            self.assertEqual(log["model"], Config().model)
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
            self.assertRegex(str(log["run_id"]), r"^r_\d{8}_\d{6}_[0-9a-f]{4}$")

            session_dir = workspace / ".minibot" / "sessions" / "s_test"
            self.assertTrue((session_dir / "messages.jsonl").exists())
            self.assertTrue((workspace / ".minibot" / "runs.jsonl").exists())
            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 4)

    def test_after_turn_hook_rewrites_result_before_persisting_final_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=False,
                compact_message=None,
            )
            engine = TurnEngine(
                _StubRunner(
                    reply="original reply",
                    messages=[
                        AgentMessage.create(role="assistant", content="original reply")
                    ],
                ),
                manager,
                Config(),
                context_manager=_StubContextManager(prepared=prepared),
                hook_manager=RuntimeHookManager([_RewriteTurnHook()]),
                run_log_store=RunLogStore(workspace),
            )

            result = engine.handle_turn(session, "hello")

            self.assertEqual(result.reply, "hooked reply")
            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.messages[-1].content, "hooked reply")
            self.assertEqual(
                _read_run_logs(workspace)[0]["final_reply_preview"],
                "hooked reply",
            )

    def test_auto_compaction_saves_compacted_transcript_before_appending_new_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            old_user = MessageEvent.create(role="user", content="old question")
            old_reply = MessageEvent.create(role="assistant", content="old answer")
            session.add_message(old_user)
            session.add_message(old_reply)
            manager.save(session)

            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=True,
                compact_message="已压缩: 2 -> 1 条消息",
            )

            def _compact_in_memory(target_session: object) -> None:
                assert hasattr(target_session, "messages")
                target_session.messages = [
                    MessageEvent.create(
                        role="assistant",
                        content="[Summary of earlier conversation]\nold summary",
                    )
                ]

            engine = TurnEngine(
                _StubRunner(
                    reply="new answer",
                    messages=[AgentMessage.create(role="assistant", content="new answer")],
                ),
                manager,
                Config(),
                context_manager=_StubContextManager(
                    prepared=prepared,
                    on_prepare=_compact_in_memory,
                ),
                run_log_store=RunLogStore(workspace),
            )

            engine.handle_turn(session, "new question")

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(
                [message.content for message in reloaded.messages],
                [
                    "[Summary of earlier conversation]\nold summary",
                    "new question",
                    "new answer",
                ],
            )

    def test_run_log_records_mcp_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=False,
                compact_message=None,
            )
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
            tool_payload = {
                "ok": False,
                "code": "error",
                "summary": "sqlite.query 返回错误结果。",
                "data": {"server": "sqlite"},
                "artifact": None,
                "truncated": False,
            }
            engine = TurnEngine(
                _StubRunner(
                    reply="done",
                    messages=[
                        AgentMessage.create(
                            role="tool",
                            content=json.dumps(tool_payload, ensure_ascii=False),
                            tool_call_id="call_1",
                            name="mcp__sqlite__query",
                        ),
                        AgentMessage.create(role="assistant", content="done"),
                    ],
                    tool_registry=registry,
                ),
                manager,
                Config(),
                context_manager=_StubContextManager(prepared=prepared),
                run_log_store=RunLogStore(workspace),
            )

            engine.handle_turn(session, "query db")

            log = _read_run_logs(workspace)[0]
            self.assertEqual(log["mcp_tool_call_count"], 1)
            self.assertEqual(log["mcp_servers_used"], ["sqlite"])
            self.assertEqual(log["mcp_transports_used"], ["stdio"])
            self.assertEqual(log["mcp_error_count"], 1)

    def test_failed_turn_from_context_manager_still_writes_failed_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            engine = TurnEngine(
                _StubRunner(),
                manager,
                Config(),
                context_manager=_StubContextManager(
                    prepare_exc=RuntimeError("context exploded")
                ),
                run_log_store=RunLogStore(workspace),
            )

            with self.assertRaisesRegex(RuntimeError, "context exploded"):
                engine.handle_turn(session, "hello there")

            logs = _read_run_logs(workspace)
            self.assertEqual(len(logs), 1)
            log = logs[0]
            self.assertEqual(log["status"], "failed")
            self.assertEqual(log["error_type"], "RuntimeError")
            self.assertEqual(log["error_message_preview"], "context exploded")
            self.assertIsNone(log["input_tokens"])
            self.assertIsNone(log["output_tokens"])
            self.assertIsNone(log["total_tokens"])
            self.assertEqual(log["llm_call_count"], 0)
            self.assertEqual(log["tool_call_count"], 0)
            self.assertEqual(log["tools_used"], [])
            self.assertIsNone(log["final_reply_preview"])

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 0)

    def test_failed_turn_from_runner_logs_error_after_user_message_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=False,
                compact_message=None,
            )
            engine = TurnEngine(
                _StubRunner(exc=ValueError("runner failed badly")),
                manager,
                Config(),
                context_manager=_StubContextManager(prepared=prepared),
                run_log_store=RunLogStore(workspace),
            )

            with self.assertRaisesRegex(ValueError, "runner failed badly"):
                engine.handle_turn(session, "trigger runner failure")

            logs = _read_run_logs(workspace)
            self.assertEqual(len(logs), 1)
            log = logs[0]
            self.assertEqual(log["status"], "failed")
            self.assertEqual(log["error_type"], "ValueError")
            self.assertIsNone(log["input_tokens"])
            self.assertIsNone(log["output_tokens"])
            self.assertIsNone(log["total_tokens"])
            self.assertEqual(log["llm_call_count"], 0)
            self.assertEqual(log["tool_call_count"], 0)
            self.assertEqual(log["tools_used"], [])

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 1)
            self.assertEqual(reloaded.messages[0].role, "user")

    def test_partial_runner_failure_persists_completed_messages_and_logs_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            prepared = PreparedContext(
                messages=[ModelMessage.create(role="system", content="sys")],
                tool_definitions=[],
                did_compact=False,
                compact_message=None,
            )
            partial_messages = [
                AgentMessage.create(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"README.md\"}",
                            },
                        }
                    ],
                ),
                AgentMessage.create(
                    role="tool",
                    content="{\"ok\":true}",
                    tool_call_id="call_1",
                    name="read_file",
                ),
            ]
            engine = TurnEngine(
                _StubRunner(
                    exc=PartialRunError(
                        cause=RuntimeError("llm unavailable"),
                        messages=partial_messages,
                        usage=TokenUsage(
                            input_tokens=100,
                            output_tokens=10,
                            total_tokens=110,
                        ),
                    )
                ),
                manager,
                Config(),
                context_manager=_StubContextManager(prepared=prepared),
                run_log_store=RunLogStore(workspace),
            )

            with self.assertRaisesRegex(RuntimeError, "llm unavailable"):
                engine.handle_turn(session, "trigger partial failure")

            logs = _read_run_logs(workspace)
            self.assertEqual(len(logs), 1)
            log = logs[0]
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


if __name__ == "__main__":
    unittest.main()
