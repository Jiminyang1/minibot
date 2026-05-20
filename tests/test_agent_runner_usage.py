from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.artifacts import ArtifactStore
from minibot.llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from minibot.runtime.agent_runner import AgentRunner, PartialRunError, RunSpec
from minibot.runtime.events import RuntimeEvent
from minibot.runtime.tool_output_materializer import ToolOutputMaterializer
from minibot.tools.registry import ToolRegistry
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.result import ToolOutput


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "model": model,
            }
        )
        return self._responses.pop(0)


class _FailingLLM(LLMClient):
    def __init__(
        self,
        responses: list[LLMResponse],
        failure: Exception,
    ) -> None:
        self._responses = list(responses)
        self._failure = failure

    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        del messages, tools, model
        if self._responses:
            return self._responses.pop(0)
        raise self._failure


class _EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, value: str) -> ToolOutput:
        del context
        return ToolOutput.success("ok", data={"value": value})


class _TrackingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[str] = []
        self.running = 0
        self.max_running = 0
        self.executions: dict[str, int] = {}

    def start(self, name: str) -> None:
        with self.lock:
            self.events.append(f"start:{name}")
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.executions[name] = self.executions.get(name, 0) + 1

    def end(self, name: str) -> None:
        with self.lock:
            self.events.append(f"end:{name}")
            self.running -= 1


class _TrackingTool(Tool):
    def __init__(
        self,
        name: str,
        *,
        state: _TrackingState,
        delay: float = 0.03,
        read_only: bool = False,
        exclusive: bool = False,
        requires_approval: bool = False,
        body: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._state = state
        self._delay = delay
        self._read_only = read_only
        self._exclusive = exclusive
        self._requires_approval = requires_approval
        self._body = body

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def exclusive(self) -> bool:
        return self._exclusive

    @property
    def requires_approval(self) -> bool:
        return self._requires_approval

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        self._state.start(self._name)
        try:
            time.sleep(self._delay)
            if self._body is not None:
                return ToolOutput.success(
                    f"{self._name} ok",
                    data={"tool": self._name},
                    content=self._body,
                    content_kind="text",
                    content_name=self._name,
                )
            return ToolOutput.success(f"{self._name} ok", data={"tool": self._name})
        finally:
            self._state.end(self._name)


class AgentRunnerUsageTests(unittest.TestCase):
    def test_fails_fast_when_model_returns_empty_final_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(_EchoTool())
            llm = _ScriptedLLM(
                [
                    LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="echo",
                                arguments='{"value":"hi"}',
                            )
                        ],
                    ),
                    LLMResponse(
                        content="",
                        debug={
                            "raw_content_type": "list",
                            "raw_content_preview": "[]",
                            "finish_reason": "stop",
                            "tool_call_count": 0,
                        },
                    ),
                ]
            )
            emitted: list[RuntimeEvent] = []
            runner = AgentRunner(
                llm,
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                event_handler=emitted.append,
            )

            with self.assertRaises(PartialRunError) as ctx:
                runner.run(
                    RunSpec(
                        session_id="s_test",
                        model="gpt-5.4-mini",
                        messages=[{"role": "user", "content": "say hi"}],
                        tool_definitions=registry.get_definitions(),
                    )
                )

            self.assertEqual(
                str(ctx.exception.cause),
                "模型返回空回复，请重试；详见上一条空回答诊断日志。",
            )
            self.assertEqual(len(ctx.exception.events), 2)
            self.assertTrue(
                any(
                    event.type == "model.request.completed"
                    and event.payload.get("empty_reply") is True
                    for event in emitted
                )
            )
            self.assertEqual(len(llm.calls), 2)

    def test_aggregates_real_usage_across_multiple_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(_EchoTool())
            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="echo",
                                    arguments='{"value":"hi"}',
                                )
                            ],
                            usage=TokenUsage(
                                input_tokens=100,
                                output_tokens=10,
                                total_tokens=110,
                            ),
                        ),
                        LLMResponse(
                            content="done",
                            usage=TokenUsage(
                                input_tokens=120,
                                output_tokens=20,
                                total_tokens=140,
                            ),
                        ),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "say hi"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(outcome.reply, "done")
            self.assertIsNotNone(outcome.usage)
            assert outcome.usage is not None
            self.assertEqual(outcome.usage.input_tokens, 220)
            self.assertEqual(outcome.usage.output_tokens, 30)
            self.assertEqual(outcome.usage.total_tokens, 250)

    def test_raises_partial_run_error_with_accumulated_events_on_later_llm_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(_EchoTool())
            runner = AgentRunner(
                _FailingLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="echo",
                                    arguments='{"value":"hi"}',
                                )
                            ],
                            usage=TokenUsage(
                                input_tokens=100,
                                output_tokens=10,
                                total_tokens=110,
                            ),
                        ),
                    ],
                RuntimeError("llm unavailable"),
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
            )

            with self.assertRaises(PartialRunError) as ctx:
                runner.run(
                    RunSpec(
                        session_id="s_test",
                        model="gpt-5.4-mini",
                        messages=[{"role": "user", "content": "say hi"}],
                        tool_definitions=registry.get_definitions(),
                    )
                )

            exc = ctx.exception
            self.assertEqual(type(exc.cause), RuntimeError)
            self.assertEqual(str(exc.cause), "llm unavailable")
            self.assertEqual([event.role for event in exc.events], ["assistant", "tool"])
            self.assertIsNotNone(exc.usage)
            assert exc.usage is not None
            self.assertEqual(exc.usage.total_tokens, 110)

    def test_batches_concurrency_safe_tools_before_exclusive_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _TrackingState()
            registry = ToolRegistry()
            registry.register(_TrackingTool("read_a", state=state, read_only=True, delay=0.05))
            registry.register(_TrackingTool("read_b", state=state, read_only=True, delay=0.05))
            registry.register(
                _TrackingTool(
                    "exec_like",
                    state=state,
                    read_only=True,
                    exclusive=True,
                    delay=0.01,
                )
            )
            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call_1", name="read_a", arguments="{}"),
                                ToolCall(id="call_2", name="read_b", arguments="{}"),
                                ToolCall(id="call_3", name="exec_like", arguments="{}"),
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                max_parallel_tools=4,
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "run tools"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(outcome.reply, "done")
            self.assertEqual(state.max_running, 2)
            self.assertGreater(
                state.events.index("start:exec_like"),
                state.events.index("end:read_a"),
            )
            self.assertGreater(
                state.events.index("start:exec_like"),
                state.events.index("end:read_b"),
            )
            self.assertEqual(
                [event.name for event in outcome.events if event.role == "tool"],
                ["read_a", "read_b", "exec_like"],
            )

    def test_parallel_tool_messages_preserve_original_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _TrackingState()
            registry = ToolRegistry()
            registry.register(_TrackingTool("slow", state=state, read_only=True, delay=0.05))
            registry.register(_TrackingTool("fast", state=state, read_only=True, delay=0.01))
            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call_1", name="slow", arguments="{}"),
                                ToolCall(id="call_2", name="fast", arguments="{}"),
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                max_parallel_tools=4,
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "run tools"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(state.max_running, 2)
            self.assertEqual(
                [event.name for event in outcome.events if event.role == "tool"],
                ["slow", "fast"],
            )

    def test_max_parallel_tools_one_falls_back_to_serial_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _TrackingState()
            registry = ToolRegistry()
            registry.register(_TrackingTool("read_a", state=state, read_only=True))
            registry.register(_TrackingTool("read_b", state=state, read_only=True))
            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call_1", name="read_a", arguments="{}"),
                                ToolCall(id="call_2", name="read_b", arguments="{}"),
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                max_parallel_tools=1,
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "run tools"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(outcome.reply, "done")
            self.assertEqual(state.max_running, 1)
            self.assertEqual(
                state.events,
                ["start:read_a", "end:read_a", "start:read_b", "end:read_b"],
            )

    def test_denied_tool_in_safe_batch_does_not_block_other_safe_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _TrackingState()
            registry = ToolRegistry()
            registry.register(
                _TrackingTool(
                    "approved",
                    state=state,
                    read_only=True,
                    requires_approval=True,
                )
            )
            registry.register(
                _TrackingTool(
                    "denied",
                    state=state,
                    read_only=True,
                    requires_approval=True,
                )
            )
            emitted: list[RuntimeEvent] = []
            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call_1", name="denied", arguments="{}"),
                                ToolCall(id="call_2", name="approved", arguments="{}"),
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                event_handler=emitted.append,
                approval_handler=lambda request: request.tool_name != "denied",
                max_parallel_tools=4,
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "run tools"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(outcome.reply, "done")
            self.assertEqual(state.executions.get("denied", 0), 0)
            self.assertEqual(state.executions.get("approved", 0), 1)

            tool_events = [event for event in outcome.events if event.role == "tool"]
            denied_payload = json.loads(tool_events[0].content)
            approved_payload = json.loads(tool_events[1].content)
            self.assertEqual(denied_payload["code"], "denied")
            self.assertEqual(approved_payload["code"], "success")
            self.assertIn("approval.required", [event.type for event in emitted])
            self.assertIn("approval.resolved", [event.type for event in emitted])
            self.assertTrue(
                any(
                    event.type == "tool_call.failed"
                    and event.payload.get("tool") == "denied"
                    for event in emitted
                )
            )

    def test_approval_mode_always_skips_prompt_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _TrackingState()
            registry = ToolRegistry()
            registry.register(
                _TrackingTool(
                    "sensitive",
                    state=state,
                    requires_approval=True,
                )
            )
            emitted: list[RuntimeEvent] = []
            handler_calls = 0

            def _approval_handler(request) -> bool:
                nonlocal handler_calls
                del request
                handler_calls += 1
                return False

            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="sensitive",
                                    arguments="{}",
                                )
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                event_handler=emitted.append,
                approval_handler=_approval_handler,
                approval_mode="always",
                max_parallel_tools=1,
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "run tool"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(outcome.reply, "done")
            self.assertEqual(handler_calls, 0)
            self.assertEqual(state.executions.get("sensitive", 0), 1)
            self.assertNotIn("approval.required", [event.type for event in emitted])
            self.assertTrue(
                any(
                    event.type == "approval.resolved"
                    and event.payload.get("auto") is True
                    for event in emitted
                )
            )

    def test_approval_mode_can_be_changed_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AgentRunner(
                _ScriptedLLM([]),
                ToolRegistry(),
                materializer=ToolOutputMaterializer(ArtifactStore(Path(tmpdir))),
                approval_mode="ask",
            )

            self.assertEqual(runner.approval_mode, "ask")
            runner.set_approval_mode("always")
            self.assertEqual(runner.approval_mode, "always")

            with self.assertRaises(ValueError):
                runner.set_approval_mode("maybe")  # type: ignore[arg-type]

    def test_parallel_large_results_materialize_distinct_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            state = _TrackingState()
            registry = ToolRegistry()
            registry.register(
                _TrackingTool(
                    "large_a",
                    state=state,
                    read_only=True,
                    body="A" * 3500,
                )
            )
            registry.register(
                _TrackingTool(
                    "large_b",
                    state=state,
                    read_only=True,
                    body="B" * 3600,
                )
            )
            store = ArtifactStore(workspace)
            runner = AgentRunner(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call_1", name="large_a", arguments="{}"),
                                ToolCall(id="call_2", name="large_b", arguments="{}"),
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                materializer=ToolOutputMaterializer(store),
                max_parallel_tools=4,
            )

            outcome = runner.run(
                RunSpec(
                    session_id="s_test",
                    model="gpt-5.4-mini",
                    messages=[{"role": "user", "content": "run tools"}],
                    tool_definitions=registry.get_definitions(),
                )
            )

            self.assertEqual(outcome.reply, "done")
            tool_events = [event for event in outcome.events if event.role == "tool"]
            payload_a = json.loads(tool_events[0].content)
            payload_b = json.loads(tool_events[1].content)
            artifact_a = payload_a["artifact"]["id"]
            artifact_b = payload_b["artifact"]["id"]

            self.assertNotEqual(artifact_a, artifact_b)
            self.assertEqual(
                store.read_page("s_test", artifact_a, offset=0, limit=5000).content,
                "A" * 3500,
            )
            self.assertEqual(
                store.read_page("s_test", artifact_b, offset=0, limit=5000).content,
                "B" * 3600,
            )


if __name__ == "__main__":
    unittest.main()
