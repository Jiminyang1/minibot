from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.langfuse_tracing import LangfuseFold, build_langfuse_fold
from minibot.llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from minibot.runtime.agent_session import AgentSession
from minibot.runtime.events import RuntimeEventEmitter
from minibot.runtime.messages import ModelMessage
from minibot.server import _for_wire
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput

from loop_harness import build_loop


class _FakeObservation:
    """Records what the fold asks Langfuse to do."""

    def __init__(self, client, *, parent, attributes, **kwargs) -> None:
        self.client = client
        self.parent = parent
        self.attributes = dict(attributes)
        self.kwargs = kwargs
        self.updates: list[dict] = []
        self.events: list[dict] = []
        self.ended = False

    @property
    def name(self):
        return self.kwargs.get("name")

    @property
    def as_type(self):
        return self.kwargs.get("as_type")

    def start_observation(self, **kwargs):
        return self.client._record(parent=self, **kwargs)

    def create_event(self, **kwargs):
        self.events.append({**kwargs, "attributes": dict(self.client.active)})

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self

    def end(self):
        self.ended = True

    def merged(self) -> dict:
        merged = dict(self.kwargs)
        for update in self.updates:
            merged.update({k: v for k, v in update.items() if v is not None})
        return merged


class _FakeLangfuse:
    def __init__(self) -> None:
        self.active: dict = {}
        self.observations: list[_FakeObservation] = []
        self.shut_down = False

    @contextmanager
    def propagate_attributes(self, **attributes):
        previous = self.active
        self.active = {**previous, **attributes}
        try:
            yield
        finally:
            self.active = previous

    def start_observation(self, **kwargs):
        return self._record(parent=None, **kwargs)

    def _record(self, *, parent, **kwargs):
        observation = _FakeObservation(
            self, parent=parent, attributes=self.active, **kwargs
        )
        self.observations.append(observation)
        return observation

    def shutdown(self):
        self.shut_down = True

    def of_type(self, as_type: str) -> list[_FakeObservation]:
        return [o for o in self.observations if o.as_type == as_type]


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


class _ReadTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "read"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    def execute(self, *, context: ToolExecutionContext, path: str = "") -> ToolOutput:
        del context
        return ToolOutput.success(f"read {path}")


def _fold(client: _FakeLangfuse, **kwargs) -> LangfuseFold:
    return LangfuseFold(
        client, propagate_attributes=client.propagate_attributes, **kwargs
    )


def _session(llm: LLMClient, workspace: Path, fold: LangfuseFold) -> AgentSession:
    registry = ToolRegistry()
    registry.register(_ReadTool())
    loop, manager = build_loop(llm, registry, workspace)
    manager.create_session("s_test")
    return AgentSession(
        agent_loop=loop, session_manager=manager, base_event_handler=fold
    )


class LangfuseFoldTests(unittest.TestCase):
    def test_turn_becomes_one_trace_with_generations_and_tool(self) -> None:
        client = _FakeLangfuse()
        llm = _ScriptedLLM(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_1", name="read_file", arguments='{"path": "a.py"}'
                        )
                    ],
                    usage=TokenUsage(
                        input_tokens=100,
                        output_tokens=9,
                        total_tokens=109,
                        cached_input_tokens=80,
                    ),
                ),
                LLMResponse(
                    content="done",
                    usage=TokenUsage(input_tokens=120, output_tokens=5, total_tokens=125),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _session(llm, Path(tmpdir), _fold(client))
            session.prompt("s_test", "read a.py", source="cli")

        [root] = client.of_type("agent")
        self.assertEqual(root.merged()["input"], "read a.py")
        self.assertEqual(root.merged()["output"], "done")
        self.assertTrue(root.ended)

        first, second = client.of_type("generation")
        self.assertIs(first.parent, root)
        self.assertEqual(
            [m["role"] for m in first.kwargs["input"]], ["system", "user"]
        )
        self.assertEqual(
            first.merged()["output"]["tool_calls"][0]["name"], "read_file"
        )
        # Cached prompt tokens are split out so they are not charged twice.
        self.assertEqual(
            first.merged()["usage_details"],
            {"input": 20, "input_cached_tokens": 80, "output": 9, "total": 109},
        )
        self.assertEqual(second.merged()["output"]["content"], "done")
        self.assertEqual(second.merged()["usage_details"]["input"], 120)

        [tool] = client.of_type("tool")
        self.assertIs(tool.parent, root)
        self.assertEqual(tool.kwargs["input"], {"path": "a.py"})
        self.assertEqual(tool.merged()["output"]["summary"], "read a.py")

        # Trace attributes land on every observation, not just the root.
        for observation in client.observations:
            self.assertTrue(observation.ended, observation.name)
            self.assertEqual(observation.attributes["session_id"], "s_test")
            self.assertEqual(observation.attributes["trace_name"], "minibot.turn")
            self.assertEqual(observation.attributes["tags"], ["cli"])

    def test_failed_run_closes_open_generation_and_marks_error(self) -> None:
        client = _FakeLangfuse()
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _session(_ScriptedLLM([]), Path(tmpdir), _fold(client))
            with self.assertRaises(RuntimeError):
                session.prompt("s_test", "hello")

        [root] = client.of_type("agent")
        self.assertEqual(root.merged()["level"], "ERROR")
        self.assertIn("llm unavailable", root.merged()["status_message"])
        [generation] = client.of_type("generation")
        self.assertTrue(generation.ended)
        self.assertEqual(generation.merged()["level"], "WARNING")

    def test_tracing_errors_never_reach_the_run(self) -> None:
        client = _FakeLangfuse()

        def boom(**kwargs):
            raise ConnectionError("langfuse down")

        client.start_observation = boom
        logs: list[str] = []
        llm = _ScriptedLLM([LLMResponse(content="still works")])
        with tempfile.TemporaryDirectory() as tmpdir:
            session = _session(llm, Path(tmpdir), _fold(client, log=logs.append))
            outcome = session.prompt("s_test", "hello")

        self.assertEqual(outcome.reply, "still works")
        self.assertEqual(len(logs), 1)

    def test_compaction_summary_becomes_its_own_generation(self) -> None:
        client = _FakeLangfuse()
        fold = _fold(client)
        emitter = RuntimeEventEmitter(run_id="r1", session_id="s1", handler=fold)
        emitter.emit("run.started", {"input": "hi"})
        emitter.emit(
            "compaction.request.started",
            {"model": "m", "messages": [{"role": "system", "content": "sum"}]},
        )
        emitter.emit(
            "compaction.request.completed",
            {"usage": {"input_tokens": 50, "output_tokens": 5}, "output": {"content": "s"}},
        )
        emitter.emit("run.completed", {"reply": "ok"})

        [generation] = client.of_type("generation")
        self.assertEqual(generation.name, "compaction.summary")
        self.assertEqual(generation.merged()["usage_details"], {"input": 50, "output": 5})
        self.assertTrue(generation.ended)

    def test_input_is_trimmed_to_latest_messages_behind_system_prompt(self) -> None:
        fold = _fold(_FakeLangfuse(), max_input_messages=2)
        messages = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": str(i)} for i in range(5)
        ]

        trimmed = fold._trim_messages(messages)

        self.assertEqual([m["content"] for m in trimmed][0], "sys")
        self.assertIn("省略了 3 条", trimmed[1]["content"])
        self.assertEqual([m["content"] for m in trimmed[2:]], ["3", "4"])
        self.assertEqual(_fold(_FakeLangfuse(), max_input_messages=0)._trim_messages(messages), messages)

    def test_disabled_without_keys_or_when_switched_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(build_langfuse_fold())
        keys = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
        with mock.patch.dict(os.environ, {**keys, "MINIBOT_LANGFUSE": "0"}, clear=True):
            self.assertIsNone(build_langfuse_fold())

    def test_server_strips_request_payload_from_wire_events(self) -> None:
        emitter = RuntimeEventEmitter(run_id="r1", session_id="s1")
        event = emitter.emit(
            "model.request.started",
            {"iteration": 1, "messages": [{"role": "user", "content": "x" * 1000}]},
        )

        wire = _for_wire(event)

        self.assertNotIn("messages", wire.payload)
        self.assertEqual(wire.payload["iteration"], 1)
        self.assertIn("messages", event.payload)


if __name__ == "__main__":
    unittest.main()
