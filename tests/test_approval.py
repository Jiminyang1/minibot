from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.llm import LLMClient, LLMResponse, ToolCall
from minibot.runtime.approval import ApprovalPolicy, ToolApprovalGate
from minibot.runtime.cancel import RunCancelled
from minibot.runtime.events import RuntimeEvent, RuntimeEventEmitter
from minibot.runtime.messages import ModelMessage
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.registry import PreparedToolCall, ToolRegistry
from minibot.tools.result import ToolOutput

from loop_harness import build_loop, run_turn


class _SensitiveTool(Tool):
    def __init__(self, name: str = "sensitive") -> None:
        super().__init__()
        self._name = name
        self.executions = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @property
    def requires_approval(self) -> bool:
        return True

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        self.executions += 1
        return ToolOutput.success("ok")


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
        return self._responses.pop(0)


def _prepared(tool: Tool) -> PreparedToolCall:
    return PreparedToolCall(
        tool=tool,
        args={},
        context=ToolExecutionContext(session_id="s_test", run_id="r_test"),
        tool_call_id="call_1",
    )


def _emitter(events: list[RuntimeEvent]) -> RuntimeEventEmitter:
    return RuntimeEventEmitter(
        run_id="r_test", session_id="s_test", handler=events.append
    )


class ToolApprovalGateTests(unittest.TestCase):
    def test_denial_returns_output_without_invoking_tool(self) -> None:
        tool = _SensitiveTool()
        events: list[RuntimeEvent] = []
        gate = ToolApprovalGate(
            ApprovalPolicy(handler=lambda request, cancel_event: False)
        )

        denial = gate.check(
            _prepared(tool),
            run_id="r_test",
            session_id="s_test",
            emitter=_emitter(events),
            cancel_event=None,
        )

        assert denial is not None
        self.assertEqual(denial.code, "denied")
        self.assertEqual(tool.executions, 0)
        self.assertEqual(
            [event.type for event in events],
            ["approval.required", "approval.resolved"],
        )
        self.assertFalse(events[-1].payload["approved"])

    def test_non_sensitive_tool_passes_without_events(self) -> None:
        class _Plain(_SensitiveTool):
            @property
            def requires_approval(self) -> bool:
                return False

        events: list[RuntimeEvent] = []
        gate = ToolApprovalGate(
            ApprovalPolicy(handler=lambda request, cancel_event: False)
        )

        result = gate.check(
            _prepared(_Plain()),
            run_id="r_test",
            session_id="s_test",
            emitter=_emitter(events),
            cancel_event=None,
        )

        self.assertIsNone(result)
        self.assertEqual(events, [])

    def test_missing_handler_approves_in_ask_mode(self) -> None:
        gate = ToolApprovalGate(ApprovalPolicy(handler=None, mode="ask"))

        result = gate.check(
            _prepared(_SensitiveTool()),
            run_id="r_test",
            session_id="s_test",
            emitter=None,
            cancel_event=None,
        )

        self.assertIsNone(result)

    def test_cancelled_run_raises_before_asking(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        gate = ToolApprovalGate(
            ApprovalPolicy(handler=lambda request, cancel_event: True)
        )

        with self.assertRaises(RunCancelled):
            gate.check(
                _prepared(_SensitiveTool()),
                run_id="r_test",
                session_id="s_test",
                emitter=None,
                cancel_event=cancel_event,
            )

    def test_handler_exception_becomes_error_output_in_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = _SensitiveTool()
            registry = ToolRegistry()
            registry.register(tool)

            def _broken_handler(request, cancel_event) -> bool:
                del request, cancel_event
                raise ValueError("approval backend down")

            loop, manager = build_loop(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(id="call_1", name="sensitive", arguments="{}")
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                Path(tmpdir),
                approval_policy=ApprovalPolicy(handler=_broken_handler),
            )

            outcome, session = run_turn(loop, manager, "run tool")

            self.assertEqual(outcome.reply, "done")
            self.assertEqual(tool.executions, 0)
            tool_message = [m for m in session.messages if m.role == "tool"][0]
            self.assertIn("审批流程失败", tool_message.content)


if __name__ == "__main__":
    unittest.main()
