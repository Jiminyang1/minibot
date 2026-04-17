from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.llm import LLMClient, LLMResponse, TokenUsage, ToolCall
from minibot.runtime.agent_runner import AgentRunner, PartialRunError, RunSpec
from minibot.tools.registry import ToolRegistry
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.result import ToolResult


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        del messages, tools, model
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

    def execute(self, *, context: ToolExecutionContext, value: str) -> ToolResult:
        del context
        return ToolResult.success("ok", data={"value": value})


class AgentRunnerUsageTests(unittest.TestCase):
    def test_aggregates_real_usage_across_multiple_llm_calls(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
