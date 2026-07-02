from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.types import CallToolResult, TextContent

from minibot.artifacts import ArtifactStore
from minibot.llm import LLMClient, LLMResponse, ToolCall
from minibot.mcp_host.client import MCPClientTimeoutError
from minibot.mcp_host.models import MCPToolSpec
from minibot.mcp_host.provider import MCPToolProxy
from minibot.runtime.messages import ModelMessage
from minibot.tools.base import ToolExecutionContext
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.registry import ToolRegistry

from loop_harness import build_loop, run_turn


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


class _FakeClient:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, remote_name: str, arguments: dict[str, object]) -> CallToolResult:
        self.calls.append((remote_name, arguments))
        if self._error is not None:
            raise self._error
        return self._result


def _tool_spec() -> MCPToolSpec:
    return MCPToolSpec(
        server_name="figma",
        remote_name="comment",
        title=None,
        description="Create a comment",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    )


class MCPToolProxyTests(unittest.TestCase):
    def test_namespaces_tool_name_and_respects_trust(self) -> None:
        trusted = MCPToolProxy(
            client=_FakeClient(result=CallToolResult(content=[TextContent(type="text", text="ok")])),
            tool_spec=_tool_spec(),
            trusted=True,
        )
        untrusted = MCPToolProxy(
            client=_FakeClient(result=CallToolResult(content=[TextContent(type="text", text="ok")])),
            tool_spec=_tool_spec(),
            trusted=False,
        )

        self.assertEqual(trusted.name, "mcp__figma__comment")
        self.assertEqual(trusted.parameters["required"], ["message"])
        self.assertTrue(trusted.exclusive)
        self.assertFalse(trusted.requires_approval)
        self.assertTrue(untrusted.requires_approval)

    def test_text_only_result_maps_to_success(self) -> None:
        proxy = MCPToolProxy(
            client=_FakeClient(
                result=CallToolResult(content=[TextContent(type="text", text="hello")])
            ),
            tool_spec=_tool_spec(),
            trusted=True,
        )

        output = proxy.execute(
            context=ToolExecutionContext(session_id="s_test"),
            message="hello",
        )

        self.assertTrue(output.ok)
        self.assertEqual(output.code, "success")
        self.assertEqual(output.content, "hello")
        self.assertEqual(output.data["server"], "figma")
        self.assertEqual(output.data["remote_tool"], "comment")
        self.assertEqual(output.data["content_block_types"], ["text"])
        self.assertFalse(output.data["has_structured_content"])

    def test_structured_or_mixed_result_maps_to_json_content(self) -> None:
        proxy = MCPToolProxy(
            client=_FakeClient(
                result=CallToolResult(
                    content=[TextContent(type="text", text="hello")],
                    structuredContent={"comment_id": "c_123"},
                )
            ),
            tool_spec=_tool_spec(),
            trusted=True,
        )

        output = proxy.execute(
            context=ToolExecutionContext(session_id="s_test"),
            message="hello",
        )

        self.assertTrue(output.ok)
        assert output.content is not None
        payload = json.loads(output.content)
        self.assertEqual(payload["structured_content"], {"comment_id": "c_123"})
        self.assertEqual(payload["content"][0]["text"], "hello")
        self.assertTrue(output.data["has_structured_content"])

    def test_remote_error_maps_to_failure(self) -> None:
        proxy = MCPToolProxy(
            client=_FakeClient(
                result=CallToolResult(
                    content=[TextContent(type="text", text="bad request")],
                    isError=True,
                )
            ),
            tool_spec=_tool_spec(),
            trusted=True,
        )

        output = proxy.execute(
            context=ToolExecutionContext(session_id="s_test"),
            message="hello",
        )

        self.assertFalse(output.ok)
        self.assertEqual(output.code, "error")

    def test_timeout_maps_to_timeout_failure(self) -> None:
        proxy = MCPToolProxy(
            client=_FakeClient(error=MCPClientTimeoutError("too slow")),
            tool_spec=_tool_spec(),
            trusted=True,
        )

        output = proxy.execute(
            context=ToolExecutionContext(session_id="s_test"),
            message="hello",
        )

        self.assertFalse(output.ok)
        self.assertEqual(output.code, "timeout")


class MCPRunnerIntegrationTests(unittest.TestCase):
    def test_agent_loop_handles_mcp_proxy_like_normal_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(
                MCPToolProxy(
                    client=_FakeClient(
                        result=CallToolResult(
                            content=[TextContent(type="text", text="created")]
                        )
                    ),
                    tool_spec=_tool_spec(),
                    trusted=True,
                )
            )
            loop, manager = build_loop(
                _ScriptedLLM(
                    [
                        LLMResponse(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="mcp__figma__comment",
                                    arguments='{"message":"hello"}',
                                )
                            ],
                        ),
                        LLMResponse(content="done"),
                    ]
                ),
                registry,
                Path(tmpdir),
            )

            outcome, session = run_turn(loop, manager, "comment")

            self.assertEqual(outcome.reply, "done")
            tool_messages = [
                message for message in session.messages if message.role == "tool"
            ]
            self.assertEqual(len(tool_messages), 1)
            payload = json.loads(tool_messages[0].content)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["data"]["server"], "figma")


if __name__ == "__main__":
    unittest.main()
