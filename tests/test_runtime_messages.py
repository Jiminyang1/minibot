from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.messages import (
    AgentMessage,
    ModelMessage,
    ModelToolCall,
    agent_message_to_session,
    format_model_messages_for_summary,
    model_message_to_openai,
    session_message_to_model,
)
from minibot.session import MessageEvent


class RuntimeMessageConversionTests(unittest.TestCase):
    def test_assistant_tool_calls_convert_to_openai_shape(self) -> None:
        message = ModelMessage.create(
            role="assistant",
            content="",
            tool_calls=[
                ModelToolCall(
                    id="call_1",
                    name="read_file",
                    arguments='{"path":"README.md"}',
                )
            ],
        )

        converted = model_message_to_openai(message)

        self.assertEqual(converted["tool_calls"][0]["id"], "call_1")
        self.assertEqual(converted["tool_calls"][0]["type"], "function")
        self.assertEqual(
            converted["tool_calls"][0]["function"],
            {"name": "read_file", "arguments": '{"path":"README.md"}'},
        )
        self.assertIsInstance(message.tool_calls[0], ModelToolCall)

    def test_tool_result_converts_name_field_for_openai(self) -> None:
        message = ModelMessage.create(
            role="tool",
            content='{"ok":true}',
            tool_call_id="call_1",
            tool_name="read_file",
        )

        converted = model_message_to_openai(message)

        self.assertEqual(converted["tool_call_id"], "call_1")
        self.assertEqual(converted["name"], "read_file")

    def test_session_reasoning_content_is_explicitly_opted_in(self) -> None:
        event = MessageEvent.create(
            role="assistant",
            content="answer",
            reasoning_content="thinking",
        )

        self.assertIsNone(session_message_to_model(event).reasoning_content)
        model_message = session_message_to_model(
            event,
            include_reasoning_content=True,
        )

        self.assertEqual(model_message.reasoning_content, "thinking")
        self.assertNotIn("reasoning_content", model_message_to_openai(model_message))
        self.assertEqual(
            model_message_to_openai(
                model_message,
                include_reasoning_content=True,
            )["reasoning_content"],
            "thinking",
        )

    def test_agent_message_persists_openai_compatible_tool_call_shape(self) -> None:
        event = agent_message_to_session(
            AgentMessage.create(
                role="assistant",
                content="",
                tool_calls=[
                    ModelToolCall(
                        id="call_1",
                        name="search_files",
                        arguments='{"query":"needle"}',
                    )
                ],
            )
        )

        self.assertEqual(event.tool_calls[0]["type"], "function")
        self.assertEqual(event.tool_calls[0]["function"]["name"], "search_files")

    def test_summary_formatter_handles_user_assistant_tool_and_call(self) -> None:
        formatted = format_model_messages_for_summary(
            [
                ModelMessage.create(role="user", content="hi"),
                ModelMessage.create(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ModelToolCall(id="call_1", name="echo", arguments='{"x":1}')
                    ],
                ),
                ModelMessage.create(
                    role="tool",
                    content='{"ok":true}',
                    tool_call_id="call_1",
                    tool_name="echo",
                ),
                ModelMessage.create(role="assistant", content="done"),
            ]
        )

        self.assertIn("USER: hi", formatted)
        self.assertIn('ASSISTANT_TOOL_CALL: echo({"x":1})', formatted)
        self.assertIn('TOOL: {"ok":true}', formatted)
        self.assertIn('TOOL_RESULT[echo]: {"ok":true}', formatted)
        self.assertIn("ASSISTANT: done", formatted)


if __name__ == "__main__":
    unittest.main()
