from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.llm import (
    TokenUsage,
    _extract_message_content,
    _extract_reasoning_content,
    _extract_token_usage,
    _prepare_messages_for_provider,
)


class LLMUsageExtractionTests(unittest.TestCase):
    def test_extracts_reasoning_content_from_model_extra(self) -> None:
        msg = types.SimpleNamespace(
            reasoning_content=None,
            model_extra={"reasoning_content": "thinking"},
        )

        self.assertEqual(_extract_reasoning_content(msg), "thinking")

    def test_strips_reasoning_content_for_plain_openai_providers(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "thinking",
            }
        ]

        prepared = _prepare_messages_for_provider(
            messages,
            include_reasoning_content=False,
        )

        self.assertNotIn("reasoning_content", prepared[0])

    def test_preserves_reasoning_content_for_deepseek_thinking_mode(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "thinking",
            }
        ]

        prepared = _prepare_messages_for_provider(
            messages,
            include_reasoning_content=True,
        )

        self.assertEqual(prepared[0]["reasoning_content"], "thinking")

    def test_extract_message_content_joins_text_parts(self) -> None:
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]

        extracted = _extract_message_content(content)

        self.assertEqual(extracted, "hello world")

    def test_extracts_chat_completions_usage_from_object_attributes(self) -> None:
        raw_usage = types.SimpleNamespace(
            prompt_tokens=123,
            completion_tokens=45,
            total_tokens=168,
        )

        usage = _extract_token_usage(raw_usage)

        self.assertEqual(
            usage,
            TokenUsage(input_tokens=123, output_tokens=45, total_tokens=168),
        )

    def test_extracts_usage_from_dict_with_input_output_naming(self) -> None:
        raw_usage = {
            "input_tokens": 90,
            "output_tokens": 10,
            "total_tokens": 100,
        }

        usage = _extract_token_usage(raw_usage)

        self.assertEqual(
            usage,
            TokenUsage(input_tokens=90, output_tokens=10, total_tokens=100),
        )

    def test_derives_total_tokens_when_provider_omits_total(self) -> None:
        raw_usage = {
            "prompt_tokens": 12,
            "completion_tokens": 8,
        }

        usage = _extract_token_usage(raw_usage)

        self.assertEqual(
            usage,
            TokenUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        )


if __name__ == "__main__":
    unittest.main()
