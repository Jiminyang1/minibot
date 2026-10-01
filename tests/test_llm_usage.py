from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.llm import TokenUsage
from minibot.llm_providers.openai_compatible import (
    _extract_message_content,
    _extract_reasoning_content,
    _extract_token_usage,
)
from minibot.runtime.messages import ModelMessage, model_messages_to_openai


class LLMUsageExtractionTests(unittest.TestCase):
    def test_extracts_reasoning_content_from_model_extra(self) -> None:
        msg = types.SimpleNamespace(
            reasoning_content=None,
            model_extra={"reasoning_content": "thinking"},
        )

        self.assertEqual(_extract_reasoning_content(msg), "thinking")

    def test_strips_reasoning_content_for_plain_openai_providers(self) -> None:
        messages = [
            ModelMessage.create(
                role="assistant",
                content="answer",
                reasoning_content="thinking",
            )
        ]

        prepared = model_messages_to_openai(
            messages,
            include_reasoning_content=False,
        )

        self.assertNotIn("reasoning_content", prepared[0])

    def test_preserves_reasoning_content_for_deepseek_thinking_mode(self) -> None:
        messages = [
            ModelMessage.create(
                role="assistant",
                content="answer",
                reasoning_content="thinking",
            )
        ]

        prepared = model_messages_to_openai(
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

    def test_extracts_deepseek_prompt_cache_hits(self) -> None:
        raw_usage = types.SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            prompt_cache_hit_tokens=896,
            prompt_cache_miss_tokens=104,
        )

        usage = _extract_token_usage(raw_usage)

        assert usage is not None
        self.assertEqual(usage.cached_input_tokens, 896)

    def test_extracts_openai_nested_cached_tokens(self) -> None:
        raw_usage = {
            "prompt_tokens": 2000,
            "completion_tokens": 20,
            "total_tokens": 2020,
            "prompt_tokens_details": {"cached_tokens": 1792},
        }

        usage = _extract_token_usage(raw_usage)

        assert usage is not None
        self.assertEqual(usage.cached_input_tokens, 1792)

    def test_cached_tokens_absent_when_provider_does_not_report(self) -> None:
        usage = _extract_token_usage({"prompt_tokens": 12, "completion_tokens": 8})

        assert usage is not None
        self.assertIsNone(usage.cached_input_tokens)


if __name__ == "__main__":
    unittest.main()
