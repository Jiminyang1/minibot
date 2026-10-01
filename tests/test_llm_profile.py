from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.llm_factory import build_llm_client_from_profile
from minibot.llm_profile import (
    LLMProfile,
    ModelCapabilities,
    OpenAICompatibleCompat,
    build_llm_profile,
)
from minibot.llm_providers.openai_compatible import OpenAICompatibleClient


class LLMProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)

    def test_builds_openai_profile_from_defaults(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-test"

        profile = build_llm_profile(model="gpt-5.4-mini")

        self.assertEqual(profile.provider, "openai")
        self.assertEqual(profile.api, "openai_chat_completions")
        self.assertEqual(profile.model, "gpt-5.4-mini")
        self.assertEqual(profile.api_key, "sk-test")
        self.assertFalse(profile.compat.include_reasoning_content)
        assert profile.capabilities is not None
        self.assertEqual(profile.capabilities.context_window_tokens, 400_000)
        self.assertEqual(profile.capabilities.max_input_tokens, 272_000)
        self.assertEqual(profile.capabilities.max_output_tokens, 128_000)

    def test_builds_deepseek_profile_from_model_prefix(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "ds-test"

        profile = build_llm_profile(model="deepseek-v4-pro")

        self.assertEqual(profile.provider, "deepseek")
        self.assertEqual(profile.api, "openai_chat_completions")
        self.assertEqual(profile.base_url, "https://api.deepseek.com")
        self.assertEqual(profile.api_key, "ds-test")
        self.assertTrue(profile.compat.include_reasoning_content)
        self.assertEqual(profile.compat.max_output_parameter, "max_tokens")
        assert profile.capabilities is not None
        self.assertEqual(profile.capabilities.context_window_tokens, 1_048_576)
        self.assertEqual(profile.capabilities.max_output_tokens, 393_216)

    def test_deepseek_profile_accepts_openai_compatible_api_key_fallback(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-compatible"
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

        profile = build_llm_profile(model="deepseek-v4-pro")

        self.assertEqual(profile.provider, "deepseek")
        self.assertEqual(profile.api_key, "sk-compatible")
        self.assertEqual(profile.base_url, "https://api.deepseek.com")

    def test_unknown_model_requires_explicit_capabilities(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-test"

        with self.assertRaisesRegex(ValueError, "MINIBOT_CONTEXT_WINDOW_TOKENS"):
            build_llm_profile(model="custom-model")

    def test_unknown_model_accepts_explicit_capabilities(self) -> None:
        os.environ["OPENAI_API_KEY"] = "sk-test"

        profile = build_llm_profile(
            model="custom-model",
            context_window_tokens=64_000,
            model_max_output_tokens=8_000,
            request_max_output_tokens=4_000,
        )

        self.assertEqual(
            profile.capabilities,
            ModelCapabilities(64_000, 8_000, source="env"),
        )
        self.assertEqual(profile.request_max_output_tokens, 4_000)

    def test_rejects_request_output_above_model_limit(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "ds-test"

        with self.assertRaisesRegex(ValueError, "超过模型能力"):
            build_llm_profile(
                model="deepseek-v4-pro",
                request_max_output_tokens=400_000,
            )

    def test_factory_creates_openai_compatible_client(self) -> None:
        client = build_llm_client_from_profile(
            LLMProfile(
                provider="openai",
                api="openai_chat_completions",
                model="gpt-5.4-mini",
                base_url=None,
                api_key="sk-test",
                compat=OpenAICompatibleCompat(include_reasoning_content=False),
            )
        )

        self.assertIsInstance(client, OpenAICompatibleClient)

    def test_factory_rejects_unknown_api(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "不支持的 LLM api"):
            build_llm_client_from_profile(
                LLMProfile(
                    provider="openai",
                    api="custom_api",
                    model="gpt-5.4-mini",
                    base_url=None,
                    api_key="sk-test",
                    compat=OpenAICompatibleCompat(),
                )
            )


if __name__ == "__main__":
    unittest.main()
