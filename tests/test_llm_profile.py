from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.llm_factory import build_llm_client_from_profile
from minibot.llm_profile import (
    LLMProfile,
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

    def test_builds_deepseek_profile_from_model_prefix(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "ds-test"

        profile = build_llm_profile(model="deepseek-chat")

        self.assertEqual(profile.provider, "deepseek")
        self.assertEqual(profile.api, "openai_chat_completions")
        self.assertEqual(profile.base_url, "https://api.deepseek.com")
        self.assertEqual(profile.api_key, "ds-test")
        self.assertTrue(profile.compat.include_reasoning_content)

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
