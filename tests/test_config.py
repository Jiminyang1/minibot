from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config


class ConfigTests(unittest.TestCase):
    def test_from_env_reads_max_parallel_tools(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_MAX_PARALLEL_TOOLS": "6"},
            clear=False,
        ):
            config = Config.from_env()

        self.assertEqual(config.max_parallel_tools, 6)

    def test_from_env_reads_compact_keep_recent_tokens(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_COMPACT_KEEP_RECENT_TOKENS": "12000"},
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.compact_keep_recent_tokens, 12000)

    def test_from_env_reads_model_budget_overrides(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MINIBOT_CONTEXT_WINDOW_TOKENS": "1048576",
                "MINIBOT_MODEL_MAX_INPUT_TOKENS": "900000",
                "MINIBOT_MODEL_MAX_OUTPUT_TOKENS": "393216",
                "MINIBOT_MAX_OUTPUT_TOKENS": "32000",
                "MINIBOT_COMPACT_TOKEN_THRESHOLD": "500000",
            },
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.context_window_tokens, 1_048_576)
        self.assertEqual(config.model_max_input_tokens, 900_000)
        self.assertEqual(config.model_max_output_tokens, 393_216)
        self.assertEqual(config.max_output_tokens, 32_000)
        self.assertEqual(config.compact_token_threshold, 500_000)

    def test_legacy_reserved_completion_tokens_is_an_output_alias(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_RESERVED_COMPLETION_TOKENS": "12000"},
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.max_output_tokens, 12_000)

    def test_new_max_output_env_wins_over_legacy_alias(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MINIBOT_MAX_OUTPUT_TOKENS": "16000",
                "MINIBOT_RESERVED_COMPLETION_TOKENS": "12000",
            },
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.max_output_tokens, 16_000)

    def test_from_env_reads_approval_mode(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_APPROVAL_MODE": "always"},
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.approval_mode, "always")

    def test_from_env_rejects_invalid_approval_mode(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_APPROVAL_MODE": "maybe"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Config.from_env()

    def test_from_env_rejects_non_integer_max_parallel_tools(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_MAX_PARALLEL_TOOLS": "oops"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                Config.from_env()

    def test_zero_and_one_disable_parallel_tools_without_error(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIBOT_MAX_PARALLEL_TOOLS": "0"},
            clear=False,
        ):
            zero_config = Config.from_env()

        with patch.dict(
            "os.environ",
            {"MINIBOT_MAX_PARALLEL_TOOLS": "1"},
            clear=False,
        ):
            one_config = Config.from_env()

        self.assertEqual(zero_config.max_parallel_tools, 0)
        self.assertEqual(one_config.max_parallel_tools, 1)

    def test_negative_max_parallel_tools_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Config(max_parallel_tools=-1)

    def test_compact_keep_recent_tokens_must_fit_below_trigger(self) -> None:
        with self.assertRaises(ValueError):
            Config(
                compact_token_threshold=1000,
                max_output_tokens=200,
                compact_keep_recent_tokens=1000,
            )


if __name__ == "__main__":
    unittest.main()
