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


if __name__ == "__main__":
    unittest.main()
