from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.mcp_servers.macos_system.bridge import AppleScriptBridge


class MacOSBridgeTests(unittest.TestCase):
    def test_parse_local_datetime_accepts_timezone_and_converts_to_local(self) -> None:
        parsed = AppleScriptBridge._parse_local_datetime(
            "2026-04-20T00:00:00+08:00",
            field_name="start_at",
        )
        expected = (
            datetime.fromisoformat("2026-04-20T00:00:00+08:00")
            .astimezone()
            .replace(tzinfo=None, microsecond=0)
        )
        self.assertEqual(parsed, expected)
        self.assertIsNone(parsed.tzinfo)

    def test_parse_local_datetime_accepts_z_suffix(self) -> None:
        parsed = AppleScriptBridge._parse_local_datetime(
            "2026-04-20T00:00:00Z",
            field_name="start_at",
        )
        expected = (
            datetime.fromisoformat("2026-04-20T00:00:00+00:00")
            .astimezone()
            .replace(tzinfo=None, microsecond=0)
        )
        self.assertEqual(parsed, expected)


if __name__ == "__main__":
    unittest.main()
