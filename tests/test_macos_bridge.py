from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.mcp_servers.macos_system.bridge import (
    AppleScriptBridge,
    AppleScriptBridgeError,
)


class _StaticMailBridge(AppleScriptBridge):
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.lines: list[str] = []

    def _run_lines(self, lines: list[str], *, args: list[str]) -> str:
        del args
        self.lines = lines
        return self.raw


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

    def test_normalize_recipients_accepts_list_and_comma_text(self) -> None:
        recipients = AppleScriptBridge._normalize_recipients(
            [" ada@example.com ", "bob@example.com, cara@example.com"],
            field_name="to",
            required=True,
        )

        self.assertEqual(
            recipients,
            ["ada@example.com", "bob@example.com", "cara@example.com"],
        )

    def test_normalize_recipients_requires_non_empty_when_requested(self) -> None:
        with self.assertRaises(AppleScriptBridgeError) as ctx:
            AppleScriptBridge._normalize_recipients(
                " , ",
                field_name="to",
                required=True,
            )

        self.assertEqual(ctx.exception.code, "invalid_args")

    def test_validate_days_back_accepts_none_and_positive_window(self) -> None:
        AppleScriptBridge._validate_days_back(None)
        AppleScriptBridge._validate_days_back(7)

    def test_validate_days_back_rejects_out_of_range_window(self) -> None:
        with self.assertRaises(AppleScriptBridgeError) as ctx:
            AppleScriptBridge._validate_days_back(0)

        self.assertEqual(ctx.exception.code, "invalid_args")
        self.assertEqual(ctx.exception.data["field"], "days_back")

    def test_list_mail_messages_sorts_unordered_records_before_limiting(self) -> None:
        field_sep = chr(31)
        record_sep = chr(30)

        def row(message_id: str, received_at: str) -> str:
            return field_sep.join(
                [
                    message_id,
                    f"Subject {message_id}",
                    "sender@example.com",
                    received_at,
                    "INBOX",
                    "iCloud",
                    "false",
                    "",
                ]
            )

        bridge = _StaticMailBridge(
            record_sep.join(
                [
                    row("older", "2026-05-18T08:00:00"),
                    row("newest", "2026-05-20T08:00:00"),
                    row("middle", "2026-05-19T08:00:00"),
                ]
            )
        )

        messages = bridge.list_mail_messages(
            account_name="iCloud",
            mailbox_name="INBOX",
            limit=2,
            unread_only=False,
            days_back=7,
        )

        self.assertEqual([item.message_id for item in messages], ["newest", "middle"])

    def test_list_mail_messages_does_not_assume_mail_order_for_date_window(self) -> None:
        bridge = _StaticMailBridge("")

        bridge.list_mail_messages(
            account_name="iCloud",
            mailbox_name="INBOX",
            limit=10,
            unread_only=False,
            days_back=7,
        )

        script = "\n".join(bridge.lines)
        self.assertNotIn("receivedDate < sinceDate then exit repeat", script)


if __name__ == "__main__":
    unittest.main()
