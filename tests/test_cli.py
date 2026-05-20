from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot import ui
from minibot.mcp_host.models import MCPHostSummary, MCPServerStatus


class CLITests(unittest.TestCase):
    def test_help_commands_include_mcp(self) -> None:
        commands = [command for command, _ in ui._COMMANDS]
        self.assertIn("/mcp", commands)
        self.assertIn("/mcp tools [server]", commands)

    def test_print_mcp_status_renders_summary(self) -> None:
        summary = MCPHostSummary(
            config_path="/tmp/mcp.json",
            configured_servers=2,
            enabled_servers=2,
            connected_servers=1,
            failed_servers=1,
            tool_count=3,
        )
        statuses = [
            MCPServerStatus(
                name="sqlite",
                transport="stdio",
                enabled=True,
                trusted=True,
                connected=True,
                tool_count=3,
                tool_names=("mcp__sqlite__query",),
            ),
            MCPServerStatus(
                name="figma",
                transport="streamable_http",
                enabled=True,
                trusted=False,
                connected=False,
                last_error="boom",
            ),
        ]

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ui.print_mcp_status(summary, statuses)

        output = buffer.getvalue()
        self.assertIn("MCP", output)
        self.assertIn("/tmp/mcp.json", output)
        self.assertIn("sqlite", output)
        self.assertIn("figma", output)
        self.assertIn("boom", output)

    def test_prompt_approval_reprompts_on_invalid_input(self) -> None:
        buffer = io.StringIO()
        with (
            patch("builtins.input", side_effect=["会议 ID：681 639 607", "y"]) as mock_input,
            redirect_stdout(buffer),
        ):
            approved = ui.prompt_approval("calendar_create_event", {"title": "meeting"})

        self.assertTrue(approved)
        self.assertEqual(mock_input.call_count, 2)
        self.assertIn("请输入 y 或 n", buffer.getvalue())

    def test_prompt_approval_empty_input_uses_safe_default_reject(self) -> None:
        with patch("builtins.input", return_value=""):
            approved = ui.prompt_approval("calendar_create_event", {"title": "meeting"})

        self.assertFalse(approved)

    def test_read_user_input_returns_single_line_without_paste_guard(self) -> None:
        with (
            patch("builtins.input", return_value="  hello  "),
            patch("minibot.ui._drain_pending_stdin", return_value=""),
        ):
            self.assertEqual(ui.read_user_input(), "hello")

    def test_read_user_input_merges_multiline_paste_into_one_message(self) -> None:
        buffer = io.StringIO()
        with (
            patch("builtins.input", return_value="line 1"),
            patch("minibot.ui._drain_pending_stdin", return_value="line 2\nline 3\n"),
            redirect_stdout(buffer),
        ):
            user_input = ui.read_user_input()

        self.assertEqual(user_input, "line 1\nline 2\nline 3")
        self.assertIn("检测到 3 行粘贴内容，已合并为一条请求", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
