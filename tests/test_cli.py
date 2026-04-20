from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
