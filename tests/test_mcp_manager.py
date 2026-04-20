from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.mcp_host.client import MCPClientConnectionError
from minibot.mcp_host.host import MCPHost
from minibot.mcp_host.models import (
    MCPServerConfig,
    MCPToolSpec,
    StdioTransportConfig,
    StreamableHTTPTransportConfig,
)
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput


class _EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        return ToolOutput.success("ok")


class _FakeClient:
    def __init__(self, config: MCPServerConfig, *, event_handler=None) -> None:
        del event_handler
        self.config = config
        self.closed = False

    def connect(self) -> list[MCPToolSpec]:
        if self.config.name == "broken":
            raise MCPClientConnectionError("boom")
        return [
            MCPToolSpec(
                server_name=self.config.name,
                remote_name="search",
                title=None,
                description=f"{self.config.name} search",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            )
        ]

    def close(self) -> None:
        self.closed = True


class MCPManagerTests(unittest.TestCase):
    def test_soft_fails_broken_servers_and_keeps_local_tools(self) -> None:
        events: list[str] = []
        host = MCPHost(
            [
                MCPServerConfig(
                    name="figma",
                    transport=StreamableHTTPTransportConfig(
                        url="https://figma.example.com/mcp"
                    ),
                ),
                MCPServerConfig(
                    name="broken",
                    transport=StreamableHTTPTransportConfig(
                        url="https://broken.example.com/mcp"
                    ),
                ),
            ],
            event_handler=events.append,
            client_factory=_FakeClient,
        )

        registry = ToolRegistry()
        registry.register(_EchoTool())
        for tool in host.connect_all():
            registry.register(tool)

        self.assertIsNotNone(registry.get("echo"))
        self.assertIsNotNone(registry.get("mcp__figma__search"))
        self.assertIsNone(registry.get("mcp__broken__search"))
        self.assertTrue(any("已跳过" in event for event in events))
        statuses = {status.name: status for status in host.status_snapshot()}
        self.assertTrue(statuses["figma"].connected)
        self.assertFalse(statuses["broken"].connected)
        self.assertEqual(statuses["figma"].tool_count, 1)
        self.assertEqual(statuses["broken"].last_error, "boom")
        summary = host.summary()
        self.assertEqual(summary.connected_servers, 1)
        self.assertEqual(summary.failed_servers, 1)
        self.assertEqual(summary.tool_count, 1)

    def test_stdio_transport_registers_real_local_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            server_path = workspace / "stdio_server.py"
            server_path.write_text(
                textwrap.dedent(
                    """
                    from mcp.server.fastmcp import FastMCP

                    app = FastMCP("Local Demo")

                    @app.tool()
                    def add(a: int, b: int) -> int:
                        return a + b

                    if __name__ == "__main__":
                        app.run(transport="stdio")
                    """
                ).strip(),
                encoding="utf-8",
            )

            host = MCPHost(
                [
                    MCPServerConfig(
                        name="filesystem",
                        transport=StdioTransportConfig(
                            command=sys.executable,
                            args=(str(server_path),),
                            env={},
                        ),
                    )
                ]
            )
            try:
                tools = host.connect_all()
            finally:
                host.close()

        self.assertEqual([tool.name for tool in tools], ["mcp__filesystem__add"])


if __name__ == "__main__":
    unittest.main()
