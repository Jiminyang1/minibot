from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.types import CallToolResult, TextContent

from minibot.mcp_host.client import (
    MCPClientConnectionError,
    MCPClientTimeoutError,
)
from minibot.mcp_host.models import (
    MCPServerConfig,
    MCPToolSpec,
    StdioTransportConfig,
    StreamableHTTPTransportConfig,
)
from minibot.mcp_host.transport.stdio import StdioMCPClient
from minibot.mcp_host.transport.streamable_http import StreamableHTTPMCPClient


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_http_server(url: str, *, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with request.urlopen(url, timeout=0.5):
                return
        except error.HTTPError:
            return
        except Exception:
            time.sleep(0.1)
    raise AssertionError(f"测试 MCP server 未能及时启动: {url}")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _extract_result_payload(result: CallToolResult) -> object:
    if result.structuredContent is not None:
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload
    texts = [
        block.text
        for block in result.content
        if isinstance(block, TextContent)
    ]
    if len(texts) != 1:
        raise AssertionError("测试结果缺少可解析的 structuredContent/text。")
    payload = json.loads(texts[0])
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


class _StubStreamableHTTPMCPClient(StreamableHTTPMCPClient):
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        connect_side_effects: list[object],
        call_side_effects: list[object],
    ) -> None:
        super().__init__(config)
        self._connect_side_effects = list(connect_side_effects)
        self._call_side_effects = list(call_side_effects)
        self.connect_attempts = 0
        self.call_attempts = 0

    def _connect_once(self) -> list[MCPToolSpec]:
        self.connect_attempts += 1
        effect = self._connect_side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    def _call_tool_once(self, remote_name: str, arguments: dict[str, object]) -> CallToolResult:
        del remote_name, arguments
        self.call_attempts += 1
        effect = self._call_side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class MCPClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MCPServerConfig(
            name="figma",
            transport=StreamableHTTPTransportConfig(
                url="https://figma.example.com/mcp",
                headers={},
            ),
            enabled=True,
            trusted=False,
            timeout_seconds=30,
        )
        self.tool_specs = [
            MCPToolSpec(
                server_name="figma",
                remote_name="comment",
                title=None,
                description="Create a comment",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            )
        ]

    def test_reconnects_once_on_connection_error_and_retries_call(self) -> None:
        client = _StubStreamableHTTPMCPClient(
            self.config,
            connect_side_effects=[self.tool_specs, self.tool_specs],
            call_side_effects=[
                MCPClientConnectionError("dropped"),
                CallToolResult(content=[TextContent(type="text", text="ok")]),
            ],
        )
        client.connect()

        result = client.call_tool("comment", {})

        self.assertEqual(client.connect_attempts, 2)
        self.assertEqual(client.call_attempts, 2)
        self.assertFalse(result.isError)
        self.assertEqual(result.content[0].text, "ok")

    def test_timeout_does_not_trigger_reconnect(self) -> None:
        client = _StubStreamableHTTPMCPClient(
            self.config,
            connect_side_effects=[self.tool_specs],
            call_side_effects=[MCPClientTimeoutError("too slow")],
        )
        client.connect()

        with self.assertRaises(MCPClientTimeoutError):
            client.call_tool("comment", {})

        self.assertEqual(client.connect_attempts, 1)
        self.assertEqual(client.call_attempts, 1)


class StdioMCPClientIntegrationTests(unittest.TestCase):
    def test_stdio_client_discovers_and_calls_real_local_server(self) -> None:
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

            client = StdioMCPClient(
                MCPServerConfig(
                    name="local_demo",
                    transport=StdioTransportConfig(
                        command=sys.executable,
                        args=(str(server_path),),
                        env={},
                    ),
                    timeout_seconds=30,
                )
            )
            try:
                tools = client.connect()
                result = client.call_tool("add", {"a": 2, "b": 3})
            finally:
                client.close()

        self.assertEqual([tool.remote_name for tool in tools], ["add"])
        self.assertFalse(result.isError)
        self.assertTrue(result.content or result.structuredContent is not None)

    def test_stdio_sqlite_server_reads_real_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            database_path = workspace / "demo.sqlite3"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        active INTEGER NOT NULL
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO users (name, active) VALUES (?, ?)",
                    [("Ada", 1), ("Linus", 0)],
                )
                connection.commit()

            server_path = (
                Path(__file__).resolve().parents[1]
                / "mcp_servers"
                / "sqlite_server.py"
            )
            client = StdioMCPClient(
                MCPServerConfig(
                    name="sqlite_demo",
                    transport=StdioTransportConfig(
                        command=sys.executable,
                        args=(str(server_path),),
                        env={"SQLITE_PATH": str(database_path)},
                    ),
                    timeout_seconds=30,
                )
            )
            try:
                tools = client.connect()
                tables_result = client.call_tool("list_tables", {})
                schema_result = client.call_tool("describe_table", {"table": "users"})
                query_result = client.call_tool(
                    "query",
                    {
                        "sql": "SELECT id, name, active FROM users ORDER BY id",
                        "limit": 1,
                    },
                )
            finally:
                client.close()

        self.assertEqual(
            sorted(tool.remote_name for tool in tools),
            ["describe_table", "list_tables", "query"],
        )
        self.assertEqual(_extract_result_payload(tables_result), ["users"])

        schema_payload = _extract_result_payload(schema_result)
        self.assertIsInstance(schema_payload, dict)
        assert isinstance(schema_payload, dict)
        self.assertEqual(schema_payload["table"], "users")
        self.assertEqual(
            [column["name"] for column in schema_payload["columns"]],
            ["id", "name", "active"],
        )

        query_payload = _extract_result_payload(query_result)
        self.assertIsInstance(query_payload, dict)
        assert isinstance(query_payload, dict)
        self.assertEqual(query_payload["columns"], ["id", "name", "active"])
        self.assertEqual(
            query_payload["rows"],
            [{"id": 1, "name": "Ada", "active": 1}],
        )
        self.assertEqual(query_payload["row_count"], 1)
        self.assertTrue(query_payload["truncated"])


class StreamableHTTPMCPClientIntegrationTests(unittest.TestCase):
    def test_streamable_http_client_discovers_and_calls_real_remote_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            port = _find_free_port()
            server_path = workspace / "streamable_http_server.py"
            server_path.write_text(
                textwrap.dedent(
                    f"""
                    from mcp.server.fastmcp import FastMCP

                    app = FastMCP("Remote Demo", host="127.0.0.1", port={port})

                    @app.tool()
                    def add(a: int, b: int) -> int:
                        return a + b

                    if __name__ == "__main__":
                        app.run(transport="streamable-http")
                    """
                ).strip(),
                encoding="utf-8",
            )

            process = subprocess.Popen(
                [sys.executable, str(server_path)],
                cwd=workspace,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            _wait_for_http_server(f"http://127.0.0.1:{port}/mcp")

            client = StreamableHTTPMCPClient(
                MCPServerConfig(
                    name="remote_demo",
                    transport=StreamableHTTPTransportConfig(
                        url=f"http://127.0.0.1:{port}/mcp",
                        headers={},
                    ),
                    timeout_seconds=30,
                )
            )
            try:
                tools = client.connect()
                result = client.call_tool("add", {"a": 2, "b": 3})
            finally:
                client.close()
                _stop_process(process)

        self.assertEqual([tool.remote_name for tool in tools], ["add"])
        self.assertFalse(result.isError)
        self.assertTrue(result.content or result.structuredContent is not None)


if __name__ == "__main__":
    unittest.main()
