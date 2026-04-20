"""Local stdio MCP transport implementation."""

from __future__ import annotations

from contextlib import AsyncExitStack
import sys
from typing import Any, Callable

from mcp.client.stdio import StdioServerParameters, stdio_client

from ..models import MCPServerConfig, StdioTransportConfig
from .base import AsyncSessionMCPClient


class _PrefixedErrLog:
    def __init__(self, server_name: str) -> None:
        self._server_name = server_name
        self._buffer = ""
        self.encoding = getattr(sys.stderr, "encoding", "utf-8")

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""
        sys.stderr.flush()

    def _emit_line(self, line: str) -> None:
        if not line.strip():
            return
        sys.stderr.write(f"[mcp:{self._server_name} stderr] {line}\n")

    def fileno(self) -> int:
        return sys.stderr.fileno()

    def isatty(self) -> bool:
        return sys.stderr.isatty()


class StdioMCPClient(AsyncSessionMCPClient):
    """Maintain one long-lived stdio MCP session behind sync methods."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        event_handler: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(config.transport, StdioTransportConfig):
            raise TypeError("StdioMCPClient 需要 stdio transport 配置。")
        super().__init__(config, event_handler=event_handler)
        self._transport = config.transport

    async def _open_streams_async(
        self,
        stack: AsyncExitStack,
    ) -> tuple[Any, Any]:
        params = StdioServerParameters(
            command=self._transport.command,
            args=list(self._transport.args),
            env=dict(self._transport.env) if self._transport.env else None,
            cwd=self._transport.cwd,
        )
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params, errlog=_PrefixedErrLog(self.config.name))
        )
        return read_stream, write_stream
