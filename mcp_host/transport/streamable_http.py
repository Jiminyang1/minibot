"""Streamable HTTP MCP client implementation."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Callable

import httpx
from mcp.client.streamable_http import streamable_http_client

from ..models import MCPServerConfig, StreamableHTTPTransportConfig
from .base import AsyncSessionMCPClient


class StreamableHTTPMCPClient(AsyncSessionMCPClient):
    """Maintain one long-lived Streamable HTTP MCP session behind sync methods."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        event_handler: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(config.transport, StreamableHTTPTransportConfig):
            raise TypeError("StreamableHTTPMCPClient 需要 streamable_http transport 配置。")
        super().__init__(config, event_handler=event_handler)
        self._transport = config.transport
    async def _open_streams_async(
        self,
        stack: AsyncExitStack,
    ) -> tuple[Any, Any]:
        http_client = httpx.AsyncClient(
            headers=self._transport.headers or None,
            timeout=httpx.Timeout(
                self.config.timeout_seconds,
                connect=self.config.timeout_seconds,
                read=self.config.timeout_seconds,
                write=self.config.timeout_seconds,
                pool=self.config.timeout_seconds,
            ),
            follow_redirects=True,
        )
        http_client = await stack.enter_async_context(http_client)
        read_stream, write_stream, _get_session_id = await stack.enter_async_context(
            streamable_http_client(self._transport.url, http_client=http_client)
        )
        return read_stream, write_stream
