"""Concrete MCP transport clients."""

from __future__ import annotations

from typing import Callable

from ..client import MCPClient
from ..models import MCPServerConfig, StdioTransportConfig, StreamableHTTPTransportConfig
from .stdio import StdioMCPClient
from .streamable_http import StreamableHTTPMCPClient


def create_mcp_client(
    config: MCPServerConfig,
    *,
    event_handler: Callable[[str], None] | None = None,
) -> MCPClient:
    """Instantiate the concrete client for one configured transport."""
    if isinstance(config.transport, StreamableHTTPTransportConfig):
        return StreamableHTTPMCPClient(config, event_handler=event_handler)
    if isinstance(config.transport, StdioTransportConfig):
        return StdioMCPClient(config, event_handler=event_handler)
    raise RuntimeError(f"未知 MCP transport: {type(config.transport).__name__}")


__all__ = [
    "create_mcp_client",
    "StdioMCPClient",
    "StreamableHTTPMCPClient",
]
