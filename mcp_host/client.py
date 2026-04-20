"""Transport-agnostic MCP client interface and shared errors."""

from __future__ import annotations

from typing import Any, Protocol

from .models import MCPServerConfig, MCPToolSpec


class MCPClientError(RuntimeError):
    """Base error for MiniBot MCP client failures."""


class MCPClientTimeoutError(MCPClientError):
    """Raised when an MCP operation times out."""


class MCPClientConnectionError(MCPClientError):
    """Raised when an MCP session cannot be used."""


class MCPClientNotFoundError(MCPClientError):
    """Raised when a discovered tool no longer exists on the server."""


class MCPClient(Protocol):
    """Sync client interface used by the MCP host and tool proxies."""

    config: MCPServerConfig

    def connect(self) -> list[MCPToolSpec]:
        """Connect to the target server and discover tools."""

    def list_tools(self) -> list[MCPToolSpec]:
        """Return the cached discovered tool catalog."""

    def call_tool(self, remote_name: str, arguments: dict[str, Any]) -> Any:
        """Call one discovered tool synchronously."""

    def close(self) -> None:
        """Release any held transport resources."""
