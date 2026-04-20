"""Shared models for MiniBot MCP host integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


TransportType: TypeAlias = Literal["streamable_http", "stdio"]


@dataclass(frozen=True)
class StreamableHTTPTransportConfig:
    """Remote Streamable HTTP transport configuration."""

    type: Literal["streamable_http"] = "streamable_http"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StdioTransportConfig:
    """Local stdio transport configuration."""

    type: Literal["stdio"] = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


MCPTransportConfig: TypeAlias = StreamableHTTPTransportConfig | StdioTransportConfig


@dataclass(frozen=True)
class MCPServerConfig:
    """Validated repo-local configuration for one MCP server."""

    name: str
    transport: MCPTransportConfig
    enabled: bool = True
    trusted: bool = False
    timeout_seconds: int = 30


@dataclass(frozen=True)
class MCPConfigLoadResult:
    """Outcome of parsing ``mcp.json``."""

    servers: list[MCPServerConfig] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MCPToolSpec:
    """One MCP tool discovered from a server."""

    server_name: str
    remote_name: str
    title: str | None
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPServerStatus:
    """Current host-side view of one configured MCP server."""

    name: str
    transport: TransportType
    enabled: bool
    trusted: bool
    connected: bool = False
    tool_count: int = 0
    tool_names: tuple[str, ...] = ()
    last_error: str | None = None


@dataclass(frozen=True)
class MCPHostSummary:
    """Small aggregate summary for UI and CLI display."""

    config_path: str | None
    configured_servers: int
    enabled_servers: int
    connected_servers: int
    failed_servers: int
    tool_count: int
