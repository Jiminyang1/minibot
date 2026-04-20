"""Transport-agnostic MCP host integration for MiniBot."""

from .client import (
    MCPClient,
    MCPClientConnectionError,
    MCPClientError,
    MCPClientNotFoundError,
    MCPClientTimeoutError,
)
from .config import load_mcp_config
from .host import MCPHost
from .models import (
    MCPConfigLoadResult,
    MCPHostSummary,
    MCPServerConfig,
    MCPServerStatus,
    MCPToolSpec,
    StdioTransportConfig,
    StreamableHTTPTransportConfig,
)
from .provider import MCPToolProxy

__all__ = [
    "MCPClient",
    "MCPClientConnectionError",
    "MCPClientError",
    "MCPClientNotFoundError",
    "MCPClientTimeoutError",
    "load_mcp_config",
    "MCPHost",
    "MCPConfigLoadResult",
    "MCPHostSummary",
    "MCPServerConfig",
    "MCPServerStatus",
    "MCPToolSpec",
    "StdioTransportConfig",
    "StreamableHTTPTransportConfig",
    "MCPToolProxy",
]
