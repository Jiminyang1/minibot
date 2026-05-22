"""Host-level orchestration for MiniBot MCP servers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .client import MCPClient, MCPClientError
from .config import load_mcp_config
from .models import (
    MCPConfigLoadResult,
    MCPHostSummary,
    MCPServerConfig,
    MCPServerStatus,
)
from .provider import MCPToolProxy
from .transport import create_mcp_client


class MCPHost:
    """Load config, connect enabled servers, and expose tool proxies."""

    def __init__(
        self,
        configs: list[MCPServerConfig],
        *,
        event_handler: Callable[[str], None] | None = None,
        client_factory: Callable[..., MCPClient] = create_mcp_client,
        config_path: Path | None = None,
    ) -> None:
        self._configs = list(configs)
        self._event_handler = event_handler
        self._client_factory = client_factory
        self._config_path = None if config_path is None else config_path.resolve()
        self._clients: list[MCPClient] = []
        self._tool_proxies: list[MCPToolProxy] | None = None
        self._statuses: dict[str, MCPServerStatus] = {
            config.name: MCPServerStatus(
                name=config.name,
                transport=config.transport.type,
                enabled=config.enabled,
                trusted=config.trusted,
            )
            for config in configs
        }

    @classmethod
    def from_config_root(
        cls,
        config_root: Path,
        *,
        event_handler: Callable[[str], None] | None = None,
        client_factory: Callable[..., MCPClient] = create_mcp_client,
    ) -> "MCPHost":
        loaded: MCPConfigLoadResult = load_mcp_config(config_root)
        config_path = config_root / "mcp.json"
        if event_handler is not None:
            for warning in loaded.warnings:
                event_handler(warning)
        return cls(
            loaded.servers,
            event_handler=event_handler,
            client_factory=client_factory,
            config_path=config_path if config_path.exists() else None,
        )

    def connect_all(self) -> list[MCPToolProxy]:
        """Eagerly connect enabled servers and build proxy tools."""
        if self._tool_proxies is not None:
            return list(self._tool_proxies)

        tool_proxies: list[MCPToolProxy] = []
        seen_names: set[str] = set()
        for config in self._configs:
            if not config.enabled:
                continue

            self._emit(
                f"MCP server `{config.name}` 正在连接 ({config.transport.type})..."
            )
            client = self._client_factory(config, event_handler=self._event_handler)
            try:
                tool_specs = client.connect()
            except MCPClientError as exc:
                self._statuses[config.name] = MCPServerStatus(
                    name=config.name,
                    transport=config.transport.type,
                    enabled=config.enabled,
                    trusted=config.trusted,
                    connected=False,
                    tool_count=0,
                    tool_names=(),
                    last_error=str(exc),
                )
                self._emit(f"MCP server `{config.name}` 初始化失败，已跳过: {exc}")
                client.close()
                continue

            self._clients.append(client)
            self._statuses[config.name] = MCPServerStatus(
                name=config.name,
                transport=config.transport.type,
                enabled=config.enabled,
                trusted=config.trusted,
                connected=True,
                tool_count=len(tool_specs),
                tool_names=tuple(
                    proxy_name
                    for proxy_name in (
                        f"mcp__{tool_spec.server_name}__{tool_spec.remote_name}"
                        for tool_spec in tool_specs
                    )
                ),
                last_error=None,
            )
            self._emit(
                f"MCP server `{config.name}` 已连接，发现 {len(tool_specs)} 个工具。"
            )
            for tool_spec in tool_specs:
                proxy = MCPToolProxy(
                    client=client,
                    tool_spec=tool_spec,
                    trusted=config.trusted,
                )
                if proxy.name in seen_names:
                    self._emit(f"MCP 工具名称冲突，已跳过: {proxy.name}")
                    continue
                seen_names.add(proxy.name)
                tool_proxies.append(proxy)

        self._tool_proxies = tool_proxies
        return list(tool_proxies)

    def status_snapshot(self) -> list[MCPServerStatus]:
        """Return one immutable status snapshot per configured server."""
        ordered_names = [config.name for config in self._configs]
        return [
            self._statuses[name]
            for name in ordered_names
            if name in self._statuses
        ]

    def summary(self) -> MCPHostSummary:
        """Return one aggregate MCP summary for UI and CLI display."""
        statuses = self.status_snapshot()
        enabled_servers = [status for status in statuses if status.enabled]
        connected_servers = [status for status in enabled_servers if status.connected]
        failed_servers = [
            status
            for status in enabled_servers
            if not status.connected and status.last_error is not None
        ]
        return MCPHostSummary(
            config_path=(
                None if self._config_path is None else str(self._config_path)
            ),
            configured_servers=len(statuses),
            enabled_servers=len(enabled_servers),
            connected_servers=len(connected_servers),
            failed_servers=len(failed_servers),
            tool_count=sum(status.tool_count for status in connected_servers),
        )

    def close(self) -> None:
        """Close all connected MCP clients."""
        for client in self._clients:
            client.close()
            status = self._statuses.get(client.config.name)
            if status is None:
                continue
            self._statuses[client.config.name] = MCPServerStatus(
                name=status.name,
                transport=status.transport,
                enabled=status.enabled,
                trusted=status.trusted,
                connected=False,
                tool_count=status.tool_count,
                tool_names=status.tool_names,
                last_error=status.last_error,
            )
        self._clients.clear()

    def _emit(self, message: str) -> None:
        if self._event_handler is not None:
            self._event_handler(message)
