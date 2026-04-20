"""Repo-local ``mcp.json`` loading for MiniBot MCP host."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .models import (
    MCPConfigLoadResult,
    MCPServerConfig,
    StdioTransportConfig,
    StreamableHTTPTransportConfig,
)

_CONFIG_FILENAME = "mcp.json"
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def load_mcp_config(workspace: Path) -> MCPConfigLoadResult:
    """Load and validate repo-local MCP config from ``workspace/mcp.json``."""
    path = workspace / _CONFIG_FILENAME
    if not path.exists():
        return MCPConfigLoadResult()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return MCPConfigLoadResult(
            warnings=[f"MCP 配置加载失败 ({path.name}): {exc}"]
        )

    if not isinstance(raw, dict):
        return MCPConfigLoadResult(
            warnings=[f"MCP 配置格式错误 ({path.name}): 顶层必须是对象。"]
        )

    raw_servers = raw.get("servers")
    if raw_servers is None:
        return MCPConfigLoadResult(
            warnings=[f"MCP 配置格式错误 ({path.name}): 缺少 `servers` 字段。"]
        )
    if not isinstance(raw_servers, list):
        return MCPConfigLoadResult(
            warnings=[f"MCP 配置格式错误 ({path.name}): `servers` 必须是数组。"]
        )

    servers: list[MCPServerConfig] = []
    warnings: list[str] = []
    seen_names: set[str] = set()
    for index, raw_entry in enumerate(raw_servers, start=1):
        parsed, entry_warnings = _parse_server_entry(
            raw_entry,
            index=index,
            workspace=workspace,
        )
        warnings.extend(entry_warnings)
        if parsed is None:
            continue
        if parsed.name in seen_names:
            warnings.append(
                f"MCP server `{parsed.name}` 重复定义，已跳过后续条目。"
            )
            continue
        seen_names.add(parsed.name)
        servers.append(parsed)

    return MCPConfigLoadResult(servers=servers, warnings=warnings)


def _parse_server_entry(
    raw_entry: Any,
    *,
    index: int,
    workspace: Path,
) -> tuple[MCPServerConfig | None, list[str]]:
    prefix = f"MCP server[{index}]"
    if not isinstance(raw_entry, dict):
        return None, [f"{prefix} 格式错误: 必须是对象。"]

    name = raw_entry.get("name")
    enabled = raw_entry.get("enabled", True)
    trusted = raw_entry.get("trusted", False)
    timeout_seconds = raw_entry.get("timeout_seconds", 30)
    raw_transport = raw_entry.get("transport")

    if not isinstance(name, str) or not name.strip():
        return None, [f"{prefix} 缺少合法 `name`。"]
    if not _SERVER_NAME_PATTERN.match(name):
        return None, [f"{prefix} 的 `name` 仅允许字母、数字、下划线和短横线。"]
    if not isinstance(enabled, bool):
        return None, [f"{prefix} 的 `enabled` 必须是布尔值。"]
    if not isinstance(trusted, bool):
        return None, [f"{prefix} 的 `trusted` 必须是布尔值。"]
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        return None, [f"{prefix} 的 `timeout_seconds` 必须是正整数。"]
    if not isinstance(raw_transport, dict):
        return None, [f"{prefix} 的 `transport` 必须是对象。"]

    transport, transport_warnings = _parse_transport_entry(
        raw_transport,
        prefix=prefix,
        enabled=enabled,
        workspace=workspace,
    )
    if transport is None:
        return None, transport_warnings

    return (
        MCPServerConfig(
            name=name,
            transport=transport,
            enabled=enabled,
            trusted=trusted,
            timeout_seconds=timeout_seconds,
        ),
        transport_warnings,
    )


def _parse_transport_entry(
    raw_transport: dict[str, Any],
    *,
    prefix: str,
    enabled: bool,
    workspace: Path,
) -> tuple[StreamableHTTPTransportConfig | StdioTransportConfig | None, list[str]]:
    transport_type = raw_transport.get("type")
    if not isinstance(transport_type, str) or not transport_type.strip():
        return None, [f"{prefix} 的 `transport.type` 必须是非空字符串。"]

    if transport_type == "streamable_http":
        url = raw_transport.get("url")
        raw_headers = raw_transport.get("headers", {})
        if not isinstance(url, str) or not url.strip():
            return None, [f"{prefix} 的 `transport.url` 必须是非空字符串。"]
        if not isinstance(raw_headers, dict):
            return None, [f"{prefix} 的 `transport.headers` 必须是对象。"]
        headers, warnings = _parse_string_map(
            raw_headers,
            label=f"{prefix} 的 transport header",
            enabled=enabled,
        )
        if headers is None:
            return None, warnings
        return (
            StreamableHTTPTransportConfig(url=url, headers=headers),
            warnings,
        )

    if transport_type == "stdio":
        command = raw_transport.get("command")
        raw_args = raw_transport.get("args", [])
        raw_env = raw_transport.get("env", {})
        raw_cwd = raw_transport.get("cwd")
        if not isinstance(command, str) or not command.strip():
            return None, [f"{prefix} 的 `transport.command` 必须是非空字符串。"]
        if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
            return None, [f"{prefix} 的 `transport.args` 必须是字符串数组。"]
        if not isinstance(raw_env, dict):
            return None, [f"{prefix} 的 `transport.env` 必须是对象。"]
        if raw_cwd is not None and not isinstance(raw_cwd, str):
            return None, [f"{prefix} 的 `transport.cwd` 必须是字符串。"]
        env, warnings = _parse_string_map(
            raw_env,
            label=f"{prefix} 的 transport env",
            enabled=enabled,
        )
        if env is None:
            return None, warnings
        cwd = None
        if raw_cwd is not None and raw_cwd.strip():
            try:
                cwd = _resolve_stdio_cwd(
                    raw_cwd,
                    workspace=workspace,
                    enabled=enabled,
                )
            except ValueError as exc:
                return None, [f"{prefix} 的 `transport.cwd` 无法解析: {exc}"]
        return (
            StdioTransportConfig(
                command=command,
                args=tuple(raw_args),
                env=env,
                cwd=cwd,
            ),
            warnings,
        )

    return None, [f"{prefix} 的 `transport.type` 不受支持: {transport_type}"]


def _parse_string_map(
    raw_mapping: dict[str, Any],
    *,
    label: str,
    enabled: bool,
) -> tuple[dict[str, str] | None, list[str]]:
    mapping: dict[str, str] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str) or not key.strip():
            return None, [f"{label} 名称必须是非空字符串。"]
        if not isinstance(value, str):
            return None, [f"{label} `{key}` 必须是字符串。"]
        if enabled:
            try:
                mapping[key] = _resolve_env_placeholders(value)
            except ValueError as exc:
                return None, [f"{label} `{key}` 无法解析: {exc}"]
        else:
            mapping[key] = value
    return mapping, []


def _resolve_env_placeholders(value: str) -> str:
    """Resolve ``${ENV_VAR}`` placeholders inside one config string."""

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None or env_value == "":
            raise ValueError(f"环境变量 `{env_name}` 未设置。")
        return env_value

    return _ENV_PATTERN.sub(replace, value)


def _resolve_stdio_cwd(value: str, *, workspace: Path, enabled: bool) -> str:
    resolved = _resolve_env_placeholders(value) if enabled else value
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        path = (workspace / path).resolve()
    else:
        path = path.resolve()
    return str(path)
