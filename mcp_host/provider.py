"""Expose MCP tools as regular MiniBot tools."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from mcp.types import AudioContent, CallToolResult, EmbeddedResource, ImageContent, TextContent

from ..tools.base import Tool, ToolExecutionContext
from ..tools.result import ToolOutput
from .client import (
    MCPClient,
    MCPClientError,
    MCPClientNotFoundError,
    MCPClientTimeoutError,
)
from .models import MCPToolSpec


def make_mcp_tool_name(server_name: str, remote_tool_name: str) -> str:
    """Return the locally exposed tool name for one MCP tool."""
    return f"mcp__{server_name}__{remote_tool_name}"


class MCPToolProxy(Tool):
    """Regular MiniBot tool that forwards to one MCP tool."""

    def __init__(
        self,
        *,
        client: MCPClient,
        tool_spec: MCPToolSpec,
        trusted: bool,
    ) -> None:
        super().__init__(workspace=None)
        self._client = client
        self._tool_spec = tool_spec
        self._trusted = trusted
        self._local_name = make_mcp_tool_name(
            tool_spec.server_name,
            tool_spec.remote_name,
        )

    @property
    def name(self) -> str:
        return self._local_name

    @property
    def description(self) -> str:
        return f"[MCP:{self._tool_spec.server_name}] {self._tool_spec.description}"

    @property
    def source(self) -> str:
        return "mcp"

    @property
    def display_name(self) -> str:
        return f"{self.server_name}.{self.remote_name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._tool_spec.input_schema)

    @property
    def server_name(self) -> str:
        return self._tool_spec.server_name

    @property
    def remote_name(self) -> str:
        return self._tool_spec.remote_name

    @property
    def transport_type(self) -> str:
        return self._client.config.transport.type

    @property
    def requires_approval(self) -> bool:
        return not self._trusted

    @property
    def exclusive(self) -> bool:
        return True

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        **kwargs: Any,
    ) -> ToolOutput:
        del context
        try:
            result = self._client.call_tool(self._tool_spec.remote_name, kwargs)
        except MCPClientTimeoutError as exc:
            return ToolOutput.failure(
                "timeout",
                str(exc),
                data={
                    "server": self._tool_spec.server_name,
                    "remote_tool": self._tool_spec.remote_name,
                },
            )
        except MCPClientNotFoundError as exc:
            return ToolOutput.failure(
                "not_found",
                str(exc),
                data={
                    "server": self._tool_spec.server_name,
                    "remote_tool": self._tool_spec.remote_name,
                },
            )
        except MCPClientError as exc:
            return ToolOutput.failure(
                "error",
                str(exc),
                data={
                    "server": self._tool_spec.server_name,
                    "remote_tool": self._tool_spec.remote_name,
                },
            )
        return _result_to_tool_output(self._tool_spec, result)


def _result_to_tool_output(
    tool_spec: MCPToolSpec,
    result: CallToolResult,
) -> ToolOutput:
    content_block_types = [_content_block_type(block) for block in result.content]
    has_structured = result.structuredContent is not None
    data = {
        "server": tool_spec.server_name,
        "remote_tool": tool_spec.remote_name,
        "content_block_types": content_block_types,
        "has_structured_content": has_structured,
    }

    summary_prefix = f"{tool_spec.server_name}.{tool_spec.remote_name}"
    content, content_kind, content_name = _materialize_result_content(tool_spec, result)

    if result.isError:
        return ToolOutput.failure(
            "error",
            f"{summary_prefix} 返回错误结果。",
            data=data,
            content=content,
            content_kind=content_kind,
            content_name=content_name,
        )

    return ToolOutput.success(
        f"{summary_prefix} 已执行。",
        data=data,
        content=content,
        content_kind=content_kind,
        content_name=content_name,
    )


def _materialize_result_content(
    tool_spec: MCPToolSpec,
    result: CallToolResult,
) -> tuple[str | None, str, str]:
    if result.content and _is_text_only(result) and result.structuredContent is None:
        text = "\n\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        return text, "text", f"{tool_spec.server_name}_{tool_spec.remote_name}.txt"

    payload = {
        "content": [_serialize_content_block(block) for block in result.content],
        "structured_content": result.structuredContent,
        "is_error": bool(result.isError),
    }
    return (
        json.dumps(payload, ensure_ascii=False, indent=2),
        "json",
        f"{tool_spec.server_name}_{tool_spec.remote_name}.json",
    )


def _is_text_only(result: CallToolResult) -> bool:
    return bool(result.content) and all(
        isinstance(block, TextContent) for block in result.content
    )


def _content_block_type(block: Any) -> str:
    block_type = getattr(block, "type", None)
    return block_type if isinstance(block_type, str) else type(block).__name__


def _serialize_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, (TextContent, ImageContent, AudioContent, EmbeddedResource)):
        return block.model_dump(mode="json", exclude_none=True, by_alias=True)
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True, by_alias=True)
    return {"type": type(block).__name__, "value": repr(block)}
