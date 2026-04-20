"""Shared async-session MCP client lifecycle."""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from copy import deepcopy
from datetime import timedelta
import threading
from typing import Any, Callable

from mcp import ClientSession
from mcp.types import ListToolsResult, Tool

from ..client import (
    MCPClientConnectionError,
    MCPClientError,
    MCPClientNotFoundError,
    MCPClientTimeoutError,
)
from ..models import MCPServerConfig, MCPToolSpec


class AsyncSessionMCPClient:
    """Shared sync facade for async MCP session transports."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        event_handler: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._event_handler = event_handler
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._loop_lock = threading.Lock()
        self._closed = False
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools_by_name: dict[str, MCPToolSpec] = {}

    def connect(self) -> list[MCPToolSpec]:
        """Connect to the target server and eagerly fetch its tool catalog."""
        tool_specs = self._connect_once()
        self._set_tool_specs(tool_specs)
        return self.list_tools()

    def list_tools(self) -> list[MCPToolSpec]:
        """Return the cached tool catalog discovered at connect time."""
        return list(self._tools_by_name.values())

    def call_tool(self, remote_name: str, arguments: dict[str, Any]) -> Any:
        """Call one remote tool synchronously and reconnect once if needed."""
        if remote_name not in self._tools_by_name:
            raise MCPClientNotFoundError(
                f"MCP 工具 {self.config.name}.{remote_name} 不存在。"
            )
        try:
            return self._call_tool_once(remote_name, arguments)
        except MCPClientTimeoutError:
            self._emit(
                f"MCP 调用超时: {self.config.name}.{remote_name} "
                f"({self.config.timeout_seconds}s)"
            )
            raise
        except MCPClientConnectionError as exc:
            self._emit(
                f"MCP server `{self.config.name}` 调用失败，准备重连重试: {exc}"
            )
            try:
                refreshed = self._connect_once()
            except MCPClientError as reconnect_exc:
                raise MCPClientConnectionError(
                    f"MCP server `{self.config.name}` 重连失败: {reconnect_exc}"
                ) from reconnect_exc
            self._set_tool_specs(refreshed)
            self._emit(
                f"MCP server `{self.config.name}` 重连成功，重新发现 "
                f"{len(refreshed)} 个工具。"
            )
            if remote_name not in self._tools_by_name:
                raise MCPClientNotFoundError(
                    f"MCP 工具 {self.config.name}.{remote_name} 在重连后已不可用。"
                ) from exc
            return self._call_tool_once(remote_name, arguments)

    def close(self) -> None:
        """Close the MCP session and stop the background event loop."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            self._closed = True
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._disconnect_async(),
                loop,
            )
            future.result(timeout=self.config.timeout_seconds)
        except MCPClientError:
            pass
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        self._thread = None
        self._loop = None
        self._loop_ready.clear()
        self._closed = True

    def _connect_once(self) -> list[MCPToolSpec]:
        timeout = self._operation_timeout_seconds
        try:
            return self._run_coro(self._connect_async(), timeout_seconds=timeout)
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientConnectionError(
                f"MCP server `{self.config.name}` 初始化失败: {exc}"
            ) from exc

    def _call_tool_once(self, remote_name: str, arguments: dict[str, Any]) -> Any:
        timeout = self._operation_timeout_seconds
        try:
            return self._run_coro(
                self._call_tool_async(remote_name, arguments),
                timeout_seconds=timeout,
            )
        except MCPClientTimeoutError:
            raise
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientConnectionError(
                f"MCP 工具 {self.config.name}.{remote_name} 调用异常: {exc}"
            ) from exc

    @property
    def _operation_timeout_seconds(self) -> int:
        return max(self.config.timeout_seconds + 5, self.config.timeout_seconds)

    def _set_tool_specs(self, tool_specs: list[MCPToolSpec]) -> None:
        self._tools_by_name = {spec.remote_name: spec for spec in tool_specs}

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._closed:
                raise MCPClientConnectionError(
                    f"MCP server `{self.config.name}` 已关闭。"
                )
            if self._loop is not None and self._thread is not None:
                return self._loop

            self._loop_ready.clear()
            thread = threading.Thread(
                target=self._run_loop,
                name=f"minibot-mcp-{self.config.name}",
                daemon=True,
            )
            thread.start()
            self._thread = thread
        if not self._loop_ready.wait(timeout=2):
            raise MCPClientConnectionError(
                f"MCP server `{self.config.name}` 后台事件循环未能及时启动。"
            )
        assert self._loop is not None
        return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def _run_coro(self, coro: Any, *, timeout_seconds: int) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise MCPClientTimeoutError(
                f"MCP server `{self.config.name}` 请求超过 {self.config.timeout_seconds} 秒。"
            ) from exc
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientConnectionError(str(exc)) from exc

    async def _connect_async(self) -> list[MCPToolSpec]:
        await self._disconnect_async()
        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._open_streams_async(stack)
            session = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
            )
            session = await stack.enter_async_context(session)
            await session.initialize()
            tool_specs = await self._list_tools_async(session)
        except Exception:
            await stack.aclose()
            raise

        self._exit_stack = stack
        self._session = session
        return tool_specs

    async def _disconnect_async(self) -> None:
        stack = self._exit_stack
        self._exit_stack = None
        self._session = None
        if stack is not None:
            await stack.aclose()

    async def _list_tools_async(
        self,
        session: ClientSession,
    ) -> list[MCPToolSpec]:
        tools: list[MCPToolSpec] = []
        cursor: str | None = None
        while True:
            page: ListToolsResult = await session.list_tools(cursor=cursor)
            tools.extend(self._convert_tool(tool) for tool in page.tools)
            cursor = page.nextCursor
            if not cursor:
                break
        return tools

    async def _call_tool_async(
        self,
        remote_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        if self._session is None:
            raise MCPClientConnectionError(
                f"MCP server `{self.config.name}` 尚未建立连接。"
            )
        return await self._session.call_tool(
            remote_name,
            arguments,
            read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
        )

    def _convert_tool(self, tool: Tool) -> MCPToolSpec:
        input_schema = tool.inputSchema
        if isinstance(input_schema, dict):
            schema = deepcopy(input_schema)
        else:
            schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        return MCPToolSpec(
            server_name=self.config.name,
            remote_name=tool.name,
            title=tool.title,
            description=tool.description or tool.title or tool.name,
            input_schema=schema,
        )

    async def _open_streams_async(
        self,
        stack: AsyncExitStack,
    ) -> tuple[Any, Any]:
        """Open transport streams and register them on ``stack``."""
        raise NotImplementedError

    def _emit(self, message: str) -> None:
        if self._event_handler is not None:
            self._event_handler(message)
