"""Composition root helpers for MiniBot runtimes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from .artifacts import ArtifactStore
from .config import Config
from .llm import OpenAIClient
from .mcp_host import MCPHost
from .prompts import SYSTEM_PROMPT
from .run_log import RunLogStore
from .runtime import (
    AgentRunner,
    ApprovalBroker,
    ApprovalRequest,
    ContextManager,
    RunController,
    RuntimeEventHandler,
    ToolOutputMaterializer,
    TurnEngine,
    make_summarizer,
)
from .session import SessionManager
from .skills import SkillRegistry
from .tools import (
    ToolRegistry,
    filesystem_toolset,
    memory_toolset,
    network_toolset,
    shell_toolset,
    skill_toolset,
)
from .user_memory import UserMemoryStore

_GLOBAL_MCP_CONFIG_ENV = "MINIBOT_MCP_CONFIG_PATH"


@dataclass(frozen=True)
class MiniBotRuntime:
    config: Config
    manager: SessionManager
    memory_store: UserMemoryStore
    artifact_store: ArtifactStore
    run_log_store: RunLogStore
    skill_registry: SkillRegistry
    tool_registry: ToolRegistry
    mcp_host: MCPHost
    context_manager: ContextManager
    runner: AgentRunner
    turn_engine: TurnEngine
    controller: RunController
    approval_broker: ApprovalBroker | None = None

    def close(self) -> None:
        self.mcp_host.close()


def build_runtime(
    *,
    config: Config,
    workspace: Path | None = None,
    run_event_handler: RuntimeEventHandler | None = None,
    log_handler: Callable[[str], None] | None = None,
    approval_handler: Callable[[ApprovalRequest], bool] | None = None,
    approval_broker: ApprovalBroker | None = None,
) -> MiniBotRuntime:
    package_dir = Path(__file__).resolve().parent
    resolved_workspace = (workspace or Path.cwd()).resolve()
    os.environ.setdefault("MINIBOT_PYTHON", sys.executable)
    os.environ.setdefault("MINIBOT_PACKAGE_DIR", str(package_dir))

    manager = SessionManager(resolved_workspace)
    artifact_store = ArtifactStore(resolved_workspace)
    memory_store = UserMemoryStore()
    run_log_store = RunLogStore(resolved_workspace)
    llm = OpenAIClient(model=config.model)
    skill_registry = SkillRegistry.from_directory(package_dir / "skills")

    tool_registry = ToolRegistry()
    tool_registry.register_all(filesystem_toolset(resolved_workspace, artifact_store))
    tool_registry.register_all(shell_toolset(resolved_workspace))
    tool_registry.register_all(network_toolset())
    tool_registry.register_all(memory_toolset(memory_store))
    tool_registry.register_all(skill_toolset(skill_registry))

    mcp_config_root, mcp_config_path, mcp_config_source = _resolve_mcp_config(
        package_dir,
    )
    if log_handler is not None:
        if mcp_config_path is None:
            log_handler("未找到全局 MCP 配置，启动时不加载 MCP server。")
        else:
            log_handler(
                f"MCP 配置路径 ({mcp_config_source}): {mcp_config_path}"
            )

    mcp_host = MCPHost.from_config_root(
        mcp_config_root,
        event_handler=log_handler,
    )
    for tool in mcp_host.connect_all():
        if tool_registry.get(tool.name) is not None:
            if log_handler is not None:
                log_handler(f"MCP 工具名称冲突，已跳过: {tool.name}")
            continue
        tool_registry.register(tool)

    summarizer = make_summarizer(llm)
    context_manager = ContextManager(
        base_system_prompt=SYSTEM_PROMPT,
        memory_store=memory_store,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        max_history_turns=config.max_history_turns,
        compact_token_threshold=config.compact_token_threshold,
        reserved_completion_tokens=config.reserved_completion_tokens,
        compact_keep_recent=config.compact_keep_recent,
        summarizer=summarizer,
        include_reasoning_content=_should_include_reasoning_content(config.model),
    )
    runner = AgentRunner(
        llm,
        tool_registry,
        materializer=ToolOutputMaterializer(artifact_store),
        event_handler=run_event_handler,
        approval_handler=approval_handler,
        approval_mode=config.approval_mode,
        max_parallel_tools=config.max_parallel_tools,
    )
    turn_engine = TurnEngine(
        runner,
        manager,
        config,
        context_manager=context_manager,
        event_handler=run_event_handler,
        run_log_store=run_log_store,
    )
    controller = RunController(
        turn_engine=turn_engine,
        manager=manager,
        approval_broker=approval_broker,
    )
    return MiniBotRuntime(
        config=config,
        manager=manager,
        memory_store=memory_store,
        artifact_store=artifact_store,
        run_log_store=run_log_store,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        mcp_host=mcp_host,
        context_manager=context_manager,
        runner=runner,
        turn_engine=turn_engine,
        controller=controller,
        approval_broker=approval_broker,
    )


def _should_include_reasoning_content(model: str) -> bool:
    raw = os.environ.get("MINIBOT_INCLUDE_REASONING_CONTENT", "auto").strip().lower()
    if raw in {"1", "true", "yes", "always"}:
        return True
    if raw in {"0", "false", "no", "never"}:
        return False

    base_url = os.environ.get("OPENAI_BASE_URL", "").lower()
    model_name = model.lower()
    return "deepseek" in base_url or model_name.startswith("deepseek-")


def _resolve_mcp_config(package_dir: Path) -> tuple[Path, Path | None, str]:
    """Resolve MiniBot-global MCP config.

    MCP tools are global to the MiniBot installation, not scoped to the
    project/workspace where the user starts the CLI. Relative stdio paths are
    still resolved by ``mcp_host.config`` against the returned config root.
    """
    package_dir = package_dir.resolve()
    explicit = os.environ.get(_GLOBAL_MCP_CONFIG_ENV, "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        config_path = (
            explicit_path
            if explicit_path.name == "mcp.json"
            else explicit_path / "mcp.json"
        )
        return config_path.parent, config_path if config_path.exists() else None, "env"

    user_config_path = (Path.home() / ".minibot" / "mcp.json").resolve()
    if user_config_path.exists():
        return user_config_path.parent, user_config_path, "user"

    package_config_path = (package_dir / "mcp.json").resolve()
    if package_config_path.exists():
        return package_dir, package_config_path, "bundled"

    return package_dir, None, "none"
