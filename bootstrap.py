"""Composition root helpers for MiniBot runtimes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading

from .artifacts import ArtifactStore
from .config import Config, resolve_state_home
from .llm_factory import build_llm_client_from_profile
from .llm_profile import build_llm_profile
from .mcp_host.host import MCPHost
from .prompts import SYSTEM_PROMPT
from .run_log import RunLogStore
from .runtime.agent_loop import AgentLoop
from .runtime.agent_session import AgentSession
from .runtime.approval import ApprovalPolicy, ApprovalRequest, ToolApprovalGate
from .runtime.approvals import ApprovalBroker
from .runtime.budget import TokenBudget
from .runtime.compactor import Compactor, make_summarizer
from .runtime.context_builder import ContextBuilder
from .runtime.events import RuntimeEventHandler, fanout
from .runtime.run_log_fold import RunLogFold
from .runtime.tool_output_materializer import ToolOutputMaterializer
from .schedule_store import ScheduleStore
from .session import SessionManager
from .skills import SkillRegistry
from .tools import (
    ToolRegistry,
    filesystem_toolset,
    history_toolset,
    memory_toolset,
    network_toolset,
    schedule_toolset,
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
    schedule_store: ScheduleStore
    skill_registry: SkillRegistry
    tool_registry: ToolRegistry
    mcp_host: MCPHost
    context_builder: ContextBuilder
    budget: TokenBudget
    compactor: Compactor
    agent_loop: AgentLoop
    agent_session: AgentSession
    approval_policy: ApprovalPolicy
    approval_broker: ApprovalBroker | None = None

    def close(self) -> None:
        self.mcp_host.close()


def build_runtime(
    *,
    config: Config,
    workspace: Path | None = None,
    run_event_handler: RuntimeEventHandler | None = None,
    log_handler: Callable[[str], None] | None = None,
    approval_handler: Callable[[ApprovalRequest, threading.Event | None], bool] | None = None,
    approval_broker: ApprovalBroker | None = None,
) -> MiniBotRuntime:
    package_dir = Path(__file__).resolve().parent
    resolved_workspace = (workspace or Path.cwd()).resolve()
    os.environ.setdefault("MINIBOT_PYTHON", sys.executable)
    os.environ.setdefault("MINIBOT_PACKAGE_DIR", str(package_dir))

    # State is global (assistant memory belongs to the user); the workspace
    # only scopes tools and is stamped onto sessions as provenance metadata.
    state_home = resolve_state_home()
    manager = SessionManager(state_home, default_workspace=resolved_workspace)
    artifact_store = ArtifactStore(state_home)
    memory_store = UserMemoryStore(state_home)
    run_log_store = RunLogStore(state_home)
    schedule_store = ScheduleStore(state_home)
    llm_profile = build_llm_profile(model=config.model)
    llm = build_llm_client_from_profile(llm_profile)
    skill_registry = SkillRegistry.from_directory(package_dir / "skills")

    tool_registry = ToolRegistry()
    tool_registry.register_all(filesystem_toolset(resolved_workspace, artifact_store))
    tool_registry.register_all(shell_toolset(resolved_workspace))
    tool_registry.register_all(network_toolset())
    tool_registry.register_all(memory_toolset(memory_store))
    tool_registry.register_all(skill_toolset(skill_registry))
    tool_registry.register_all(history_toolset(manager))
    tool_registry.register_all(
        schedule_toolset(schedule_store, workspace=resolved_workspace)
    )

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

    include_reasoning = llm_profile.compat.include_reasoning_content
    context_builder = ContextBuilder(
        base_system_prompt=SYSTEM_PROMPT,
        memory_store=memory_store,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        include_reasoning_content=include_reasoning,
        workspace=resolved_workspace,
    )
    budget = TokenBudget(
        compact_token_threshold=config.compact_token_threshold,
        reserved_completion_tokens=config.reserved_completion_tokens,
        include_reasoning_content=include_reasoning,
    )
    compactor = Compactor(
        session_manager=manager,
        context_builder=context_builder,
        budget=budget,
        tool_registry=tool_registry,
        summarizer=make_summarizer(llm),
        keep_recent_tokens=config.compact_keep_recent_tokens,
        include_reasoning_content=include_reasoning,
    )
    approval_policy = ApprovalPolicy(
        handler=approval_handler,
        mode=config.approval_mode,
    )
    agent_loop = AgentLoop(
        llm=llm,
        tool_registry=tool_registry,
        session_manager=manager,
        context_builder=context_builder,
        budget=budget,
        compactor=compactor,
        materializer=ToolOutputMaterializer(artifact_store),
        model=config.model,
        approval_gate=ToolApprovalGate(approval_policy),
        max_iterations=config.max_iterations,
        max_parallel_tools=config.max_parallel_tools,
        llm_max_retries=config.llm_max_retries,
    )
    agent_session = AgentSession(
        agent_loop=agent_loop,
        session_manager=manager,
        # runs.jsonl is a fold over the same event stream the UIs subscribe to.
        base_event_handler=fanout(
            RunLogFold(run_log_store, tool_registry=tool_registry),
            run_event_handler,
        ),
    )
    return MiniBotRuntime(
        config=config,
        manager=manager,
        memory_store=memory_store,
        artifact_store=artifact_store,
        run_log_store=run_log_store,
        schedule_store=schedule_store,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        mcp_host=mcp_host,
        context_builder=context_builder,
        budget=budget,
        compactor=compactor,
        agent_loop=agent_loop,
        agent_session=agent_session,
        approval_policy=approval_policy,
        approval_broker=approval_broker,
    )


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

    user_config_path = (resolve_state_home() / "mcp.json").resolve()
    if user_config_path.exists():
        return user_config_path.parent, user_config_path, "user"

    package_config_path = (package_dir / "mcp.json").resolve()
    if package_config_path.exists():
        return package_dir, package_config_path, "bundled"

    return package_dir, None, "none"
