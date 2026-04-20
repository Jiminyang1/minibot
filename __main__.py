"""Run MiniBot: python -m minibot"""

from __future__ import annotations

from pathlib import Path

from .config import Config, load_env
from .prompts import SYSTEM_PROMPT


def main() -> None:
    load_env()
    try:
        config = Config.from_env()
    except ValueError as exc:
        print(f"配置错误: {exc}")
        return

    from .cli import run_repl
    from .llm import OpenAIClient
    from .run_log import RunLogStore
    from .mcp_host import MCPHost
    from .runtime import (
        AgentRunner,
        ContextManager,
        ToolOutputMaterializer,
        TurnEngine,
        make_summarizer,
    )
    from .skills import SkillRegistry
    from .user_memory import UserMemoryStore
    from .artifacts import ArtifactStore
    from .session import SessionManager
    from .tools import (
        ToolRegistry,
        filesystem_toolset,
        memory_toolset,
        network_toolset,
        shell_toolset,
        skill_toolset,
    )
    from .ui import prompt_approval, tool_log

    package_dir = Path(__file__).resolve().parent
    workspace = Path.cwd()
    manager = SessionManager(workspace)
    artifact_store = ArtifactStore(workspace)
    memory_store = UserMemoryStore()
    run_log_store = RunLogStore(workspace)

    try:
        llm = OpenAIClient(model=config.model)
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    skill_registry = SkillRegistry.from_directory(package_dir / "skills")

    tool_registry = ToolRegistry()
    tool_registry.register_all(filesystem_toolset(workspace, artifact_store))
    tool_registry.register_all(shell_toolset(workspace))
    tool_registry.register_all(network_toolset())
    tool_registry.register_all(memory_toolset(memory_store))
    tool_registry.register_all(skill_toolset(skill_registry))
    mcp_config_root = workspace
    workspace_mcp_path = workspace / "mcp.json"
    package_mcp_path = package_dir / "mcp.json"
    if not workspace_mcp_path.exists() and package_mcp_path.exists():
        mcp_config_root = package_dir
        tool_log(
            f"当前目录未找到 mcp.json，改用包目录配置: {package_mcp_path}"
        )
    if (mcp_config_root / "mcp.json").exists():
        tool_log(f"MCP 配置路径: {mcp_config_root / 'mcp.json'}")

    mcp_host = MCPHost.from_workspace(
        mcp_config_root,
        event_handler=tool_log,
    )
    for tool in mcp_host.connect_all():
        if tool_registry.get(tool.name) is not None:
            tool_log(f"MCP 工具名称冲突，已跳过: {tool.name}")
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
    )
    runner = AgentRunner(
        llm,
        tool_registry,
        materializer=ToolOutputMaterializer(artifact_store),
        event_handler=tool_log,
        approval_handler=None if config.auto_approve else prompt_approval,
        max_parallel_tools=config.max_parallel_tools,
    )
    turn_engine = TurnEngine(
        runner,
        manager,
        config,
        context_manager=context_manager,
        event_handler=tool_log,
        run_log_store=run_log_store,
    )
    try:
        run_repl(turn_engine, manager, memory_store, mcp_host)
    finally:
        mcp_host.close()


if __name__ == "__main__":
    main()
