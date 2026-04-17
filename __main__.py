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
    from .runtime import AgentRunner, ContextManager, TurnEngine, make_summarizer
    from .skills import SkillRegistry
    from .user_memory import UserMemoryStore
    from .session import SessionManager
    from .tools import (
        ToolRegistry,
        filesystem_toolset,
        macos_toolset,
        memory_toolset,
        network_toolset,
        shell_toolset,
    )
    from .ui import prompt_approval, tool_log

    workspace = Path.cwd()
    manager = SessionManager(workspace)
    memory_store = UserMemoryStore()

    try:
        llm = OpenAIClient(model=config.model)
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    tool_registry = ToolRegistry()
    tool_registry.register_all(filesystem_toolset(workspace, manager))
    tool_registry.register_all(shell_toolset(workspace, manager))
    tool_registry.register_all(network_toolset(manager))
    tool_registry.register_all(macos_toolset())
    tool_registry.register_all(memory_toolset(memory_store))
    skill_registry = SkillRegistry.from_directory(
        Path(__file__).resolve().parent / "skills"
    )

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
        event_handler=tool_log,
        approval_handler=None if config.auto_approve else prompt_approval,
    )
    turn_engine = TurnEngine(
        runner,
        manager,
        config,
        context_manager=context_manager,
        event_handler=tool_log,
    )
    run_repl(turn_engine, manager, memory_store)


if __name__ == "__main__":
    main()
