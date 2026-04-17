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

    from .agent_runner import AgentRunner
    from .cli import run_repl
    from .context_manager import ContextManager, make_summarizer
    from .llm import OpenAIClient
    from .turn_engine import TurnEngine
    from .user_memory import UserMemoryStore
    from .session import SessionManager
    from .tools import (
        ToolRegistry,
        filesystem_toolset,
        memory_toolset,
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
    tool_registry.register_all(filesystem_toolset(workspace))
    tool_registry.register_all(shell_toolset(workspace))
    tool_registry.register_all(memory_toolset(memory_store))

    summarizer = make_summarizer(llm)
    context_manager = ContextManager(
        base_system_prompt=SYSTEM_PROMPT,
        memory_store=memory_store,
        tool_registry=tool_registry,
        max_history_turns=config.max_history_turns,
        compact_token_threshold=config.compact_token_threshold,
        reserved_completion_tokens=config.reserved_completion_tokens,
        compact_keep_recent=config.compact_keep_recent,
        summarizer=summarizer,
    )
    runner = AgentRunner(
        llm,
        event_handler=tool_log,
        approval_handler=None if config.auto_approve else prompt_approval,
    )
    turn_engine = TurnEngine(
        runner,
        manager,
        tool_registry,
        config,
        context_manager=context_manager,
        event_handler=tool_log,
    )
    run_repl(turn_engine, manager, memory_store)


if __name__ == "__main__":
    main()
