"""Run MiniBot: python -m minibot"""

from __future__ import annotations

from pathlib import Path

from .config import Config, load_env


def main() -> None:
    load_env()
    try:
        config = Config.from_env()
    except ValueError as exc:
        print(f"配置错误: {exc}")
        return

    from .agent import AgentRunner, AgentSpec
    from .cli import run_repl
    from .compaction import make_summarizer
    from .llm import OpenAIClient
    from .loop import TurnEngine
    from .memory import MemoryStore
    from .session import SessionManager
    from .tools import ForgetTool, RememberTool, create_default_registry
    from .ui import prompt_approval, tool_log

    workspace = Path.cwd()
    manager = SessionManager(workspace)
    memory_store = MemoryStore(workspace)

    try:
        llm = OpenAIClient(model=config.model)
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    tool_registry = create_default_registry(workspace)
    tool_registry.register(RememberTool(memory_store))
    tool_registry.register(ForgetTool(memory_store))

    spec = AgentSpec(default_model=config.model, tool_registry=tool_registry)
    runner = AgentRunner(
        llm,
        spec,
        event_handler=tool_log,
        approval_handler=None if config.auto_approve else prompt_approval,
    )
    turn_engine = TurnEngine(
        spec,
        runner,
        manager,
        config,
        summarizer=make_summarizer(llm),
        memory_store=memory_store,
        event_handler=tool_log,
    )
    run_repl(turn_engine, manager, memory_store)


if __name__ == "__main__":
    main()
