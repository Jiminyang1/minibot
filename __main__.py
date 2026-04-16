"""Run MiniBot: python -m minibot"""

from __future__ import annotations

from pathlib import Path

from .config import Config, load_env


def main() -> None:
    load_env()
    config = Config.from_env()

    from .agent import Agent
    from .cli import run_repl
    from .llm import OpenAIClient
    from .loop import AgentLoop
    from .session import SessionManager

    manager = SessionManager(Path.cwd())

    try:
        llm = OpenAIClient(model=config.model)
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    agent = Agent(
        llm,
        event_handler=lambda msg: print(f"  🔧 {msg}"),
    )
    loop = AgentLoop(agent, llm, manager, config)
    run_repl(loop, manager)


main()
