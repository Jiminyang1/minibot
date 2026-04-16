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
    from .session import SessionManager

    manager = SessionManager(Path.cwd())

    try:
        llm = OpenAIClient(model=config.model)
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    emit = lambda msg: print(f"  🔧 {msg}")
    spec = AgentSpec(default_model=config.model)
    runner = AgentRunner(
        llm,
        spec,
        event_handler=emit,
    )
    turn_engine = TurnEngine(
        spec,
        runner,
        manager,
        config,
        summarizer=make_summarizer(llm),
        event_handler=emit,
    )
    run_repl(turn_engine, manager)


if __name__ == "__main__":
    main()
