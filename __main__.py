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
    from .tools import create_default_registry

    workspace = Path.cwd()
    manager = SessionManager(workspace)

    try:
        llm = OpenAIClient(model=config.model)
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    emit = lambda msg: print(f"  🔧 {msg}")

    if config.auto_approve:
        approval = None
    else:
        def approval(tool_name: str, args: dict) -> bool:
            preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
            answer = input(f"  ⚠️  允许执行 {tool_name}({preview})? [y/N] ").strip().lower()
            return answer in {"y", "yes"}

    spec = AgentSpec(
        default_model=config.model,
        tool_registry=create_default_registry(workspace),
    )
    runner = AgentRunner(
        llm,
        spec,
        event_handler=emit,
        approval_handler=approval,
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
