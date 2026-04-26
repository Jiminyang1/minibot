"""Run MiniBot: python -m minibot"""

from __future__ import annotations

from .bootstrap import build_runtime
from .config import Config, load_env


def main() -> None:
    load_env()
    try:
        config = Config.from_env()
    except ValueError as exc:
        print(f"配置错误: {exc}")
        return

    from .cli import run_repl
    from .runtime import ApprovalRequest
    from .ui import print_runtime_event, prompt_approval, tool_log

    def _approval_handler(request: ApprovalRequest) -> bool:
        return prompt_approval(request.tool_name, request.args)

    try:
        runtime = build_runtime(
            config=config,
            run_event_handler=print_runtime_event,
            log_handler=tool_log,
            approval_handler=None if config.auto_approve else _approval_handler,
        )
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return

    try:
        run_repl(
            runtime.turn_engine,
            runtime.manager,
            runtime.memory_store,
            runtime.mcp_host,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
