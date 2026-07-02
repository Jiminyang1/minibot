"""Shared helpers for driving the real AgentLoop in tests."""

from __future__ import annotations

from pathlib import Path
import threading

from minibot.artifacts import ArtifactStore
from minibot.llm import LLMClient
from minibot.runtime.agent_loop import AgentLoop, TurnOutcome
from minibot.runtime.approval import ApprovalPolicy, ToolApprovalGate
from minibot.runtime.budget import TokenBudget
from minibot.runtime.compactor import Compactor
from minibot.runtime.context_builder import ContextBuilder
from minibot.runtime.events import RuntimeEvent, RuntimeEventEmitter
from minibot.runtime.tool_output_materializer import ToolOutputMaterializer
from minibot.session import Session, SessionManager
from minibot.tools.registry import ToolRegistry


def build_loop(
    llm: LLMClient,
    registry: ToolRegistry,
    workspace: Path,
    *,
    model: str = "gpt-5.4-mini",
    max_iterations: int = 20,
    max_parallel_tools: int = 4,
    approval_policy: ApprovalPolicy | None = None,
    summarizer=None,
    compact_token_threshold: int = 400_000,
    reserved_completion_tokens: int = 4096,
    compact_keep_recent_tokens: int = 16_000,
    base_system_prompt: str = "BASE",
) -> tuple[AgentLoop, SessionManager]:
    manager = SessionManager(workspace)
    context_builder = ContextBuilder(
        base_system_prompt=base_system_prompt,
        memory_store=None,
        skill_registry=None,
        tool_registry=registry,
    )
    budget = TokenBudget(
        compact_token_threshold=compact_token_threshold,
        reserved_completion_tokens=reserved_completion_tokens,
    )
    compactor = Compactor(
        session_manager=manager,
        context_builder=context_builder,
        budget=budget,
        tool_registry=registry,
        summarizer=summarizer or (lambda request: "summary"),
        keep_recent_tokens=compact_keep_recent_tokens,
    )
    loop = AgentLoop(
        llm=llm,
        tool_registry=registry,
        session_manager=manager,
        context_builder=context_builder,
        budget=budget,
        compactor=compactor,
        materializer=ToolOutputMaterializer(ArtifactStore(workspace)),
        model=model,
        approval_gate=(
            None if approval_policy is None else ToolApprovalGate(approval_policy)
        ),
        max_iterations=max_iterations,
        max_parallel_tools=max_parallel_tools,
    )
    return loop, manager


def run_turn(
    loop: AgentLoop,
    manager: SessionManager,
    user_input: str,
    *,
    session_id: str = "s_test",
    session: Session | None = None,
    run_id: str = "r_test",
    events: list[RuntimeEvent] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[TurnOutcome, Session]:
    if session is None:
        session = manager.load(session_id) or manager.create_session(session_id)
    emitter = RuntimeEventEmitter(
        run_id=run_id,
        session_id=session.session_id,
        handler=None if events is None else events.append,
    )
    outcome = loop.run_turn(
        session,
        user_input,
        emitter=emitter,
        cancel_event=cancel_event,
    )
    return outcome, session
