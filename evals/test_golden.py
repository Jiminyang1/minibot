"""Golden evals as pytest: one test per (case, stage), against the real model.

Opt in with ``-m golden`` (nothing here is collected otherwise, so plain
``uv run pytest`` and the unittest suite never call the model)::

    uv run pytest evals -m golden                          # every case, every stage
    uv run pytest evals -m golden -k tool_args             # one stage across cases
    uv run pytest evals -m golden --golden-case file-read-answer
    uv run pytest evals -m golden --golden-source local    # cases.json, not the dataset
    python -m minibot.evals pytest -k tool_args            # same, -m golden added

Test ids are ``test_stage[<case>-<stage>]``. Cases come from the Langfuse
dataset ``minibot-golden`` when Langfuse is configured (``--golden-source
auto``, the default) — that replays exactly what the dataset holds, so run
``python -m minibot.evals sync`` after editing ``cases.json`` — and from
``cases.json`` otherwise. Each case's sandboxed turn runs once per session
and every stage test asserts on that cached output; a stage the case does
not specify is skipped, not passed. A failure names the stage, the scorer's
comment and, with Langfuse, the trace URL of that run.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ..config import Config, load_env
from .golden import GoldenSession
from .runner import (
    STAGES,
    build_judge,
    dataset_cases,
    langfuse_configured,
    load_cases,
    with_lessons,
)

pytestmark = pytest.mark.golden


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "golden_case" not in metafunc.fixturenames:
        return
    cases = _cases(metafunc.config)
    metafunc.parametrize(
        ("golden_case", "stage"),
        [pytest.param(case, stage, id=f"{case['id']}-{stage}") for case in cases for stage in STAGES],
    )


@pytest.fixture(scope="session")
def golden_session(request: pytest.FixtureRequest) -> Iterator[GoldenSession]:
    load_env()
    config = with_lessons(Config.from_env(), request.config.getoption("--golden-lessons"))
    langfuse = None
    if langfuse_configured():
        from langfuse import get_client

        langfuse = get_client()
    session = GoldenSession(config, judge=build_judge(config), langfuse=langfuse)
    yield session
    session.close()


def test_stage(golden_case: dict[str, Any], stage: str, golden_session: GoldenSession) -> None:
    result = golden_session.check(golden_case, stage)
    if result is None:
        pytest.skip(f"用例未指定 {stage}")
    assert result.passed, result.message


def _cases(config: pytest.Config) -> list[dict[str, Any]]:
    cached = getattr(config, "_golden_cases", None)
    if cached is not None:
        return cached
    load_env()
    source = config.getoption("golden_source", default="auto")
    if source == "auto":
        source = "langfuse" if langfuse_configured() else "local"
    case_ids = config.getoption("golden_case", default=[]) or None
    try:
        if source == "langfuse":
            if not langfuse_configured():
                raise pytest.UsageError("--golden-source langfuse 需要 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY。")
            from langfuse import get_client

            cases = dataset_cases(get_client(), case_ids)
        else:
            cases = load_cases(case_ids)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    config._golden_cases = cases  # type: ignore[attr-defined]
    return cases
