"""pytest wiring for the golden suite; see ``test_golden.py`` for usage.

The suite calls the real model, so it is opt-in: unless the ``-m``
expression selects ``golden``, ``test_golden.py`` is not even collected
(no case loading, no network). ``python -m minibot.evals pytest`` adds
``-m golden`` for you.
"""

from __future__ import annotations

from typing import Any

import pytest

_GOLDEN_FILE = "test_golden.py"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("golden", "MiniBot golden evals")
    group.addoption(
        "--golden-source",
        choices=("auto", "local", "langfuse"),
        default="auto",
        help="用例来源：Langfuse dataset 或本地 cases.json（auto = 配置了 Langfuse 就用 dataset）",
    )
    group.addoption(
        "--golden-lessons",
        choices=("off", "on"),
        default="off",
        help="是否注入已批准的 lesson（on = 用 ~/.minibot/lessons.json）",
    )
    group.addoption(
        "--golden-case",
        action="append",
        default=[],
        metavar="ID",
        help="只跑指定用例，可重复",
    )


def pytest_ignore_collect(collection_path: Any, config: pytest.Config) -> bool | None:
    if collection_path.name == _GOLDEN_FILE and not golden_requested(config):
        return True
    return None


def golden_requested(config: pytest.Config) -> bool:
    """True when the ``-m`` expression selects tests marked ``golden``."""
    markexpr = config.getoption("markexpr", default="") or ""
    if not markexpr.strip():
        return False
    try:
        from _pytest.mark.expression import Expression

        return Expression.compile(markexpr).evaluate(lambda name, **_: name == "golden")
    except Exception:  # Private API; fall back to a plain mention.
        return "golden" in markexpr and "not golden" not in markexpr
