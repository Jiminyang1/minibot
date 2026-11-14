"""Lessons loop (experimental): mine failures, draft fixes, review them, retrieve them.

Not yet run on real failures; thresholds come from a small hand-written set
(``benchmark_zh.json``, re-measured by ``python -m minibot.lessons.benchmark``).
"""

from __future__ import annotations

import importlib.util


def fastembed_available() -> bool:
    """True when the ``lessons`` extra is installed.

    Lives here, not in ``embed``: ``embed`` imports numpy, which arrives only
    with the extra, so the question must be answerable without it.
    """
    return importlib.util.find_spec("fastembed") is not None
