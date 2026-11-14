"""The lessons extra is optional: a plain ``uv sync`` has no numpy or fastembed.

These tests import nothing from the extra, so they also run without it. Each
one runs in a child process where both packages are made unimportable.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# ``sys.modules[name] = None`` makes ``import name`` fail and
# ``importlib.util.find_spec(name)`` return None, like a missing package.
_WITHOUT_EXTRA = "import sys; sys.modules['numpy'] = None; sys.modules['fastembed'] = None; "


def _run(code: str, home: str) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        MINIBOT_HOME=home,
    )
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_EXTRA + code],
        capture_output=True,
        text=True,
        env=env,
    )


class WithoutLessonsExtraTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_mode_starts_without_lessons(self) -> None:
        # MINIBOT_LESSONS defaults to auto, so this is every plain install.
        result = _run(
            "from pathlib import Path; "
            "from minibot.bootstrap import _build_lesson_retriever; "
            "from minibot.config import Config; "
            f"print(_build_lesson_retriever(Config(), Path({self.home!r})))",
            self.home,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "None")

    def test_on_mode_names_the_missing_extra(self) -> None:
        result = _run(
            "from pathlib import Path; "
            "from minibot.bootstrap import _build_lesson_retriever; "
            "from minibot.config import Config; "
            f"_build_lesson_retriever(Config(lessons_mode='on'), Path({self.home!r}))",
            self.home,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RuntimeError", result.stderr)
        self.assertIn("uv sync --extra lessons", result.stderr)

    def test_cli_lists_and_mine_names_the_missing_extra(self) -> None:
        listed = _run(
            "from minibot.lessons.__main__ import main; raise SystemExit(main(['list']))",
            self.home,
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)

        mined = _run(
            "from minibot.lessons.__main__ import main; "
            "raise SystemExit(main(['mine', '--dry-run']))",
            self.home,
        )
        self.assertEqual(mined.returncode, 1, mined.stderr)
        self.assertIn("uv sync --extra lessons", mined.stderr)
        self.assertNotIn("Traceback", mined.stderr)


if __name__ == "__main__":
    unittest.main()
