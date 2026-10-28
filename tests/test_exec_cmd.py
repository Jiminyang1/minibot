from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.tools.base import ToolExecutionContext
from minibot.tools.exec_cmd import ExecTool


def _pgrep(marker: str) -> list[str]:
    result = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return result.stdout.split()


def _wait_gone(marker: str, timeout: float = 2.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = _pgrep(marker)
        if not pids:
            return []
        time.sleep(0.05)
    return _pgrep(marker)


class ExecToolProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmpdir.name)
        self._markers: list[str] = []

    def tearDown(self) -> None:
        # A failed assertion must not leave a stray sleep behind.
        for marker in self._markers:
            subprocess.run(["pkill", "-f", marker], capture_output=True)
        self._tmpdir.cleanup()

    def _marker(self, seconds: str) -> str:
        marker = f"sleep {seconds}"
        self._markers.append(marker)
        return marker

    def test_timeout_kills_grandchildren(self) -> None:
        # Before: only bash was killed and `sleep` lived on for its full time.
        marker = self._marker("37.123")
        tool = ExecTool(workspace=self.workspace)
        tool._TIMEOUT = 1
        context = ToolExecutionContext(session_id="s_test")

        started = time.monotonic()
        result = tool.execute(context=context, command=f"{marker}; echo x")
        elapsed = time.monotonic() - started

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "timeout")
        self.assertEqual(result.data["timeout_seconds"], 1)
        self.assertLess(elapsed, 3)
        self.assertEqual(_wait_gone(marker), [])

    def test_timeout_kills_background_children(self) -> None:
        marker = self._marker("38.123")
        tool = ExecTool(workspace=self.workspace)
        tool._TIMEOUT = 1
        context = ToolExecutionContext(session_id="s_test")

        result = tool.execute(context=context, command=f"echo start; {marker} & wait")

        self.assertEqual(result.code, "timeout")
        self.assertEqual(result.data.get("stdout_preview"), "start\n")
        self.assertEqual(_wait_gone(marker), [])

    def test_cancel_event_interrupts_running_command(self) -> None:
        marker = self._marker("39.123")
        tool = ExecTool(workspace=self.workspace)
        cancel = threading.Event()
        context = ToolExecutionContext(session_id="s_test", cancel_event=cancel)
        timer = threading.Timer(0.3, cancel.set)

        started = time.monotonic()
        timer.start()
        try:
            result = tool.execute(context=context, command=f"{marker}; echo x")
        finally:
            timer.cancel()
        elapsed = time.monotonic() - started

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "interrupted")
        self.assertLess(elapsed, 2)
        self.assertEqual(_wait_gone(marker), [])

    def test_normal_command_still_succeeds(self) -> None:
        tool = ExecTool(workspace=self.workspace)
        context = ToolExecutionContext(session_id="s_test", cancel_event=threading.Event())

        result = tool.execute(context=context, command="echo hello; pwd")

        self.assertTrue(result.ok)
        self.assertEqual(result.code, "success")
        self.assertEqual(result.data["exit_code"], 0)
        lines = result.data["stdout"].splitlines()
        self.assertEqual(lines[0], "hello")
        self.assertEqual(Path(lines[1]).resolve(), self.workspace.resolve())

    def test_pipeline_failure_reports_exit_code(self) -> None:
        tool = ExecTool(workspace=self.workspace)
        context = ToolExecutionContext(session_id="s_test")

        result = tool.execute(context=context, command="exit 4 | cat")

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "error")
        self.assertEqual(result.data["exit_code"], 4)


if __name__ == "__main__":
    unittest.main()
