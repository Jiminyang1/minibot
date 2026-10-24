"""Shell execution tool."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from typing import Any

from .base import Tool, ToolExecutionContext
from .result import ToolOutput

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\b|\brm\s+.*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\b"), "rm -rf (递归强制删除)"),
    (re.compile(r"\brm\s+-rf?\b|\brm\s+-fr?\b"),                  "rm -rf (递归强制删除)"),
    (re.compile(r"\bdd\b"),                                        "dd (磁盘级写入/擦除)"),
    (re.compile(r":\s*\(\s*\)\s*\{.*:\s*\|.*&.*\}"),              "Fork 炸弹"),
    (re.compile(r"\bmkfs\b"),                                      "mkfs (格式化磁盘)"),
    (re.compile(r"\bfdisk\b|\bparted\b"),                          "磁盘分区工具"),
    (re.compile(r">\s*/dev/(s|h|v|xv)d[a-z]"),                    "直接写入磁盘设备"),
    (re.compile(r"\bchmod\s+-R\s+777\b|\bchmod\s+777\b"),         "chmod 777 (开放全局写权限)"),
    (re.compile(r"\bchown\s+-R\b"),                                "chown -R (递归更改所有者)"),
    (re.compile(r"\bshred\b|\bwipe\b"),                            "shred/wipe (安全擦除文件)"),
    (re.compile(r"\bpoweroff\b|\breboot\b|\bshutdown\b|\bhalt\b"), "关机/重启命令"),
    (re.compile(r"\bkillall\b|\bpkill\s+-9\b"),                    "批量强制杀进程"),
    (re.compile(r"curl\s+.*\|\s*(ba)?sh|wget\s+.*\|\s*(ba)?sh"),  "管道执行远程脚本"),
    (re.compile(r"\biptables\s+-F\b|\bnft\s+flush\b"),             "清空防火墙规则"),
    (re.compile(r"\bsudo\s+su\b|\bsudo\s+-i\b|\bsudo\s+-s\b"),    "获取 root shell"),
]


# pipefail: `pytest ... | tail` must report pytest's failure, not tail's 0.
_BASH = shutil.which("bash")
# Under pipefail, `producer | head` exits 128+SIGPIPE once head stops reading.
_SIGPIPE_EXIT = 128 + signal.SIGPIPE
# How often the wait loop wakes up to check cancel and the deadline.
_POLL_INTERVAL = 0.1
# Time the process group gets between SIGTERM and SIGKILL.
_KILL_GRACE = 0.5


def _check_dangerous(command: str) -> str | None:
    """Return a human-readable warning if *command* matches a dangerous pattern."""
    for pattern, label in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return label
    return None


def _stop_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill the command and everything it spawned (SIGTERM, then SIGKILL)."""
    if not hasattr(os, "killpg"):
        proc.kill()
        return
    # start_new_session made bash a group leader, so its pid is the pgid.
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + _KILL_GRACE
    while time.monotonic() < deadline:
        # Reap bash so its zombie does not keep the group alive.
        proc.poll()
        try:
            os.killpg(pgid, 0)
        except OSError:
            return
        time.sleep(0.02)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _preview_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class ExecTool(Tool):
    """Execute shell commands with a minimal safety check."""

    _PREVIEW_CHARS = 2000

    @property
    def layer(self) -> str:
        return "kernel"

    @property
    def name(self) -> str:
        return "exec"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "执行 shell 命令"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令，例如 ls -l 或 pwd, 注意不要执行危险命令",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    _TIMEOUT = 30

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        command: str,
        **kwargs: Any,
    ) -> ToolOutput:
        danger = _check_dangerous(command)
        if danger:
            return ToolOutput.failure(
                "permission_denied",
                "命令被安全策略拒绝执行。",
                data={"command": command, "reason": danger},
            )
        # Own session/process group: on timeout or cancel we kill the whole
        # group, not just bash, so `sleep 100; echo x` leaves nothing behind.
        proc = subprocess.Popen(
            [_BASH, "-o", "pipefail", "-c", command] if _BASH else command,
            shell=_BASH is None, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self._workspace,
            start_new_session=True,
        )
        cancel_event = context.cancel_event
        deadline = time.monotonic() + self._TIMEOUT
        stopped: str | None = None
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=_POLL_INTERVAL)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel_event is not None and cancel_event.is_set():
                stopped = "interrupted"
                break
            if time.monotonic() >= deadline:
                stopped = "timeout"
                break

        if stopped is not None:
            _stop_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                # Something outside the group still holds the pipes.
                proc.kill()
                proc.wait()
                stdout, stderr = "", ""
            data: dict[str, Any] = {"command": command}
            if stopped == "timeout":
                data["timeout_seconds"] = self._TIMEOUT
            if stdout:
                data["stdout_preview"] = _preview_text(stdout, self._PREVIEW_CHARS)[0]
            if stderr:
                data["stderr_preview"] = _preview_text(stderr, self._PREVIEW_CHARS)[0]
            if stopped == "timeout":
                return ToolOutput.failure(
                    "timeout",
                    f"命令执行超过 {self._TIMEOUT} 秒，已终止。",
                    data=data,
                )
            return ToolOutput.failure(
                "interrupted",
                "命令已被用户中断。",
                data=data,
            )

        returncode = proc.returncode
        stdout = stdout or ""
        stderr = stderr or ""
        stdout_preview, stdout_truncated = _preview_text(stdout, self._PREVIEW_CHARS)
        stderr_preview, stderr_truncated = _preview_text(stderr, self._PREVIEW_CHARS)
        truncated = stdout_truncated or stderr_truncated

        data: dict[str, Any] = {
            "command": command,
            "exit_code": returncode,
        }
        if stdout_truncated:
            data["stdout_preview"] = stdout_preview
        else:
            data["stdout"] = stdout
        if stderr_truncated:
            data["stderr_preview"] = stderr_preview
        else:
            data["stderr"] = stderr

        full_output = None
        if truncated:
            full_output = (
                f"$ {command}\n"
                f"[exit_code] {returncode}\n\n"
                "[stdout]\n"
                f"{stdout}\n\n"
                "[stderr]\n"
                f"{stderr}"
            )

        if returncode == _SIGPIPE_EXIT:
            return ToolOutput.success(
                f"命令已执行，退出码 {returncode}"
                "（管道下游提前停止读取，上游收到 SIGPIPE，通常无害）。",
                data=data,
                truncated=truncated,
                content=full_output,
                content_kind="text",
                content_name="exec_output",
            )

        if returncode != 0:
            return ToolOutput.failure(
                "error",
                f"命令执行失败，退出码 {returncode}。",
                data=data,
                truncated=truncated,
                content=full_output,
                content_kind="text",
                content_name="exec_output",
            )

        return ToolOutput.success(
            f"命令已执行，退出码 {returncode}。",
            data=data,
            truncated=truncated,
            content=full_output,
            content_kind="text",
            content_name="exec_output",
        )
