"""Shell execution tool."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from .base import Tool, ToolExecutionContext
from .result import ToolResult

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


def _check_dangerous(command: str) -> str | None:
    """Return a human-readable warning if *command* matches a dangerous pattern."""
    for pattern, label in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return label
    return None


def _preview_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


class ExecTool(Tool):
    """Execute shell commands with a minimal safety check."""

    _PREVIEW_CHARS = 2000

    @property
    def name(self) -> str:
        return "exec"

    @property
    def requires_approval(self) -> bool:
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
    ) -> ToolResult:
        danger = _check_dangerous(command)
        if danger:
            return ToolResult.failure(
                "permission_denied",
                "命令被安全策略拒绝执行。",
                data={"command": command, "reason": danger},
            )
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=self._TIMEOUT,
                cwd=self._workspace,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                "timeout",
                f"命令执行超过 {self._TIMEOUT} 秒，已终止。",
                data={"command": command, "timeout_seconds": self._TIMEOUT},
            )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        stdout_preview, stdout_truncated = _preview_text(stdout, self._PREVIEW_CHARS)
        stderr_preview, stderr_truncated = _preview_text(stderr, self._PREVIEW_CHARS)
        truncated = stdout_truncated or stderr_truncated

        data: dict[str, Any] = {
            "command": command,
            "exit_code": result.returncode,
        }
        if stdout_truncated:
            data["stdout_preview"] = stdout_preview
        else:
            data["stdout"] = stdout
        if stderr_truncated:
            data["stderr_preview"] = stderr_preview
        else:
            data["stderr"] = stderr

        artifact = None
        if truncated:
            full_output = (
                f"$ {command}\n"
                f"[exit_code] {result.returncode}\n\n"
                "[stdout]\n"
                f"{stdout}\n\n"
                "[stderr]\n"
                f"{stderr}"
            )
            artifact = self._require_session_manager().put_artifact_text(
                context.session_id,
                full_output,
                kind="text",
                name="exec_output",
            )

        if result.returncode != 0:
            return ToolResult.failure(
                "error",
                f"命令执行失败，退出码 {result.returncode}。",
                data=data,
                artifact=artifact,
                truncated=truncated,
            )

        return ToolResult.success(
            f"命令已执行，退出码 {result.returncode}。",
            data=data,
            artifact=artifact,
            truncated=truncated,
        )
