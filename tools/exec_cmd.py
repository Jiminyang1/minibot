"""Shell execution tool."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from .base import Tool

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


class ExecTool(Tool):
    """Execute shell commands with a minimal safety check."""

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
    _MAX_OUTPUT = 10_000

    def execute(self, *, command: str, **kwargs: Any) -> str:
        danger = _check_dangerous(command)
        if danger:
            return (
                f"[安全拦截] 命令被拒绝执行。\n"
                f"匹配到危险操作: {danger}\n"
                f"原始命令: {command}"
            )
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=self._TIMEOUT,
                cwd=self._workspace,
            )
        except subprocess.TimeoutExpired:
            return f"[超时] 命令执行超过 {self._TIMEOUT} 秒，已终止。"
        output = result.stdout or result.stderr or "(命令没有输出)"
        if len(output) > self._MAX_OUTPUT:
            output = output[:self._MAX_OUTPUT] + f"\n...(输出已截断，共 {len(output)} 字符)"
        return output
