"""File reading tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


def execute(args: dict[str, Any]) -> str:
    path = Path(str(args["path"]))
    return path.read_text(encoding="utf-8")
