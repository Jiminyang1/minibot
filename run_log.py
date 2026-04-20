"""Append-only run summary logging for MiniBot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Literal
import uuid


RunStatus = Literal["success", "failed"]


def utc_now() -> str:
    """Return an ISO8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def make_run_id(now: datetime | None = None) -> str:
    """Generate a short run id for one handled turn."""
    current = now or datetime.now(UTC)
    stamp = current.strftime("%Y%m%d_%H%M%S")
    return f"r_{stamp}_{uuid.uuid4().hex[:4]}"


def preview_text(text: str, limit: int) -> str:
    """Collapse whitespace and truncate to *limit* chars."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return compact[:limit]
    return compact[: limit - 3] + "..."


@dataclass(frozen=True)
class RunLogRecord:
    """One persisted summary log for a handled user turn."""

    run_id: str
    session_id: str
    turn_index: int
    timestamp: str
    ended_at: str
    status: RunStatus
    model: str
    user_input_preview: str
    duration_ms: int
    did_compact: bool
    compact_message: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    llm_call_count: int
    tool_call_count: int
    tools_used: list[str]
    mcp_tool_call_count: int
    mcp_servers_used: list[str]
    mcp_transports_used: list[str]
    mcp_error_count: int
    final_reply_preview: str | None
    error_type: str | None
    error_message_preview: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RunLogStore:
    """Persist run summaries to ``.minibot/runs.jsonl``."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.state_dir = (workspace or Path.cwd()).resolve() / ".minibot"
        self.runs_path = self.state_dir / "runs.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: RunLogRecord) -> None:
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
