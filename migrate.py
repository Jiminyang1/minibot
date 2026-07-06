"""One-shot migration of scattered per-workspace state into the global home.

Earlier MiniBot versions stored sessions under ``<workspace>/.minibot``.
This collects them into the global state home (``~/.minibot`` or
``MINIBOT_HOME``), renaming on session-id collision and merging run logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil


@dataclass
class MigrationReport:
    source: Path
    moved_sessions: list[str] = field(default_factory=list)
    renamed_sessions: list[tuple[str, str]] = field(default_factory=list)
    merged_run_records: int = 0
    skipped: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"{self.source}:"]
        if not self.moved_sessions and not self.merged_run_records and not self.skipped:
            out.append("  没有可迁移的内容")
            return out
        if self.moved_sessions:
            out.append(f"  迁移会话 {len(self.moved_sessions)} 个")
        for old, new in self.renamed_sessions:
            out.append(f"  id 冲突重命名: {old} -> {new}")
        if self.merged_run_records:
            out.append(f"  合并 run 记录 {self.merged_run_records} 条")
        for reason in self.skipped:
            out.append(f"  跳过: {reason}")
        return out


def migrate_workspace_state(workspace: Path, state_home: Path) -> MigrationReport:
    """Move ``<workspace>/.minibot`` sessions and run logs into *state_home*."""
    workspace = workspace.resolve()
    state_home = state_home.resolve()
    source = workspace / ".minibot"
    report = MigrationReport(source=source)

    if not source.exists():
        report.skipped.append("目录不存在")
        return report
    if source == state_home:
        report.skipped.append("已经是全局状态目录")
        return report

    target_sessions = state_home / "sessions"
    target_sessions.mkdir(parents=True, exist_ok=True)

    source_sessions = source / "sessions"
    if source_sessions.exists():
        for session_dir in sorted(source_sessions.iterdir()):
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            final_id = session_id
            suffix = 1
            while (target_sessions / final_id).exists():
                final_id = f"{session_id}_m{suffix}"
                suffix += 1
            shutil.move(str(session_dir), str(target_sessions / final_id))
            if final_id != session_id:
                _rewrite_meta_id(target_sessions / final_id, final_id)
                report.renamed_sessions.append((session_id, final_id))
            _ensure_meta_workspace(target_sessions / final_id, workspace)
            report.moved_sessions.append(final_id)

    source_runs = source / "runs.jsonl"
    if source_runs.exists():
        target_runs = state_home / "runs.jsonl"
        with target_runs.open("a", encoding="utf-8") as out:
            for line in source_runs.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    report.merged_run_records += 1
        source_runs.unlink()

    # current_session pointers and lock files are per-installation scratch;
    # the global home keeps its own.
    return report


def _rewrite_meta_id(session_dir: Path, new_id: str) -> None:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(meta, dict):
        meta["session_id"] = new_id
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _ensure_meta_workspace(session_dir: Path, workspace: Path) -> None:
    """Stamp provenance onto migrated sessions that predate the field."""
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(meta, dict) and not meta.get("workspace"):
        meta["workspace"] = str(workspace)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
