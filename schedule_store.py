"""Scheduled-task storage and a minimal cron subset.

Tasks live in ``<state_home>/schedule.json`` (atomic rewrite, fcntl-guarded
so the CLI tools and the daemon can mutate concurrently). Cron expressions
are evaluated in local time — "每天早上 8 点" means the user's 8 o'clock.

The cron subset supports the five standard fields with ``*``, numbers,
comma lists, ranges, and ``/step``; day-of-month and day-of-week combine
with OR when both are restricted (vixie-cron semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import threading
from typing import Any, Literal
import uuid


ScheduleKind = Literal["cron", "once", "heartbeat"]

_FIELD_BOUNDS = (
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 7),  # day of week (0 and 7 are both Sunday)
)
_MAX_SCAN_MINUTES = 366 * 24 * 60


def parse_cron(expr: str) -> list[set[int]]:
    """Parse a five-field cron expression or raise ``ValueError``."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"cron 表达式必须是 5 个字段: {expr!r}")
    parsed: list[set[int]] = []
    for raw, (low, high) in zip(fields, _FIELD_BOUNDS):
        values: set[int] = set()
        for part in raw.split(","):
            values |= _parse_part(part, low, high, expr)
        parsed.append(values)
    # Normalise Sunday: 7 -> 0.
    if 7 in parsed[4]:
        parsed[4].discard(7)
        parsed[4].add(0)
    return parsed


def _parse_part(part: str, low: int, high: int, expr: str) -> set[int]:
    part = part.strip()
    step = 1
    if "/" in part:
        part, raw_step = part.split("/", 1)
        if not raw_step.isdigit() or int(raw_step) < 1:
            raise ValueError(f"cron step 无效: {expr!r}")
        step = int(raw_step)
    if part in {"*", ""}:
        start, end = low, high
    elif "-" in part:
        raw_start, raw_end = part.split("-", 1)
        if not raw_start.isdigit() or not raw_end.isdigit():
            raise ValueError(f"cron 字段无效: {expr!r}")
        start, end = int(raw_start), int(raw_end)
    elif part.isdigit():
        start = end = int(part)
    else:
        raise ValueError(f"cron 字段无效: {expr!r}")
    if start < low or end > high or start > end:
        raise ValueError(f"cron 字段超出范围 [{low},{high}]: {expr!r}")
    return set(range(start, end + 1, step))


def cron_matches(parsed: list[set[int]], moment: datetime) -> bool:
    minute, hour, dom, month, dow = parsed
    if moment.minute not in minute or moment.hour not in hour:
        return False
    if moment.month not in month:
        return False
    cron_weekday = (moment.weekday() + 1) % 7  # cron: Sunday=0
    dom_all = dom == set(range(1, 32))
    dow_all = dow == set(range(0, 7))
    dom_hit = moment.day in dom
    dow_hit = cron_weekday in dow
    if dom_all and dow_all:
        return True
    if dom_all:
        return dow_hit
    if dow_all:
        return dom_hit
    return dom_hit or dow_hit  # vixie OR when both restricted


def cron_next(expr: str, after: datetime) -> datetime:
    """First matching moment strictly after *after* (minute resolution)."""
    parsed = parse_cron(expr)
    moment = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(_MAX_SCAN_MINUTES):
        if cron_matches(parsed, moment):
            return moment
        moment += timedelta(minutes=1)
    raise ValueError(f"一年内找不到下一次触发时间: {expr!r}")


def local_now() -> datetime:
    return datetime.now().astimezone()


def parse_local_time(raw: str) -> datetime:
    """Parse an ISO timestamp; naive values are taken as local time."""
    moment = datetime.fromisoformat(raw)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment


@dataclass(frozen=True)
class ScheduledTask:
    """One scheduled prompt: a cron routine, one-shot reminder, or heartbeat.

    A heartbeat is a cron-timed patrol that reuses one persistent session
    (``session_id``) and reviews the HEARTBEAT.md checklist instead of
    executing a fixed instruction.
    """

    id: str
    title: str
    prompt: str
    kind: ScheduleKind
    expr: str  # cron expression, or ISO timestamp for kind="once"
    enabled: bool = True
    created_at: str = ""
    last_run_at: str | None = None
    last_status: str | None = None
    workspace: str | None = None
    session_id: str | None = None  # heartbeat only: the reused session

    def next_run(self, *, now: datetime | None = None) -> datetime | None:
        """Next due moment, or None when the task will not fire again."""
        if not self.enabled:
            return None
        if self.kind == "once":
            if self.last_run_at is not None:
                return None
            return parse_local_time(self.expr)
        anchor_raw = self.last_run_at or self.created_at
        anchor = parse_local_time(anchor_raw) if anchor_raw else (now or local_now())
        return cron_next(self.expr, anchor)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "kind": self.kind,
            "expr": self.expr,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "workspace": self.workspace,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        raw_kind = data.get("kind")
        kind: ScheduleKind = (
            raw_kind if raw_kind in ("cron", "once", "heartbeat") else "cron"
        )
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            prompt=str(data.get("prompt", "")),
            kind=kind,
            expr=str(data.get("expr", "")),
            enabled=bool(data.get("enabled", True)),
            created_at=str(data.get("created_at", "")),
            last_run_at=data.get("last_run_at"),
            last_status=data.get("last_status"),
            workspace=data.get("workspace"),
            session_id=data.get("session_id"),
        )


class ScheduleStore:
    """Atomic JSON persistence for scheduled tasks under the state home."""

    def __init__(self, state_home: Path | None = None) -> None:
        if state_home is None:
            from .config import resolve_state_home

            state_home = resolve_state_home()
        self.state_dir = state_home.resolve()
        self.path = self.state_dir / "schedule.json"
        self._lock_path = self.state_dir / "schedule.lock"
        self._thread_lock = threading.Lock()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        *,
        title: str,
        prompt: str,
        kind: ScheduleKind,
        expr: str,
        workspace: str | None = None,
    ) -> ScheduledTask:
        if kind == "once":
            parse_local_time(expr)
        else:
            parse_cron(expr)  # cron and heartbeat both run on cron time
        task = ScheduledTask(
            id="t_" + uuid.uuid4().hex[:10],
            title=title.strip() or prompt[:30],
            prompt=prompt,
            kind=kind,
            expr=expr,
            created_at=local_now().isoformat(timespec="seconds"),
            workspace=workspace,
        )
        with self._locked():
            tasks = self._read()
            tasks.append(task)
            self._write(tasks)
        return task

    def list(self) -> list[ScheduledTask]:
        with self._locked():
            return self._read()

    def get(self, task_id: str) -> ScheduledTask | None:
        for task in self.list():
            if task.id == task_id:
                return task
        return None

    def remove(self, task_id: str) -> bool:
        with self._locked():
            tasks = self._read()
            kept = [task for task in tasks if task.id != task_id]
            if len(kept) == len(tasks):
                return False
            self._write(kept)
            return True

    def update(self, task: ScheduledTask, **changes: Any) -> ScheduledTask:
        updated = replace(task, **changes)
        with self._locked():
            tasks = [updated if t.id == task.id else t for t in self._read()]
            self._write(tasks)
        return updated

    def _read(self) -> list[ScheduledTask]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [ScheduledTask.from_dict(item) for item in data if isinstance(item, dict)]

    def _write(self, tasks: list[ScheduledTask]) -> None:
        payload = json.dumps(
            [task.to_dict() for task in tasks], ensure_ascii=False, indent=2
        )
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(payload + "\n", encoding="utf-8")
        temp.replace(self.path)

    def _locked(self):
        return _FileLock(self._thread_lock, self._lock_path)


class _FileLock:
    """Thread lock + fcntl file lock for cross-process read-modify-write."""

    def __init__(self, thread_lock: threading.Lock, lock_path: Path) -> None:
        self._thread_lock = thread_lock
        self._lock_path = lock_path
        self._handle = None

    def __enter__(self) -> None:
        self._thread_lock.acquire()
        self._handle = self._lock_path.open("a", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *exc_info: object) -> None:
        assert self._handle is not None
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
        self._thread_lock.release()
