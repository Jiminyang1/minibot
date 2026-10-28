"""Lesson storage: ``<state_home>/lessons.json``.

A lesson is a human-reviewed rule — "when <trigger>, do <guidance>" — drafted
from mined failures. Only ``approved`` lessons ever reach a model request.

The file is rewritten atomically under a thread lock plus an fcntl lock, so
the review CLI and a running agent can share it. Ids come from a persistent
counter and are never reused: a rejected ``les_3`` must not come back as a
different rule (traces and evals still name it).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
import fcntl
import json
from pathlib import Path
import threading
from typing import Any, Literal, TypeAlias, TypeVar
import uuid

LessonStatus: TypeAlias = Literal["draft", "approved", "rejected"]
LESSON_STATUSES: tuple[LessonStatus, ...] = ("draft", "approved", "rejected")

_FORMAT_VERSION = 1
_T = TypeVar("_T")


@dataclass(frozen=True)
class Evidence:
    """One mined incident a lesson was drawn from (kept short)."""

    incident_id: str
    kind: str
    session_id: str
    excerpt: str
    message_ids: tuple[str, ...] = ()
    run_id: str | None = None
    # The user's own words for that turn; retrieval matches new messages
    # against these (request vs request), not only against the trigger.
    request: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "kind": self.kind,
            "session_id": self.session_id,
            "message_ids": list(self.message_ids),
            "run_id": self.run_id,
            "excerpt": self.excerpt,
            "request": self.request,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            incident_id=str(data.get("incident_id", "")),
            kind=str(data.get("kind", "")),
            session_id=str(data.get("session_id", "")),
            excerpt=str(data.get("excerpt", "")),
            message_ids=tuple(str(m) for m in data.get("message_ids") or ()),
            run_id=str(data["run_id"]) if data.get("run_id") else None,
            request=str(data.get("request", "")),
        )


@dataclass(frozen=True)
class Lesson:
    id: str
    status: LessonStatus
    trigger: str
    guidance: str
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    created_at: str = ""
    reviewed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "trigger": self.trigger,
            "guidance": self.guidance,
            "evidence": [item.to_dict() for item in self.evidence],
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        status = str(data.get("status", "draft"))
        if status not in LESSON_STATUSES:
            status = "draft"
        return cls(
            id=str(data["id"]),
            status=status,  # type: ignore[arg-type]
            trigger=str(data.get("trigger", "")),
            guidance=str(data.get("guidance", "")),
            evidence=tuple(
                Evidence.from_dict(item)
                for item in data.get("evidence") or ()
                if isinstance(item, dict)
            ),
            created_at=str(data.get("created_at", "")),
            reviewed_at=str(data["reviewed_at"]) if data.get("reviewed_at") else None,
        )


@dataclass
class _State:
    lessons: list[Lesson]
    next_index: int
    mined: set[str]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class LessonStore:
    """Load / mutate lessons and the set of already-mined incident ids."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from ..config import resolve_state_home

            path = resolve_state_home() / "lessons.json"
        self.path = path.expanduser().resolve()
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._thread_lock = threading.Lock()

    # ── reads ─────────────────────────────────────────────────────

    def list(self, status: LessonStatus | None = None) -> list[Lesson]:
        lessons = self._read().lessons
        if status is None:
            return lessons
        return [lesson for lesson in lessons if lesson.status == status]

    def get(self, lesson_id: str) -> Lesson | None:
        for lesson in self._read().lessons:
            if lesson.id == lesson_id:
                return lesson
        return None

    def mined_incident_ids(self) -> set[str]:
        return set(self._read().mined)

    def signature(self) -> tuple[int, int] | None:
        """``(mtime_ns, size)`` of the file; a cheap change check for caches."""
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    # ── writes ────────────────────────────────────────────────────

    def add_draft(
        self,
        trigger: str,
        guidance: str,
        evidence: Iterable[Evidence] = (),
    ) -> Lesson:
        trigger, guidance = clean_rule(trigger, guidance)

        def mutate(state: _State) -> Lesson:
            lesson = Lesson(
                id=f"les_{state.next_index}",
                status="draft",
                trigger=trigger,
                guidance=guidance,
                evidence=_dedupe_evidence((), evidence),
                created_at=now_iso(),
            )
            state.next_index += 1
            state.lessons.append(lesson)
            return lesson

        return self._mutate(mutate)

    def attach_evidence(
        self, lesson_id: str, evidence: Iterable[Evidence]
    ) -> Lesson | None:
        evidence = list(evidence)

        def mutate(state: _State) -> Lesson | None:
            return _replace_lesson(
                state,
                lesson_id,
                lambda lesson: replace(
                    lesson, evidence=_dedupe_evidence(lesson.evidence, evidence)
                ),
            )

        return self._mutate(mutate)

    def update(
        self,
        lesson_id: str,
        change: Callable[[Lesson], Lesson],
    ) -> Lesson | None:
        """Apply *change* to one lesson under the lock; ``None`` if missing.

        *change* may raise to abort; nothing is written then.
        """
        return self._mutate(lambda state: _replace_lesson(state, lesson_id, change))

    def mark_mined(self, incident_ids: Iterable[str]) -> None:
        ids = set(incident_ids)
        if not ids:
            return

        def mutate(state: _State) -> None:
            state.mined |= ids

        self._mutate(mutate)

    # ── file plumbing ─────────────────────────────────────────────

    def _mutate(self, fn: Callable[[_State], _T]) -> _T:
        with _FileLock(self._thread_lock, self._lock_path):
            # Strict: rewriting from an unreadable file would wipe every lesson.
            state = self._read(strict=True)
            result = fn(state)
            self._write(state)
            return result

    def _read(self, *, strict: bool = False) -> _State:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise ValueError(f"无法读取经验库 {self.path}: {exc}") from exc
            data = {}
        if not isinstance(data, dict):
            if strict:
                raise ValueError(f"经验库格式无效: {self.path}")
            data = {}
        lessons = [
            Lesson.from_dict(raw)
            for raw in data.get("lessons") or ()
            if isinstance(raw, dict) and "id" in raw
        ]
        raw_next = data.get("next_index")
        next_index = max(
            raw_next if isinstance(raw_next, int) and raw_next > 0 else 1,
            _highest_index(lessons) + 1,
        )
        mined = {str(item) for item in data.get("mined_incidents") or ()}
        return _State(lessons=lessons, next_index=next_index, mined=mined)

    def _write(self, state: _State) -> None:
        payload = {
            "version": _FORMAT_VERSION,
            "next_index": state.next_index,
            "lessons": [lesson.to_dict() for lesson in state.lessons],
            "mined_incidents": sorted(state.mined),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)


class _FileLock:
    """Thread lock + fcntl file lock for cross-process read-modify-write."""

    def __init__(self, thread_lock: threading.Lock, lock_path: Path) -> None:
        self._thread_lock = thread_lock
        self._lock_path = lock_path
        self._handle = None

    def __enter__(self) -> None:
        self._thread_lock.acquire()
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._lock_path.open("a", encoding="utf-8")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
            raise

    def __exit__(self, *exc: object) -> None:
        assert self._handle is not None
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
        finally:
            self._handle = None
            self._thread_lock.release()


def _replace_lesson(
    state: _State,
    lesson_id: str,
    change: Callable[[Lesson], Lesson],
) -> Lesson | None:
    for index, lesson in enumerate(state.lessons):
        if lesson.id == lesson_id:
            updated = change(lesson)
            state.lessons[index] = updated
            return updated
    return None


def _dedupe_evidence(
    existing: Iterable[Evidence], new: Iterable[Evidence]
) -> tuple[Evidence, ...]:
    seen: set[str] = set()
    merged: list[Evidence] = []
    for item in (*existing, *new):
        if item.incident_id in seen:
            continue
        seen.add(item.incident_id)
        merged.append(item)
    return tuple(merged)


def clean_rule(trigger: str, guidance: str) -> tuple[str, str]:
    trigger = " ".join(trigger.split())
    guidance = " ".join(guidance.split())
    if not trigger or not guidance:
        raise ValueError("trigger 和 guidance 都不能为空。")
    return trigger, guidance


def _highest_index(lessons: Iterable[Lesson]) -> int:
    highest = 0
    for lesson in lessons:
        if lesson.id.startswith("les_") and lesson.id[4:].isdigit():
            highest = max(highest, int(lesson.id[4:]))
    return highest
