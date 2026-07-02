"""Structured runtime events for MiniBot runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import threading
from typing import Any, TypeAlias


RuntimeEventType: TypeAlias = str
RuntimeEventHandler: TypeAlias = Callable[["RuntimeEvent"], None]


def fanout(*handlers: RuntimeEventHandler | None) -> RuntimeEventHandler | None:
    """Compose event handlers so every subscriber sees the same stream."""
    active = [handler for handler in handlers if handler is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def handle(event: RuntimeEvent) -> None:
        for handler in active:
            handler(event)

    return handle


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RuntimeEvent:
    """One structured event emitted during a single agent run."""

    id: str
    run_id: str
    session_id: str
    seq: int
    type: RuntimeEventType
    created_at: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeEventEmitter:
    """Create monotonic per-run events and pass them to a handler."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        handler: RuntimeEventHandler | None = None,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self._handler = handler
        self._seq = 0
        self._lock = threading.Lock()

    def emit(
        self,
        event_type: RuntimeEventType,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        with self._lock:
            self._seq += 1
            seq = self._seq
        event = RuntimeEvent(
            id=f"{self.run_id}:{seq}",
            run_id=self.run_id,
            session_id=self.session_id,
            seq=seq,
            type=event_type,
            created_at=utc_now(),
            payload=payload or {},
        )
        if self._handler is not None:
            self._handler(event)
        return event
