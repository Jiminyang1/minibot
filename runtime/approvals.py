"""Approval rendezvous services for MiniBot runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import threading

from .cancel import RunCancelled
from .hooks_builtin import ApprovalRequest


__all__ = ["ApprovalBroker"]


@dataclass
class _PendingApproval:
    event: threading.Event
    approved: bool | None = None


class ApprovalBroker:
    """In-memory rendezvous for web approval requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], _PendingApproval] = {}
        self._early_decisions: dict[tuple[str, str], bool] = {}
        self._cancelled: set[tuple[str, str]] = set()

    def wait(
        self,
        request: ApprovalRequest,
        cancel_event: threading.Event | None = None,
        *,
        poll_interval: float = 0.1,
    ) -> bool:
        key = (request.run_id, request.approval_id)
        if cancel_event is not None and cancel_event.is_set():
            self._mark_cancelled(key)
            raise RunCancelled("run cancelled while waiting for approval")

        with self._lock:
            if key in self._early_decisions:
                return self._early_decisions.pop(key)
            pending = self._pending.setdefault(
                key,
                _PendingApproval(event=threading.Event()),
            )

        while not pending.event.wait(poll_interval):
            if cancel_event is not None and cancel_event.is_set():
                self._mark_cancelled(key)
                raise RunCancelled("run cancelled while waiting for approval")

        with self._lock:
            approved = bool(pending.approved)
            self._pending.pop(key, None)
            return approved

    def resolve(self, run_id: str, approval_id: str, approved: bool) -> bool:
        key = (run_id, approval_id)
        with self._lock:
            if key in self._cancelled:
                self._cancelled.discard(key)
                return False
            pending = self._pending.get(key)
            if pending is None:
                self._early_decisions[key] = approved
                return False
            pending.approved = approved
            pending.event.set()
            return True

    def _mark_cancelled(self, key: tuple[str, str]) -> None:
        with self._lock:
            pending = self._pending.pop(key, None)
            self._early_decisions.pop(key, None)
            self._cancelled.add(key)
            if pending is not None:
                pending.event.set()
