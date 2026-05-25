"""Run cancellation primitives shared across runtime layers."""

from __future__ import annotations


class RunCancelled(RuntimeError):
    """Raised when a run is cooperatively cancelled."""

