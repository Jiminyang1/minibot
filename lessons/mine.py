"""Mine incidents (things that went wrong) from sessions and ``runs.jsonl``.

Read-only over the state home. An incident is a short, structured record of
one failure; clustering and drafting work on these, never on raw transcripts.

Incident kinds:

- ``tool_error``     a tool result with ``ok: false`` and a failure code
- ``tool_denied``    the user refused an approval (``code: denied``)
- ``turn_failed``    the fixed closing reply of a turn that raised
- ``max_iterations`` the fixed reply when the tool loop hit its limit
- ``run_failed``     a failed run in ``runs.jsonl`` with no session turn to
                     attach to (the session turn wins when both exist)
- ``correction``     a user message right after a final reply that reads as
                     "that was wrong"
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ..run_log import preview_text
from ..runtime.agent_loop import (
    CANCELLED_TURN_REPLY,
    FAILED_TURN_REPLY,
    MAX_ITERATIONS_REPLY,
)
from ..session import MessageEvent, Session, SessionManager
from .store import Evidence

TOOL_FAILURE_CODES = frozenset(
    {"invalid_args", "error", "not_found", "timeout", "permission_denied"}
)

# Deliberately small and explicit: a false positive costs a reviewer one
# "reject", a long fuzzy list floods the review queue.
CORRECTION_PHRASES: tuple[str, ...] = (
    "不对",
    "错了",
    "不是这个",
    "不是我要的",
    "重新",
    "wrong",
    "not what i",
)
_ENGLISH_PHRASE = re.compile(r"[a-z]")

_FAILED_REPLY_RE = re.compile(
    re.escape(FAILED_TURN_REPLY).replace(
        re.escape("{error_type}"), r"(?P<error_type>[^）]+)"
    )
)

_REQUEST_CHARS = 160
_PROBLEM_CHARS = 240
_CORRECTION_CHARS = 160
_ARGS_CHARS = 120


@dataclass(frozen=True)
class Incident:
    id: str
    kind: str
    session_id: str
    request: str
    problem: str
    tool: str | None = None
    code: str | None = None
    args_preview: str | None = None
    correction: str | None = None
    message_ids: tuple[str, ...] = ()
    run_id: str | None = None

    def embed_text(self) -> str:
        """What clustering compares: the failure first, the request second.

        No kind or tool label up front: a shared template prefix pulls
        unrelated failures together (measured with bge-small-zh: failures of
        different tools score 0.72–0.81 with the prefix, 0.61–0.76 without).
        The tool name is already in ``problem``.
        """
        parts = [self.problem]
        if self.correction:
            parts.append(f"用户纠正: {self.correction}")
        parts.append(f"请求: {self.request}")
        return " | ".join(parts)

    def to_prompt_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "request": self.request,
            "problem": self.problem,
        }
        for key in ("tool", "code", "args_preview", "correction"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data

    def to_evidence(self) -> Evidence:
        excerpt = self.problem
        if self.correction:
            excerpt = f"{excerpt} → 用户: {self.correction}"
        return Evidence(
            incident_id=self.id,
            kind=self.kind,
            session_id=self.session_id,
            excerpt=preview_text(f"[{self.request}] {excerpt}", 200),
            message_ids=self.message_ids,
            run_id=self.run_id,
            request=self.request,
        )


def mine_incidents(state_home: Path) -> list[Incident]:
    """All incidents under *state_home*, in a stable order."""
    manager = SessionManager(state_home)
    failed_runs = _failed_runs(state_home / "runs.jsonl")
    claimed_runs: set[str] = set()
    incidents: list[Incident] = []
    # SessionManager lists native sessions only; directories written by
    # minibot-ts are skipped there.
    for listed in sorted(manager.list_sessions(), key=lambda s: s.session_id):
        session = manager.load(listed.session_id)
        if session is None:
            continue
        incidents.extend(_session_incidents(session, failed_runs, claimed_runs))
    for (session_id, _), record in sorted(failed_runs.items()):
        run_id = str(record.get("run_id") or "")
        if run_id in claimed_runs:
            continue
        incidents.append(_run_incident(session_id, record))
    return incidents


# ── sessions ──────────────────────────────────────────────────────


def _session_incidents(
    session: Session,
    failed_runs: dict[tuple[str, int], dict[str, Any]],
    claimed_runs: set[str],
) -> Iterator[Incident]:
    # Entries, not the projection: compacted turns are history too, and the
    # projected summary message is not a user turn.
    messages = [entry.message for entry in session.entries if entry.message is not None]
    request = ""
    request_id = ""
    turn_index = 0
    calls: dict[str, tuple[str, str]] = {}
    previous: MessageEvent | None = None
    for message in messages:
        if message.role == "user":
            if _is_final_reply(previous) and is_correction(message.content):
                assert previous is not None
                yield _incident(
                    "correction",
                    session.session_id,
                    message.id,
                    request=request,
                    problem=f"回复: {preview_text(previous.content, _PROBLEM_CHARS)}",
                    correction=preview_text(message.content, _CORRECTION_CHARS),
                    message_ids=(request_id, previous.id, message.id),
                )
            turn_index += 1
            request = preview_text(message.content, _REQUEST_CHARS)
            request_id = message.id
            calls = {}
        elif message.role == "assistant":
            for call in message.tool_calls or ():
                calls[call.id] = (call.name, call.arguments)
            if not message.tool_calls:
                incident = _closing_incident(
                    session.session_id,
                    message,
                    request,
                    request_id,
                    failed_runs.get((session.session_id, turn_index)),
                    claimed_runs,
                )
                if incident is not None:
                    yield incident
        elif message.role == "tool":
            incident = _tool_incident(session.session_id, message, calls, request, request_id)
            if incident is not None:
                yield incident
        previous = message


def _closing_incident(
    session_id: str,
    message: MessageEvent,
    request: str,
    request_id: str,
    run_record: dict[str, Any] | None,
    claimed_runs: set[str],
) -> Incident | None:
    """The fixed replies a turn ends with when it did not finish."""
    if message.content == MAX_ITERATIONS_REPLY:
        return _incident(
            "max_iterations",
            session_id,
            message.id,
            request=request,
            problem="工具调用轮次达到上限，请求未完成",
            message_ids=(request_id, message.id),
        )
    match = _FAILED_REPLY_RE.fullmatch(message.content)
    if match is None:
        # Includes the cancelled-turn reply: a cancel is the user's choice.
        return None
    problem = f"本轮因错误中止: {match.group('error_type')}"
    run_id = None
    if run_record is not None:
        # The same failure is also a failed run; keep one incident with the
        # run's error message instead of two.
        run_id = str(run_record.get("run_id") or "") or None
        if run_id:
            claimed_runs.add(run_id)
        if run_record.get("error_message_preview"):
            problem += f" — {run_record['error_message_preview']}"
    return _incident(
        "turn_failed",
        session_id,
        message.id,
        request=request,
        problem=preview_text(problem, _PROBLEM_CHARS),
        message_ids=(request_id, message.id),
        run_id=run_id,
    )


def _tool_incident(
    session_id: str,
    message: MessageEvent,
    calls: dict[str, tuple[str, str]],
    request: str,
    request_id: str,
) -> Incident | None:
    try:
        result = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(result, dict) or result.get("ok") is not False:
        return None
    code = str(result.get("code") or "")
    if code == "denied":
        kind = "tool_denied"
    elif code in TOOL_FAILURE_CODES:
        kind = "tool_error"
    else:
        return None
    name, arguments = calls.get(message.tool_call_id or "", (message.name or "?", ""))
    summary = str(result.get("summary") or "")
    return _incident(
        kind,
        session_id,
        message.id,
        request=request,
        problem=preview_text(f"{name} 失败 ({code}): {summary}", _PROBLEM_CHARS),
        tool=name,
        code=code,
        args_preview=preview_text(arguments, _ARGS_CHARS) or None,
        message_ids=(request_id, message.id),
    )


def _is_final_reply(message: MessageEvent | None) -> bool:
    if message is None or message.role != "assistant" or message.tool_calls:
        return False
    content = message.content
    return not (
        content in {CANCELLED_TURN_REPLY, MAX_ITERATIONS_REPLY}
        or _FAILED_REPLY_RE.fullmatch(content)
    )


def is_correction(text: str) -> bool:
    lowered = text.lower()
    for phrase in CORRECTION_PHRASES:
        if _ENGLISH_PHRASE.match(phrase):
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                return True
        elif phrase in lowered:
            return True
    return False


# ── runs.jsonl ────────────────────────────────────────────────────


def _failed_runs(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Failed, non-cancelled runs keyed by ``(session_id, turn_index)``."""
    failed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("status") != "failed":
            continue
        if record.get("error_type") == "RunCancelled":
            continue
        session_id = str(record.get("session_id") or "")
        turn_index = record.get("turn_index")
        if not session_id or not isinstance(turn_index, int):
            continue
        failed[(session_id, turn_index)] = record
    return failed


def _run_incident(session_id: str, record: dict[str, Any]) -> Incident:
    run_id = str(record.get("run_id") or "")
    error_type = str(record.get("error_type") or "Exception")
    message = str(record.get("error_message_preview") or "")
    problem = f"运行失败: {error_type}" + (f" — {message}" if message else "")
    return Incident(
        id=_incident_id("run_failed", session_id, run_id),
        kind="run_failed",
        session_id=session_id,
        request=preview_text(str(record.get("user_input_preview") or ""), _REQUEST_CHARS),
        problem=preview_text(problem, _PROBLEM_CHARS),
        run_id=run_id or None,
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


# ── helpers ───────────────────────────────────────────────────────


def _incident(
    kind: str,
    session_id: str,
    anchor_id: str,
    *,
    request: str,
    problem: str,
    tool: str | None = None,
    code: str | None = None,
    args_preview: str | None = None,
    correction: str | None = None,
    message_ids: tuple[str, ...] = (),
    run_id: str | None = None,
) -> Incident:
    return Incident(
        id=_incident_id(kind, session_id, anchor_id),
        kind=kind,
        session_id=session_id,
        request=request,
        problem=problem,
        tool=tool,
        code=code,
        args_preview=args_preview,
        correction=correction,
        message_ids=tuple(m for m in message_ids if m),
        run_id=run_id,
    )


def _incident_id(kind: str, session_id: str, anchor: str) -> str:
    """Stable across re-mines: derived from where the incident lives."""
    digest = hashlib.sha1(f"{kind}\0{session_id}\0{anchor}".encode()).hexdigest()
    return f"inc_{digest[:12]}"
