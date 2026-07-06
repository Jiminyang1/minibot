"""Cross-session history search: the agent's access to its own past.

Agentic search over raw session logs and compaction summaries — no vector
store, no pre-digestion. At personal-assistant scale (one user, thousands
of messages) keyword search over the source of truth beats an embedding
pipeline on transparency and freshness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from .base import Tool, ToolExecutionContext
from .result import ToolOutput

if TYPE_CHECKING:
    from ..session import Session, SessionManager


class SearchHistoryTool(Tool):
    """Keyword search across all stored conversations."""

    _DEFAULT_LIMIT = 8
    _MAX_LIMIT = 20
    _MAX_SESSIONS_SCANNED = 200
    _SNIPPET_BEFORE = 80
    _SNIPPET_AFTER = 160

    def __init__(self, session_manager: "SessionManager") -> None:
        super().__init__()
        self._sessions = session_manager

    @property
    def name(self) -> str:
        return "search_history"

    @property
    def description(self) -> str:
        return (
            "跨会话搜索历史对话（含被压缩会话的摘要）。当用户提到"
            "“上次 / 之前 / 那次聊过 / 我说过 / 我们讨论过”等指向过去对话的内容时使用。"
            "多个关键词用空格分隔，全部命中才算匹配；结果带 session_id 便于追溯。"
            "找不到时先减少或更换关键词再试。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，空格分隔多个词（AND 关系），大小写不敏感",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "只搜最近 N 天的会话",
                },
                "session_id": {
                    "type": "string",
                    "description": "只搜指定会话",
                },
                "workspace": {
                    "type": "string",
                    "description": "按会话创建时的工作目录过滤（子串匹配）",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "最多返回多少条匹配（默认 8）",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    @property
    def read_only(self) -> bool:
        return True

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        query: str,
        days: int | None = None,
        session_id: str | None = None,
        workspace: str | None = None,
        limit: int | None = None,
    ) -> ToolOutput:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return ToolOutput.failure(
                "invalid_args",
                "query 不能为空。",
                data={"tool": self.name},
            )
        limit = min(limit or self._DEFAULT_LIMIT, self._MAX_LIMIT)
        cutoff = None
        if days is not None:
            cutoff = (
                (datetime.now(UTC) - timedelta(days=days))
                .isoformat()
                .replace("+00:00", "Z")
            )

        matches: list[dict[str, Any]] = []
        searched = 0
        for meta in self._sessions.list_sessions():
            if searched >= self._MAX_SESSIONS_SCANNED:
                break
            sid = meta.session_id
            if sid == context.session_id:
                # The ongoing conversation is "now", not history.
                continue
            if session_id is not None and sid != session_id:
                continue
            if workspace is not None and workspace not in (meta.workspace or ""):
                continue
            if cutoff is not None and (meta.updated_at or "") < cutoff:
                continue
            session = self._sessions.load(sid)
            if session is None:
                continue
            searched += 1
            matches.extend(self._match_session(session, terms, cutoff))

        matches.sort(key=lambda item: item["created_at"], reverse=True)
        shown = matches[:limit]

        if not shown:
            return ToolOutput.success(
                f"在 {searched} 个历史会话中未找到匹配，可减少或更换关键词。",
                data={"matches": [], "searched_sessions": searched, "query": query},
            )

        lines: list[str] = []
        for index, item in enumerate(shown, start=1):
            lines.append(
                f"[{index}] {item['session_id']} · {item['title']} · "
                f"{item['created_at'][:10]} · {item['kind']}"
            )
            lines.append(f"    {item['snippet']}")
        return ToolOutput.success(
            f"在 {searched} 个历史会话中找到 {len(matches)} 条匹配"
            f"（展示最新 {len(shown)} 条）。",
            data={
                "matches": shown,
                "total_matches": len(matches),
                "searched_sessions": searched,
            },
            content="\n".join(lines),
            content_kind="text",
            content_name="history_search",
        )

    def _match_session(
        self,
        session: "Session",
        terms: list[str],
        cutoff: str | None,
    ) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for entry in session.entries:
            if entry.type == "compaction":
                text = entry.summary or ""
                kind = "summary"
                created_at = entry.created_at or ""
            elif entry.message is not None:
                text = entry.message.content or ""
                kind = entry.message.role
                created_at = entry.message.created_at or ""
            else:
                continue
            if cutoff is not None and created_at and created_at < cutoff:
                continue
            lowered = text.lower()
            if not all(term in lowered for term in terms):
                continue
            found.append(
                {
                    "session_id": session.session_id,
                    "title": session.title,
                    "workspace": session.workspace,
                    "kind": kind,
                    "created_at": created_at,
                    "snippet": self._snippet(text, terms[0]),
                }
            )
        return found

    def _snippet(self, text: str, first_term: str) -> str:
        compact = " ".join(text.split())
        lowered = compact.lower()
        position = lowered.find(first_term)
        if position < 0:
            position = 0
        start = max(0, position - self._SNIPPET_BEFORE)
        end = min(len(compact), position + self._SNIPPET_AFTER)
        snippet = compact[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(compact):
            snippet = snippet + "…"
        return snippet
