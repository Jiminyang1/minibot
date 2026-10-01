"""Token budget math for one model request.

Owns the input budget, request-size estimation (including the incremental
estimate derived from the provider's observed usage), and the over-budget
error messages. Holds no session data beyond a bounded per-session baseline
used for the incremental estimate.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .context_builder import BuiltRequest
from .messages import session_message_to_model
from .token_budget import (
    RequestTokenEstimate,
    estimate_messages_tokens,
    estimate_request_token_breakdown,
    estimate_request_tokens,
)

if TYPE_CHECKING:
    from ..session import Session


class TokenBudget:
    """Decide whether an assembled request fits the input budget."""

    _MAX_TRACKED_SESSIONS = 512

    def __init__(
        self,
        *,
        context_window_tokens: int,
        model_max_input_tokens: int | None = None,
        max_output_tokens: int,
        compact_token_threshold: int | None = None,
        include_reasoning_content: bool = False,
    ) -> None:
        if context_window_tokens <= 0:
            raise ValueError("context_window_tokens 必须大于 0。")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens 必须大于 0。")
        if max_output_tokens >= context_window_tokens:
            raise ValueError("max_output_tokens 必须小于 context_window_tokens。")
        if model_max_input_tokens is not None:
            if model_max_input_tokens <= 0:
                raise ValueError("model_max_input_tokens 必须大于 0。")
            if model_max_input_tokens > context_window_tokens:
                raise ValueError(
                    "model_max_input_tokens 不能超过 context_window_tokens。"
                )
        context_input_limit = context_window_tokens - max_output_tokens
        input_limit = (
            context_input_limit
            if model_max_input_tokens is None
            else min(context_input_limit, model_max_input_tokens)
        )
        trigger = (
            input_limit
            if compact_token_threshold is None
            else compact_token_threshold
        )
        if trigger <= 0:
            raise ValueError("compact_token_threshold 必须大于 0。")
        if trigger > input_limit:
            raise ValueError(
                "compact_token_threshold 不能超过模型硬输入上限: "
                f"{trigger} > {input_limit}。"
            )
        self.context_window_tokens = context_window_tokens
        self.model_max_input_tokens = model_max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.compact_token_threshold = trigger
        self.include_reasoning_content = include_reasoning_content
        # Per-session message count at the last built request, used to estimate
        # the next request from the observed input usage + only the new messages.
        # Bounded so a long-running server does not accumulate dead sessions.
        self._request_message_counts: dict[str, int] = {}
        self._request_counts_lock = threading.Lock()

    @property
    def input_budget(self) -> int:
        """Hard input limit after reserving the enforced response ceiling."""

        context_input_limit = self.context_window_tokens - self.max_output_tokens
        if self.model_max_input_tokens is None:
            return context_input_limit
        return min(context_input_limit, self.model_max_input_tokens)

    @property
    def compaction_trigger_tokens(self) -> int:
        """Application policy threshold, independent of the model hard limit."""

        return self.compact_token_threshold

    def should_compact(self, request_tokens: int) -> bool:
        return request_tokens > self.compaction_trigger_tokens

    def estimate(self, built: BuiltRequest) -> int:
        """Full token estimate for an assembled request (cold path only)."""
        return estimate_request_tokens(
            built.messages,
            built.tool_definitions,
            include_reasoning_content=self.include_reasoning_content,
        )

    def estimate_breakdown(self, built: BuiltRequest) -> RequestTokenEstimate:
        return estimate_request_token_breakdown(
            built.messages,
            built.tool_definitions,
            include_reasoning_content=self.include_reasoning_content,
        )

    def request_tokens(
        self,
        built: BuiltRequest,
        *,
        session: Session,
        observed_input_tokens: int | None,
    ) -> int:
        if observed_input_tokens is None:
            return self.estimate(built)
        # This intentionally reads without taking _request_counts_lock. A stale
        # or missing baseline only falls back to the full estimate path.
        previous_count = self._request_message_counts.get(session.session_id)
        if previous_count is None or previous_count > len(session.messages):
            return max(observed_input_tokens, self.estimate(built))
        added_messages = session.messages[previous_count:]
        if not added_messages:
            return observed_input_tokens
        added_tokens = estimate_messages_tokens(
            [
                session_message_to_model(
                    message,
                    include_reasoning_content=self.include_reasoning_content,
                )
                for message in added_messages
            ],
            include_reasoning_content=self.include_reasoning_content,
        )
        return observed_input_tokens + added_tokens

    def ensure_fits(
        self,
        built: BuiltRequest,
        *,
        request_tokens: int | None = None,
    ) -> None:
        """Raise a user-actionable error when *built* exceeds the budget."""
        tokens = self.estimate(built) if request_tokens is None else request_tokens
        if tokens <= self.input_budget:
            return
        if built.memory_tokens >= self.input_budget:
            raise RuntimeError(
                "当前用户长期记忆占用过大，已超过输入预算。"
                "请删除部分 `/memory` 条目后重试。"
            )
        raise RuntimeError(
            "当前上下文仍然超过输入预算，请手动 `/compact` 或开启新会话后重试。"
        )

    def remember(self, session: Session) -> None:
        with self._request_counts_lock:
            counts = self._request_message_counts
            # pop-then-set moves the key to the end so eviction is true LRU.
            # (updating an existing key in place would not reorder it, which
            # would let an early-but-active session be evicted first.)
            counts.pop(session.session_id, None)
            counts[session.session_id] = len(session.messages)
            overflow = len(counts) - self._MAX_TRACKED_SESSIONS
            if overflow > 0:
                # dicts preserve insertion order; drop the least-recently-updated.
                for stale_id in list(counts.keys())[:overflow]:
                    counts.pop(stale_id, None)

    def forget(self, session_id: str) -> None:
        """Drop the cached request-size baseline for a deleted/closed session."""
        with self._request_counts_lock:
            self._request_message_counts.pop(session_id, None)
