from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.budget import TokenBudget
from minibot.runtime.compaction import (
    SummaryRequest,
    find_cut_point,
    prepare_compaction,
)
from minibot.runtime.compactor import Compactor
from minibot.runtime.context_builder import ContextBuilder
from minibot.runtime.messages import (
    format_model_messages_for_summary,
    session_message_to_model,
)
from minibot.runtime.token_budget import estimate_messages_tokens
from minibot.session import MessageEvent, Session, SessionManager
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput


def _tool_call(name: str = "echo", arguments: str = "{}") -> dict[str, object]:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class _DummyTool(Tool):
    def __init__(self, *, read_only: bool) -> None:
        super().__init__()
        self._read_only = read_only

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @property
    def read_only(self) -> bool:
        return self._read_only

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        return ToolOutput.success("ok")


class _Harness:
    """A Compactor wired to a real on-disk session for one test."""

    def __init__(
        self,
        workspace: Path,
        *,
        tool_registry: ToolRegistry | None = None,
        compact_token_threshold: int = 5000,
        reserved_completion_tokens: int = 20,
        keep_recent_tokens: int = 1,
        summarizer=None,
    ) -> None:
        self.manager = SessionManager(workspace)
        self.session = self.manager.create_session("s_test")
        registry = tool_registry or ToolRegistry()
        self.context_builder = ContextBuilder(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=registry,
        )
        self.budget = TokenBudget(
            compact_token_threshold=compact_token_threshold,
            reserved_completion_tokens=reserved_completion_tokens,
        )
        self.compactor = Compactor(
            session_manager=self.manager,
            context_builder=self.context_builder,
            budget=self.budget,
            tool_registry=registry,
            summarizer=summarizer or (lambda request: "summary"),
            keep_recent_tokens=keep_recent_tokens,
        )

    def add(self, *messages: MessageEvent) -> None:
        from minibot.session import SessionEntry

        for message in messages:
            self.session.add_message(message)
            self.manager.append_entries(
                self.session.session_id, [SessionEntry.from_message(message)]
            )

    def persisted_entry_types(self) -> list[str]:
        path = (
            self.manager.sessions_dir / self.session.session_id / "messages.jsonl"
        )
        return [
            json.loads(line)["type"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class ContextCompactionTests(unittest.TestCase):
    def test_find_cut_point_prefers_user_turn_boundary(self) -> None:
        messages = [
            MessageEvent.create(role="user", content="old question " * 20),
            MessageEvent.create(role="assistant", content="old answer"),
            MessageEvent.create(role="user", content="recent question"),
            MessageEvent.create(role="assistant", content="recent answer"),
        ]
        keep_recent_tokens = (
            estimate_messages_tokens([session_message_to_model(messages[-1])]) + 1
        )

        cut = find_cut_point(messages, keep_recent_tokens=keep_recent_tokens)

        self.assertEqual(cut, 2)
        self.assertEqual(messages[cut].role, "user")

    def test_find_cut_point_never_starts_kept_slice_with_tool_result(self) -> None:
        user = MessageEvent.create(role="user", content="run tool")
        assistant = MessageEvent.create(
            role="assistant",
            content="",
            tool_calls=[_tool_call()],
        )
        tool = MessageEvent.create(
            role="tool",
            content="x" * 2000,
            tool_call_id="call_1",
            name="echo",
        )
        final = MessageEvent.create(role="assistant", content="done")
        keep_recent_tokens = estimate_messages_tokens(
            [session_message_to_model(final)]
        ) + max(1, estimate_messages_tokens([session_message_to_model(tool)]) // 2)

        cut = find_cut_point(
            [user, assistant, tool, final],
            keep_recent_tokens,
        )

        self.assertEqual(cut, 3)
        self.assertNotEqual([user, assistant, tool, final][cut].role, "tool")

    def test_prepare_compaction_splits_only_oversized_turn(self) -> None:
        old_user = MessageEvent.create(role="user", content="old question")
        old_answer = MessageEvent.create(role="assistant", content="old answer")
        current_user = MessageEvent.create(role="user", content="current request")
        current_answer = MessageEvent.create(role="assistant", content="a" * 10000)
        messages = [old_user, old_answer, current_user, current_answer]

        preparation = prepare_compaction(messages, keep_recent_tokens=10)

        assert preparation is not None
        self.assertTrue(preparation.is_split_turn)
        self.assertEqual(preparation.messages_to_summarize, [old_user, old_answer])
        self.assertEqual(preparation.turn_prefix_messages, [current_user])
        self.assertEqual(messages[preparation.first_kept_message_index], current_answer)

    def test_summary_serialization_truncates_tool_results(self) -> None:
        message = MessageEvent.create(
            role="tool",
            content="x" * 2100,
            tool_call_id="call_1",
            name="read_file",
        )

        formatted = format_model_messages_for_summary([session_message_to_model(message)])

        self.assertIn("TOOL_RESULT[read_file]:", formatted)
        self.assertIn("characters truncated for summary", formatted)
        self.assertLess(len(formatted), 2100)

    def test_reduce_appends_and_persists_compaction_entry_at_cut_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summarized_roles: list[str] = []
            harness = _Harness(
                Path(tmpdir),
                summarizer=lambda request: summarized_roles.extend(
                    message.role for message in request.messages
                )
                or "summary",
            )
            harness.add(
                MessageEvent.create(role="user", content="old " * 2000),
                MessageEvent.create(role="assistant", content="answer"),
                MessageEvent.create(role="user", content="current question"),
            )

            message = harness.compactor.reduce(harness.session, tokens_before=10000)

            self.assertIn("已压缩", message)
            self.assertIn("user", summarized_roles)
            self.assertEqual(harness.session.entries[-1].type, "compaction")
            self.assertIsNotNone(harness.session.entries[-1].first_kept_entry_id)
            self.assertIn(
                "[Summary of earlier conversation]\nsummary",
                [msg.content for msg in harness.session.messages],
            )
            # Persistence happens inside reduce — no pending state to flush.
            self.assertEqual(
                harness.persisted_entry_types(),
                ["message", "message", "message", "compaction"],
            )

    def test_oversized_read_only_tool_tail_is_dropped_as_one_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(_DummyTool(read_only=True))
            harness = _Harness(
                Path(tmpdir),
                tool_registry=registry,
                compact_token_threshold=2000,
                reserved_completion_tokens=100,
                summarizer=lambda request: "summary before tool block",
            )
            large_tool_content = "alpha beta gamma delta epsilon " * 2000
            harness.add(
                MessageEvent.create(role="user", content="run echo"),
                MessageEvent.create(
                    role="assistant", content="", tool_calls=[_tool_call()]
                ),
                MessageEvent.create(
                    role="tool",
                    content=large_tool_content,
                    tool_call_id="call_1",
                    name="echo",
                ),
            )
            tokens_before = harness.budget.estimate(
                harness.context_builder.build(harness.session.messages)
            )

            message = harness.compactor.reduce(
                harness.session, tokens_before=tokens_before
            )

            self.assertIn("已丢弃过大的只读工具事务块", message)
            self.assertEqual(len(harness.session.entries), 4)
            self.assertEqual(harness.session.entries[-1].type, "compaction")
            self.assertNotIn(
                large_tool_content,
                [msg.content for msg in harness.session.messages],
            )
            self.assertIn(
                "summary before tool block", harness.session.messages[0].content
            )
            self.assertIn(
                "Omitted oversized read-only tool transaction",
                harness.session.messages[0].content,
            )
            self.assertEqual(harness.persisted_entry_types()[-1], "compaction")

    def test_oversized_non_read_only_tool_tail_fails_instead_of_dropping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(_DummyTool(read_only=False))
            harness = _Harness(
                Path(tmpdir),
                tool_registry=registry,
                compact_token_threshold=2000,
                reserved_completion_tokens=100,
            )
            harness.add(
                MessageEvent.create(role="user", content="run echo"),
                MessageEvent.create(
                    role="assistant", content="", tool_calls=[_tool_call()]
                ),
                MessageEvent.create(
                    role="tool",
                    content="alpha beta gamma delta epsilon " * 2000,
                    tool_call_id="call_1",
                    name="echo",
                ),
            )
            tokens_before = harness.budget.estimate(
                harness.context_builder.build(harness.session.messages)
            )

            with self.assertRaisesRegex(RuntimeError, "包含非只读工具"):
                harness.compactor.reduce(harness.session, tokens_before=tokens_before)

    def test_repeated_compaction_passes_previous_summary_and_merges_file_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seen_requests: list[SummaryRequest] = []
            harness = _Harness(
                Path(tmpdir),
                summarizer=lambda request: seen_requests.append(request)
                or "updated summary",
            )
            read_call = MessageEvent.create(
                role="assistant",
                content="",
                tool_calls=[_tool_call("read_file", '{"path": "a.py"}')],
            )
            user = MessageEvent.create(role="user", content="change file")
            write_call = MessageEvent.create(
                role="assistant",
                content="",
                tool_calls=[_tool_call("edit_file", '{"path": "b.py"}')],
            )
            write_result = MessageEvent.create(
                role="tool",
                content="ok",
                tool_call_id="call_1",
                name="edit_file",
            )
            current_user = MessageEvent.create(role="user", content="current")
            harness.add(read_call, user, write_call, write_result, current_user)
            entry = harness.session.compact_with_summary(
                "previous summary",
                first_kept_entry_id=user.id,
                tokens_before=123,
                details={"read_files": ["old.py"], "modified_files": ["a.py"]},
            )
            harness.manager.append_entries(harness.session.session_id, [entry])

            harness.compactor.reduce(harness.session, tokens_before=10000)

            self.assertEqual(seen_requests[0].previous_summary, "previous summary")
            latest = harness.session.entries[-1]
            self.assertEqual(latest.type, "compaction")
            self.assertEqual(
                latest.details,
                {"read_files": ["old.py"], "modified_files": ["a.py", "b.py"]},
            )
            self.assertIn("<read-files>\nold.py\n</read-files>", latest.summary or "")
            self.assertIn(
                "<modified-files>\na.py\nb.py\n</modified-files>", latest.summary or ""
            )

    def test_read_artifact_file_result_updates_read_file_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = _Harness(Path(tmpdir))
            read_call = MessageEvent.create(
                role="assistant",
                content="",
                tool_calls=[_tool_call("read_artifact", '{"artifact_id": "art_1"}')],
            )
            read_result = MessageEvent.create(
                role="tool",
                content=json.dumps(
                    {
                        "ok": True,
                        "code": "success",
                        "summary": "read artifact",
                        "data": {"kind": "file", "name": "large.py"},
                        "artifact": None,
                        "truncated": False,
                    },
                    separators=(",", ":"),
                ),
                tool_call_id="call_1",
                name="read_artifact",
            )
            current_user = MessageEvent.create(role="user", content="current")
            harness.add(read_call, read_result, current_user)

            harness.compactor.reduce(harness.session, tokens_before=10000)

            latest = harness.session.entries[-1]
            self.assertEqual(
                latest.details,
                {"read_files": ["large.py"], "modified_files": []},
            )
            self.assertIn("<read-files>\nlarge.py\n</read-files>", latest.summary or "")

    def test_tool_drop_summary_skips_projected_previous_summary_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seen_requests: list[SummaryRequest] = []
            harness = _Harness(
                Path(tmpdir),
                summarizer=lambda request: seen_requests.append(request)
                or "updated summary",
            )
            entry = harness.session.compact_with_summary(
                "previous summary",
                first_kept_entry_id=None,
                tokens_before=123,
            )
            harness.manager.append_entries(harness.session.session_id, [entry])
            prefix_user = MessageEvent.create(role="user", content="new prefix")
            prefix = [*harness.session.messages, prefix_user]

            summary = harness.compactor._summarize_prefix_before_tool_drop(
                harness.session,
                prefix,
                tool_names=["read_file"],
                cancel_event=None,
            )

            self.assertEqual(seen_requests[0].previous_summary, "previous summary")
            self.assertEqual(
                [message.content for message in seen_requests[0].messages],
                ["new prefix"],
            )
            self.assertNotIn(
                "Summary of earlier conversation",
                seen_requests[0].messages[0].content,
            )
            self.assertIn("updated summary", summary)
            self.assertIn("Omitted oversized read-only tool transaction", summary)

    def test_manual_compact_reports_noop_when_nothing_to_do(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = _Harness(Path(tmpdir), keep_recent_tokens=100000)
            harness.add(MessageEvent.create(role="user", content="hi"))

            did_compact, message = harness.compactor.compact_now(harness.session)

            self.assertFalse(did_compact)
            self.assertIn("没有可在安全切点处压缩", message)


if __name__ == "__main__":
    unittest.main()
