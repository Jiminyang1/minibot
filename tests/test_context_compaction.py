from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.compaction import (
    SummaryRequest,
    find_cut_point,
    prepare_compaction,
)
from minibot.runtime.context_manager import (
    ContextWindowManager,
    estimate_messages_tokens,
)
from minibot.runtime.messages import (
    format_model_messages_for_summary,
    session_message_to_model,
)
from minibot.session import MessageEvent, Session
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

    def test_build_context_appends_compaction_entry_at_token_cut_point(self) -> None:
        session = Session("s_test")
        session.add_message(MessageEvent.create(role="user", content="old " * 2000))
        session.add_message(MessageEvent.create(role="assistant", content="answer"))
        current_user = MessageEvent.create(role="user", content="current question")
        session.add_message(current_user)
        summarized_roles: list[str] = []
        manager = ContextWindowManager(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=ToolRegistry(),
            compact_token_threshold=5000,
            reserved_completion_tokens=20,
            compact_keep_recent_tokens=1,
            summarizer=lambda request: summarized_roles.extend(
                message.role for message in request.messages
            )
            or "summary",
        )

        context = manager.build_context(session=session, observed_input_tokens=10000)

        self.assertTrue(context.did_compact)
        pending_entries = session.pop_pending_compaction_entries()
        self.assertEqual(len(pending_entries), 1)
        pending = pending_entries[0]
        self.assertEqual(pending.type, "compaction")
        self.assertIsNotNone(pending.first_kept_entry_id)
        self.assertIn("user", summarized_roles)
        self.assertEqual(session.entries[-1], pending)
        self.assertIn(
            "[Summary of earlier conversation]\nsummary",
            [message.content for message in session.messages],
        )

    def test_oversized_read_only_tool_tail_is_dropped_as_one_block(self) -> None:
        registry = ToolRegistry()
        registry.register(_DummyTool(read_only=True))
        session = Session("s_test")
        user = MessageEvent.create(role="user", content="run echo")
        assistant = MessageEvent.create(
            role="assistant",
            content="",
            tool_calls=[_tool_call()],
        )
        large_tool_content = "alpha beta gamma delta epsilon " * 2000
        tool = MessageEvent.create(
            role="tool",
            content=large_tool_content,
            tool_call_id="call_1",
            name="echo",
        )
        for message in (user, assistant, tool):
            session.add_message(message)

        manager = ContextWindowManager(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=registry,
            compact_token_threshold=2000,
            reserved_completion_tokens=100,
            compact_keep_recent_tokens=1,
            summarizer=lambda request: "summary before tool block",
        )

        context = manager.build_context(session=session)

        self.assertTrue(context.did_compact)
        self.assertIn("已丢弃过大的只读工具事务块", context.compact_message or "")
        self.assertEqual(len(session.entries), 4)
        self.assertEqual(session.entries[-1].type, "compaction")
        self.assertNotIn(large_tool_content, [message.content for message in session.messages])
        self.assertIn("summary before tool block", session.messages[0].content)
        self.assertIn("Omitted oversized read-only tool transaction", session.messages[0].content)

    def test_oversized_non_read_only_tool_tail_fails_instead_of_dropping(self) -> None:
        registry = ToolRegistry()
        registry.register(_DummyTool(read_only=False))
        session = Session("s_test")
        session.add_message(MessageEvent.create(role="user", content="run echo"))
        session.add_message(
            MessageEvent.create(
                role="assistant",
                content="",
                tool_calls=[_tool_call()],
            )
        )
        session.add_message(
            MessageEvent.create(
                role="tool",
                content="alpha beta gamma delta epsilon " * 2000,
                tool_call_id="call_1",
                name="echo",
            )
        )
        manager = ContextWindowManager(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=registry,
            compact_token_threshold=2000,
            reserved_completion_tokens=100,
            compact_keep_recent_tokens=1,
            summarizer=lambda request: "summary",
        )

        with self.assertRaisesRegex(RuntimeError, "包含非只读工具"):
            manager.build_context(session=session)

    def test_repeated_compaction_passes_previous_summary_and_merges_file_details(self) -> None:
        session = Session("s_test")
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
        for message in (read_call, user, write_call, write_result, current_user):
            session.add_message(message)
        session.compact_with_summary(
            "previous summary",
            first_kept_entry_id=user.id,
            tokens_before=123,
            details={"read_files": ["old.py"], "modified_files": ["a.py"]},
        )
        session.pop_pending_compaction_entries()
        seen_requests: list[SummaryRequest] = []
        manager = ContextWindowManager(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=ToolRegistry(),
            compact_token_threshold=5000,
            reserved_completion_tokens=20,
            compact_keep_recent_tokens=1,
            summarizer=lambda request: seen_requests.append(request) or "updated summary",
        )

        context = manager.build_context(session=session, observed_input_tokens=10000)

        self.assertTrue(context.did_compact)
        self.assertEqual(seen_requests[0].previous_summary, "previous summary")
        pending = session.pop_pending_compaction_entries()[0]
        self.assertEqual(
            pending.details,
            {"read_files": ["old.py"], "modified_files": ["a.py", "b.py"]},
        )
        self.assertIn("<read-files>\nold.py\n</read-files>", pending.summary or "")
        self.assertIn("<modified-files>\na.py\nb.py\n</modified-files>", pending.summary or "")

    def test_read_artifact_file_result_updates_read_file_details(self) -> None:
        session = Session("s_test")
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
        for message in (read_call, read_result, current_user):
            session.add_message(message)
        manager = ContextWindowManager(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=ToolRegistry(),
            compact_token_threshold=5000,
            reserved_completion_tokens=20,
            compact_keep_recent_tokens=1,
            summarizer=lambda request: "summary",
        )

        context = manager.build_context(session=session, observed_input_tokens=10000)

        self.assertTrue(context.did_compact)
        pending = session.pop_pending_compaction_entries()[0]
        self.assertEqual(
            pending.details,
            {"read_files": ["large.py"], "modified_files": []},
        )
        self.assertIn("<read-files>\nlarge.py\n</read-files>", pending.summary or "")

    def test_tool_drop_summary_skips_projected_previous_summary_message(self) -> None:
        session = Session("s_test")
        session.compact_with_summary(
            "previous summary",
            first_kept_entry_id=None,
            tokens_before=123,
        )
        session.pop_pending_compaction_entries()
        prefix_user = MessageEvent.create(role="user", content="new prefix")
        prefix = [*session.messages, prefix_user]
        seen_requests: list[SummaryRequest] = []
        manager = ContextWindowManager(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=ToolRegistry(),
            compact_token_threshold=5000,
            reserved_completion_tokens=20,
            compact_keep_recent_tokens=1,
            summarizer=lambda request: seen_requests.append(request) or "updated summary",
        )

        summary = manager._summarize_prefix_before_tool_drop(
            session,
            prefix,
            tool_names=["read_file"],
            cancel_event=None,
        )

        self.assertEqual(seen_requests[0].previous_summary, "previous summary")
        self.assertEqual([message.content for message in seen_requests[0].messages], ["new prefix"])
        self.assertNotIn("Summary of earlier conversation", seen_requests[0].messages[0].content)
        self.assertIn("updated summary", summary)
        self.assertIn("Omitted oversized read-only tool transaction", summary)


if __name__ == "__main__":
    unittest.main()
