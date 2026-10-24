from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.session import (
    MessageEvent,
    Session,
    SessionContextProjector,
    SessionEntry,
    SessionManager,
)
from minibot.runtime.messages import model_message_to_openai, session_message_to_model


class SessionStoreTests(unittest.TestCase):
    def test_reasoning_content_is_persisted_but_not_sent_to_model(self) -> None:
        event = MessageEvent.create(
            role="assistant",
            content="final answer",
            reasoning_content="private chain",
        )

        self.assertEqual(event.to_dict()["reasoning_content"], "private chain")
        self.assertIsNone(session_message_to_model(event).reasoning_content)
        model_message = session_message_to_model(
            event,
            include_reasoning_content=True,
        )
        self.assertEqual(model_message.reasoning_content, "private chain")
        self.assertEqual(
            model_message_to_openai(
                model_message,
                include_reasoning_content=True,
            )["reasoning_content"],
            "private chain",
        )

    def test_native_layout_uses_session_directory_and_append_only_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")

            user_event = MessageEvent.create(role="user", content="hello")
            manager.append_entries(
                session.session_id, [SessionEntry.from_message(user_event)]
            )

            session_dir = workspace / "sessions" / "s_test"
            self.assertTrue((session_dir / "meta.json").exists())
            self.assertTrue((session_dir / "messages.jsonl").exists())

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 1)
            self.assertEqual(reloaded.messages[0].content, "hello")

    def test_sessions_written_by_minibot_ts_are_ignored_not_touched(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            foreign = workspace / "sessions" / "s_20261003_002929"
            foreign.mkdir(parents=True)
            meta = {"id": foreign.name, "title": "ts", "createdAt": "x", "updatedAt": "y",
                    "workspace": "/w", "messageCount": 2}
            (foreign / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (foreign / "entries.jsonl").write_text("{}\n", encoding="utf-8")
            manager.create_session("s_native")

            self.assertEqual([s.session_id for s in manager.list_sessions()], ["s_native"])
            self.assertIsNone(manager.load(foreign.name))
            self.assertEqual(manager.latest_session().session_id, "s_native")
            self.assertEqual(json.loads((foreign / "meta.json").read_text()), meta)
            self.assertFalse((foreign / "messages.jsonl").exists())

    def test_session_id_validation_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))

            with self.assertRaises(ValueError):
                manager.create_session("../outside")
            self.assertIsNone(manager.load("../outside"))
            self.assertFalse(manager.delete_session("../outside"))

    def test_explicit_create_does_not_overwrite_existing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            event = MessageEvent.create(role="user", content="keep me")
            session.add_message(event)
            manager.save(session)

            with self.assertRaises(FileExistsError):
                manager.create_session("s_test")

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual([message.content for message in reloaded.messages], ["keep me"])

    def test_compaction_entry_projects_summary_without_rewriting_old_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            old_user = MessageEvent.create(role="user", content="old question")
            old_reply = MessageEvent.create(role="assistant", content="old answer")
            session.add_message(old_user)
            session.add_message(old_reply)
            manager.save(session)

            compaction = SessionEntry.compaction(
                summary="old summary",
                first_kept_entry_id=None,
                details={"read_files": ["a.py"], "modified_files": ["b.py"]},
            )
            manager.append_entries("s_test", [compaction])
            reloaded = manager.load("s_test")

            assert reloaded is not None
            self.assertEqual(
                [message.content for message in reloaded.messages],
                ["<conversation-summary>\nold summary\n</conversation-summary>"],
            )
            summary = reloaded.messages[0]
            self.assertEqual(summary.role, "user")
            self.assertEqual(summary.id, f"{compaction.id}_summary")
            self.assertEqual(summary.created_at, compaction.created_at)
            self.assertTrue(SessionContextProjector.is_summary_message(summary))
            raw_lines = (
                workspace / "sessions" / "s_test" / "messages.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 3)
            reloaded_compaction = reloaded.entries[-1]
            self.assertEqual(
                reloaded_compaction.details,
                {"read_files": ["a.py"], "modified_files": ["b.py"]},
            )

    def test_legacy_compaction_entry_without_details_loads(self) -> None:
        record = {
            "type": "compaction",
            "id": "c_legacy",
            "created_at": "2026-01-01T00:00:00+00:00",
            "summary": "legacy summary",
            "first_kept_entry_id": None,
            "tokens_before": 100,
        }

        entry = SessionEntry.from_dict(record)

        self.assertIsNone(entry.details)
        self.assertNotIn("details", entry.to_dict())

    def test_incomplete_tool_transaction_at_tail_is_held_back(self) -> None:
        # A tail block may still be running: the live turn appends its tool
        # results one by one, so it must not be projected until complete.
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            session.add_message(
                MessageEvent.create(role="user", content="run tool")
            )
            session.add_message(_tool_call_message("call_1", "call_2"))
            session.add_message(_tool_result("call_1", "one"))
            manager.save(session)

            reloaded = manager.load("s_test")

            assert reloaded is not None
            self.assertEqual([message.role for message in reloaded.messages], ["user"])

    def test_interrupted_tool_calls_are_kept_with_synthetic_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            session.add_message(
                MessageEvent.create(role="user", content="run tools")
            )
            session.add_message(_tool_call_message("call_1", "call_2"))
            session.add_message(_tool_result("call_2", "two"))
            # The run died; the next turn's input ends the dangling block.
            session.add_message(
                MessageEvent.create(role="user", content="what happened?")
            )
            manager.save(session)

            reloaded = manager.load("s_test")

            assert reloaded is not None
            self.assertEqual(
                [message.role for message in reloaded.messages],
                ["user", "assistant", "tool", "tool", "user"],
            )
            first, second = reloaded.messages[2], reloaded.messages[3]
            # Results follow call order; the recorded one is kept verbatim.
            self.assertEqual(first.tool_call_id, "call_1")
            self.assertEqual(first.name, "echo")
            payload = json.loads(first.content)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["code"], "interrupted")
            self.assertEqual(second.tool_call_id, "call_2")
            self.assertEqual(second.content, "two")
            # Nothing synthetic is written back to disk.
            raw_lines = (
                workspace / "sessions" / "s_test" / "messages.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 4)

    def test_closing_assistant_reply_completes_dangling_tool_block(self) -> None:
        # A cancelled/failed turn is closed by an assistant reply right after
        # its unanswered tool-call block; that reply must end the block too.
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.create_session("s_test")
            session.add_message(MessageEvent.create(role="user", content="run tools"))
            session.add_message(_tool_call_message("call_1", "call_2"))
            session.add_message(
                MessageEvent.create(role="assistant", content="本轮已被用户取消，请求未完成。")
            )
            manager.save(session)

            reloaded = manager.load("s_test")

            assert reloaded is not None
            self.assertEqual(
                [message.role for message in reloaded.messages],
                ["user", "assistant", "tool", "tool", "assistant"],
            )
            for message, call_id in zip(reloaded.messages[2:4], ["call_1", "call_2"], strict=True):
                self.assertEqual(message.tool_call_id, call_id)
                self.assertEqual(json.loads(message.content)["code"], "interrupted")

    def test_total_turn_count_includes_compacted_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.create_session("s_test")
            for text in ("one", "two", "three"):
                session.add_message(MessageEvent.create(role="user", content=text))
                session.add_message(
                    MessageEvent.create(role="assistant", content=f"re: {text}")
                )
            last_user = session.messages[4]
            session.compact_with_summary("earlier", first_kept_entry_id=last_user.id)

            # The projection only shows the kept turn; the total does not shrink.
            self.assertEqual(session.turn_count(), 1)
            self.assertEqual(session.total_turn_count(), 3)

    def test_projected_summary_is_preamble_not_a_turn(self) -> None:
        session = Session("s_test")
        for text in ("one", "two"):
            session.add_message(MessageEvent.create(role="user", content=text))
            session.add_message(
                MessageEvent.create(role="assistant", content=f"re: {text}")
            )
        last_user = session.messages[2]
        session.compact_with_summary("earlier", first_kept_entry_id=last_user.id)

        preamble, turns = session._split_preamble_and_turns()

        # Summary then the kept user message: two user messages in a row.
        self.assertEqual(
            [message.role for message in session.messages],
            ["user", "user", "assistant"],
        )
        self.assertEqual([m.id for m in preamble], [session.messages[0].id])
        self.assertEqual(len(turns), 1)
        self.assertIs(turns[0][0], last_user)
        self.assertEqual(session.turn_count(), 1)

        # A summary with nothing kept leaves no turn at all.
        session.compact_with_summary("everything", first_kept_entry_id=None)
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.turn_count(), 0)
        self.assertEqual(session.total_turn_count(), 2)

    def test_real_user_message_is_not_mistaken_for_summary(self) -> None:
        tagged = MessageEvent.create(
            role="user",
            content="<conversation-summary>\npasted\n</conversation-summary>",
        )
        self.assertFalse(SessionContextProjector.is_summary_message(tagged))


def _tool_call_message(*call_ids: str) -> MessageEvent:
    return MessageEvent.create(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    )


def _tool_result(call_id: str, content: str) -> MessageEvent:
    return MessageEvent.create(
        role="tool",
        tool_call_id=call_id,
        name="echo",
        content=content,
    )


if __name__ == "__main__":
    unittest.main()
