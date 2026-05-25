from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.session import MessageEvent, SessionEntry, SessionManager
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

            session_dir = workspace / ".minibot" / "sessions" / "s_test"
            self.assertTrue((session_dir / "meta.json").exists())
            self.assertTrue((session_dir / "messages.jsonl").exists())

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 1)
            self.assertEqual(reloaded.messages[0].content, "hello")

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
                ["[Summary of earlier conversation]\nold summary"],
            )
            raw_lines = (
                workspace / ".minibot" / "sessions" / "s_test" / "messages.jsonl"
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

    def test_incomplete_tool_transaction_is_not_projected_back_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            session = manager.create_session("s_test")
            session.add_message(
                MessageEvent.create(role="user", content="run tool")
            )
            session.add_message(
                MessageEvent.create(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": "{}",
                            },
                        }
                    ],
                )
            )
            manager.save(session)

            reloaded = manager.load("s_test")

            assert reloaded is not None
            self.assertEqual([message.role for message in reloaded.messages], ["user"])


if __name__ == "__main__":
    unittest.main()
