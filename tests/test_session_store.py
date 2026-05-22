from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.session import MessageEvent, SessionManager
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
            session.add_message(user_event)
            manager.append_messages(session.session_id, [user_event])
            manager.update_metadata(session)

            session_dir = workspace / ".minibot" / "sessions" / "s_test"
            self.assertTrue((session_dir / "meta.json").exists())
            self.assertTrue((session_dir / "messages.jsonl").exists())

            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.message_count, 1)
            self.assertEqual(reloaded.messages[0].content, "hello")

if __name__ == "__main__":
    unittest.main()
