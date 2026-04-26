from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - depends on optional test dependency install
    TestClient = None

from minibot.runtime.events import RuntimeEventEmitter
from minibot.runtime.turn_engine import TurnResult
from minibot.session import MessageEvent, SessionManager
from minibot.server import create_app


class _FakeController:
    def __init__(self) -> None:
        self.resolved = None

    def run_turn(self, *, session_id, user_input, event_handler):
        emitter = RuntimeEventEmitter(
            run_id="r_test",
            session_id=session_id or "s_test",
            handler=event_handler,
        )
        emitter.emit("run.started", {"input": user_input})
        emitter.emit("tool_call.started", {"tool": "read_file"})
        emitter.emit("tool_call.completed", {"tool": "read_file", "ok": True})
        emitter.emit("message.completed", {"content": "ok"})
        emitter.emit("run.completed", {"reply": "ok"})
        return TurnResult(reply="ok", did_compact=False)

    def resolve_approval(self, *, run_id, approval_id, approved):
        self.resolved = (run_id, approval_id, approved)
        return True


def _runtime(manager: SessionManager | None = None, controller: _FakeController | None = None):
    return types.SimpleNamespace(
        controller=controller or _FakeController(),
        manager=manager or SessionManager(Path(tempfile.mkdtemp())),
    )


@unittest.skipIf(TestClient is None, "fastapi is not installed")
class ServerSSETests(unittest.TestCase):
    def test_index_serves_browser_ui(self) -> None:
        runtime = _runtime()
        client = TestClient(create_app(runtime))

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("MiniBot", response.text)
        self.assertIn("/static/app.js", response.text)

    def test_session_messages_endpoint_returns_persisted_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.create_session("s_test")
            user = MessageEvent.create(role="user", content="hello")
            assistant = MessageEvent.create(role="assistant", content="hi")
            session.add_message(user)
            session.add_message(assistant)
            manager.save(session)
            manager.set_current_session("s_test")
            client = TestClient(create_app(_runtime(manager=manager)))

            response = client.get("/sessions/current/messages")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["session"]["session_id"], "s_test")
            self.assertEqual(
                [message["role"] for message in payload["messages"]],
                ["user", "assistant"],
            )
            self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_runs_stream_emits_standard_sse_events(self) -> None:
        runtime = _runtime()
        client = TestClient(create_app(runtime))

        response = client.post("/runs/stream", json={"input": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: run.started", response.text)
        self.assertIn("event: tool_call.started", response.text)
        self.assertIn("event: message.completed", response.text)
        self.assertIn("data:", response.text)

    def test_approval_endpoint_forwards_decision_to_controller(self) -> None:
        controller = _FakeController()
        runtime = _runtime(controller=controller)
        client = TestClient(create_app(runtime))

        response = client.post(
            "/runs/r_test/approvals/ap_test",
            json={"approved": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "matched_pending": True})
        self.assertEqual(controller.resolved, ("r_test", "ap_test", False))


if __name__ == "__main__":
    unittest.main()
