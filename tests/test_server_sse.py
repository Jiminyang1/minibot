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
        self.cancelled = None

    def run_turn(
        self,
        *,
        session_id,
        user_input,
        event_handler,
        run_id=None,
        mode="default",
    ):
        del mode
        emitter = RuntimeEventEmitter(
            run_id=run_id or "r_test",
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

    def cancel_run(self, run_id):
        self.cancelled = run_id
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

    def test_create_run_returns_id_and_events_endpoint_replays_backlog(self) -> None:
        runtime = _runtime()
        client = TestClient(create_app(runtime))

        created = client.post("/runs", json={"input": "hi"})

        self.assertEqual(created.status_code, 202)
        run_id = created.json()["run_id"]

        response = client.get(f"/runs/{run_id}/events")

        self.assertEqual(response.status_code, 200)
        self.assertIn("id: 1", response.text)
        self.assertIn("event: run.started", response.text)
        self.assertIn("event: tool_call.started", response.text)
        self.assertIn("event: message.completed", response.text)
        self.assertIn("event: run.completed", response.text)

    def test_events_endpoint_honors_last_event_id(self) -> None:
        runtime = _runtime()
        client = TestClient(create_app(runtime))
        run_id = client.post("/runs", json={"input": "hi"}).json()["run_id"]

        response = client.get(
            f"/runs/{run_id}/events",
            headers={"Last-Event-ID": "2"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("event: run.started", response.text)
        self.assertIn("event: tool_call.completed", response.text)
        self.assertIn("event: run.completed", response.text)

    def test_cancel_endpoint_forwards_run_id_to_controller(self) -> None:
        controller = _FakeController()
        runtime = _runtime(controller=controller)
        client = TestClient(create_app(runtime))

        response = client.post("/runs/r_test/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "run_id": "r_test"})
        self.assertEqual(controller.cancelled, "r_test")

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
