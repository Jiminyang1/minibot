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
from minibot.runtime.agent_loop import TurnOutcome
from minibot.session import MessageEvent, SessionManager
from minibot.server import create_app


class _FakeAgentSession:
    def __init__(self) -> None:
        self.cancelled = None
        self.prompts: list[str] = []

    def prompt(
        self,
        session_id,
        user_input,
        *,
        event_handler,
        run_id=None,
        source=None,
    ):
        self.prompts.append(user_input)
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
        return TurnOutcome(reply="ok", did_compact=False)

    def abort(self, run_id):
        self.cancelled = run_id
        return True


class _FakeApprovalBroker:
    def __init__(self) -> None:
        self.resolved = None

    def resolve(self, run_id, approval_id, approved):
        self.resolved = (run_id, approval_id, approved)
        return True


def _runtime(
    manager: SessionManager | None = None,
    agent_session: _FakeAgentSession | None = None,
    approval_broker: _FakeApprovalBroker | None = None,
):
    manager = manager or SessionManager(Path(tempfile.mkdtemp()))
    return types.SimpleNamespace(
        agent_session=agent_session or _FakeAgentSession(),
        approval_broker=approval_broker or _FakeApprovalBroker(),
        manager=manager,
        budget=types.SimpleNamespace(forget=lambda session_id: None),
    )


def _client(app, host: str = "127.0.0.1"):
    # TestClient defaults to Host "testserver", which the server rejects.
    return TestClient(app, base_url=f"http://{host}")


@unittest.skipIf(TestClient is None, "fastapi is not installed")
class ServerSSETests(unittest.TestCase):
    def test_index_serves_browser_ui(self) -> None:
        runtime = _runtime()
        client = _client(create_app(runtime))

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
            client = _client(create_app(_runtime(manager=manager)))

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
        client = _client(create_app(runtime))

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
        client = _client(create_app(runtime))
        run_id = client.post("/runs", json={"input": "hi"}).json()["run_id"]

        response = client.get(
            f"/runs/{run_id}/events",
            headers={"Last-Event-ID": "2"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("event: run.started", response.text)
        self.assertIn("event: tool_call.completed", response.text)
        self.assertIn("event: run.completed", response.text)

    def test_cancel_endpoint_forwards_run_id_to_agent_session(self) -> None:
        agent_session = _FakeAgentSession()
        runtime = _runtime(agent_session=agent_session)
        client = _client(create_app(runtime))

        response = client.post("/runs/r_test/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "run_id": "r_test"})
        self.assertEqual(agent_session.cancelled, "r_test")

    def test_approval_endpoint_forwards_decision_to_broker(self) -> None:
        broker = _FakeApprovalBroker()
        runtime = _runtime(approval_broker=broker)
        client = _client(create_app(runtime))

        response = client.post(
            "/runs/r_test/approvals/ap_test",
            json={"approved": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "matched_pending": True})
        self.assertEqual(broker.resolved, ("r_test", "ap_test", False))



@unittest.skipIf(TestClient is None, "fastapi is not installed")
class ServerHostOriginTests(unittest.TestCase):
    def test_foreign_host_is_rejected_and_no_run_starts(self) -> None:
        agent_session = _FakeAgentSession()
        client = _client(create_app(_runtime(agent_session=agent_session)))

        created = client.post(
            "/runs", json={"input": "hi"}, headers={"Host": "evil.test"}
        )
        listed = client.get("/sessions", headers={"Host": "evil.test:8765"})

        self.assertEqual(created.status_code, 400)
        self.assertEqual(listed.status_code, 400)
        self.assertEqual(agent_session.prompts, [])

    def test_loopback_hosts_with_and_without_port_are_accepted(self) -> None:
        client = _client(create_app(_runtime()))

        for host in ("localhost:8765", "127.0.0.1", "127.0.0.1:8765", "[::1]:8765"):
            with self.subTest(host=host):
                response = client.get("/sessions", headers={"Host": host})
                self.assertEqual(response.status_code, 200)

    def test_testserver_host_is_not_allowed_by_default(self) -> None:
        client = TestClient(create_app(_runtime()))

        self.assertEqual(client.get("/sessions").status_code, 400)

    def test_extra_allowed_host_is_accepted(self) -> None:
        app = create_app(
            _runtime(), allowed_hosts=["127.0.0.1", "minibot.lan"]
        )

        self.assertEqual(_client(app, "minibot.lan:8765").get("/").status_code, 200)
        self.assertEqual(_client(app, "localhost").get("/").status_code, 400)

    def test_post_with_foreign_origin_is_rejected(self) -> None:
        broker = _FakeApprovalBroker()
        agent_session = _FakeAgentSession()
        client = _client(
            create_app(_runtime(agent_session=agent_session, approval_broker=broker))
        )

        created = client.post(
            "/runs",
            json={"input": "hi"},
            headers={"Origin": "http://evil.test"},
        )
        approved = client.post(
            "/runs/r_test/approvals/ap_test",
            json={"approved": True},
            headers={"Origin": "http://evil.test"},
        )
        null_origin = client.post(
            "/runs/r_test/cancel", headers={"Origin": "null"}
        )

        self.assertEqual(created.status_code, 403)
        self.assertEqual(approved.status_code, 403)
        self.assertEqual(null_origin.status_code, 403)
        self.assertEqual(agent_session.prompts, [])
        self.assertIsNone(broker.resolved)
        self.assertIsNone(agent_session.cancelled)

    def test_post_without_origin_or_with_loopback_origin_is_accepted(self) -> None:
        client = _client(create_app(_runtime()))

        for headers in (
            {},
            {"Origin": "http://127.0.0.1:8765"},
            {"Origin": "http://localhost:8765"},
        ):
            with self.subTest(headers=headers):
                created = client.post("/runs", json={"input": "hi"}, headers=headers)
                self.assertEqual(created.status_code, 202)

    def test_get_with_foreign_origin_is_not_origin_checked(self) -> None:
        # Reads are protected by the Host check; Origin only gates writes.
        client = _client(create_app(_runtime()))

        response = client.get("/sessions", headers={"Origin": "http://evil.test"})

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
