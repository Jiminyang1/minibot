from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.cancel import RunCancelled
from minibot.runtime.agent_session import AgentSession, SessionBusyError
from minibot.runtime.approvals import ApprovalBroker
from minibot.runtime.events import RuntimeEvent, RuntimeEventEmitter
from minibot.runtime.hooks_builtin import ApprovalRequest
from minibot.runtime.turn_engine import TurnResult
from minibot.session import SessionManager, SessionNotFoundError


class _BlockingTurnEngine:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_seen = threading.Event()

    def handle_turn(
        self,
        session,
        user_input,
        *,
        run_id=None,
        event_emitter=None,
        cancel_event=None,
        mode="default",
    ):
        del session, user_input, run_id, mode
        self.started.set()
        self.release.wait(timeout=2)
        if cancel_event is not None and cancel_event.is_set():
            self.cancel_seen.set()
            raise RunCancelled("run cancelled by test")
        if event_emitter is not None:
            event_emitter.emit("message.completed", {"content": "ok"})
        return TurnResult(reply="ok", did_compact=False)


class RuntimeEventTests(unittest.TestCase):
    def test_runtime_event_emitter_generates_stable_ids_and_sequence(self) -> None:
        emitted: list[RuntimeEvent] = []
        emitter = RuntimeEventEmitter(
            run_id="r_test",
            session_id="s_test",
            handler=emitted.append,
        )

        first = emitter.emit("run.started", {})
        second = emitter.emit("run.completed", {"reply": "ok"})

        self.assertEqual(first.id, "r_test:1")
        self.assertEqual(second.id, "r_test:2")
        self.assertEqual([event.seq for event in emitted], [1, 2])
        self.assertEqual(emitted[1].payload["reply"], "ok")

    def test_approval_broker_blocks_until_resolution(self) -> None:
        broker = ApprovalBroker()
        request = ApprovalRequest(
            run_id="r_test",
            session_id="s_test",
            approval_id="ap_test",
            tool_call_id="call_1",
            tool_name="exec",
            args={},
        )
        result: list[bool] = []

        thread = threading.Thread(target=lambda: result.append(broker.wait(request)))
        thread.start()
        time.sleep(0.02)

        self.assertEqual(result, [])
        self.assertTrue(broker.resolve("r_test", "ap_test", True))
        thread.join(timeout=2)

        self.assertEqual(result, [True])

    def test_approval_broker_unblocks_when_run_is_cancelled(self) -> None:
        broker = ApprovalBroker()
        cancel_event = threading.Event()
        request = ApprovalRequest(
            run_id="r_test",
            session_id="s_test",
            approval_id="ap_test",
            tool_call_id="call_1",
            tool_name="exec",
            args={},
        )
        errors: list[BaseException] = []

        def wait() -> None:
            try:
                broker.wait(request, cancel_event=cancel_event, poll_interval=0.01)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.02)
        cancel_event.set()
        thread.join(timeout=2)

        self.assertIsInstance(errors[0], RunCancelled)
        self.assertFalse(broker.resolve("r_test", "ap_test", True))

    def test_agent_session_rejects_concurrent_turns_for_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            manager.create_session("s_test")
            engine = _BlockingTurnEngine()
            agent_session = AgentSession(turn_engine=engine, session_manager=manager)

            first_events: list[RuntimeEvent] = []
            first_thread = threading.Thread(
                target=lambda: agent_session.prompt(
                    "s_test",
                    "hello",
                    event_handler=first_events.append,
                )
            )
            first_thread.start()
            self.assertTrue(engine.started.wait(timeout=2))

            with self.assertRaises(SessionBusyError):
                agent_session.prompt(
                    "s_test",
                    "second",
                    event_handler=lambda event: None,
                )

            engine.release.set()
            first_thread.join(timeout=2)

            self.assertIn("run.started", [event.type for event in first_events])
            self.assertIn("run.completed", [event.type for event in first_events])

    def test_agent_session_reports_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            engine = _BlockingTurnEngine()
            agent_session = AgentSession(turn_engine=engine, session_manager=manager)

            with self.assertRaises(SessionNotFoundError):
                agent_session.prompt(
                    "missing",
                    "hello",
                    event_handler=lambda event: None,
                )

    def test_agent_session_abort_signals_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            manager.create_session("s_test")
            engine = _BlockingTurnEngine()
            agent_session = AgentSession(turn_engine=engine, session_manager=manager)
            events: list[RuntimeEvent] = []
            errors: list[BaseException] = []

            def run() -> None:
                try:
                    agent_session.prompt(
                        "s_test",
                        "hello",
                        run_id="r_abort",
                        event_handler=events.append,
                    )
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(engine.started.wait(timeout=2))

            self.assertTrue(agent_session.abort("r_abort"))
            engine.release.set()
            thread.join(timeout=2)

            self.assertTrue(engine.cancel_seen.is_set())
            self.assertIsInstance(errors[0], RunCancelled)
            self.assertIn("run.cancelled", [event.type for event in events])
            self.assertFalse(agent_session.abort("r_abort"))
            self.assertFalse(agent_session.is_busy("s_test"))


if __name__ == "__main__":
    unittest.main()
