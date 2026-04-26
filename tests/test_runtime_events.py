from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.agent_runner import ApprovalRequest
from minibot.runtime.controller import ApprovalBroker, RunController, SessionBusyError
from minibot.runtime.events import RuntimeEvent, RuntimeEventEmitter
from minibot.runtime.turn_engine import TurnResult
from minibot.session import SessionManager


class _BlockingTurnEngine:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def handle_turn(self, session, user_input, *, run_id=None, event_emitter=None):
        del session, user_input, run_id
        self.started.set()
        self.release.wait(timeout=2)
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

    def test_run_controller_rejects_concurrent_turns_for_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            manager.create_session("s_test")
            engine = _BlockingTurnEngine()
            controller = RunController(turn_engine=engine, manager=manager)

            first_events: list[RuntimeEvent] = []
            first_thread = threading.Thread(
                target=lambda: controller.run_turn(
                    session_id="s_test",
                    user_input="hello",
                    event_handler=first_events.append,
                )
            )
            first_thread.start()
            self.assertTrue(engine.started.wait(timeout=2))

            with self.assertRaises(SessionBusyError):
                controller.run_turn(
                    session_id="s_test",
                    user_input="second",
                    event_handler=lambda event: None,
                )

            engine.release.set()
            first_thread.join(timeout=2)

            self.assertIn("run.started", [event.type for event in first_events])
            self.assertIn("run.completed", [event.type for event in first_events])


if __name__ == "__main__":
    unittest.main()
