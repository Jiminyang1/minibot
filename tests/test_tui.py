from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from textual.widgets import Collapsible, Input, Markdown, Static

from minibot.config import Config
from minibot.runtime.agent_loop import TurnOutcome
from minibot.runtime.approval import ApprovalPolicy, ApprovalRequest
from minibot.runtime.events import RuntimeEventEmitter
from minibot.session import SessionManager
from minibot.tui.app import ApprovalModal, MinibotApp


class _ScriptedAgentSession:
    """Replays a scripted event stream through the caller's handler."""

    def __init__(self, script=None, *, needs_approval: bool = False) -> None:
        self.script = script or []
        self.needs_approval = needs_approval
        self.approval_result: bool | None = None
        self.policy: ApprovalPolicy | None = None
        self.prompts: list[str] = []

    def prompt(self, session_id, user_input, *, run_id=None, event_handler=None):
        self.prompts.append(user_input)
        emitter = RuntimeEventEmitter(
            run_id=run_id or "r_tui",
            session_id=session_id,
            handler=event_handler,
        )
        emitter.emit("run.started", {"input_preview": user_input})
        if self.needs_approval and self.policy is not None:
            request = ApprovalRequest(
                run_id=run_id or "r_tui",
                session_id=session_id,
                approval_id="ap_tui",
                tool_call_id="call_1",
                tool_name="exec",
                args={"command": "ls"},
            )
            self.approval_result = self.policy.request(request)
        for event_type, payload in self.script:
            emitter.emit(event_type, payload)
        emitter.emit("run.completed", {"reply": "done"})
        return TurnOutcome(reply="done")

    def abort(self, run_id):
        del run_id
        return False


class _FailingAgentSession(_ScriptedAgentSession):
    def prompt(self, session_id, user_input, *, run_id=None, event_handler=None):
        emitter = RuntimeEventEmitter(
            run_id=run_id or "r_tui_failure",
            session_id=session_id,
            handler=event_handler,
        )
        emitter.emit("run.started", {"input_preview": user_input})
        emitter.emit(
            "run.failed",
            {"error_type": "RuntimeError", "message": "boom"},
        )
        raise RuntimeError("boom")


def _runtime(tmpdir: str, agent_session) -> types.SimpleNamespace:
    manager = SessionManager(Path(tmpdir))
    policy = ApprovalPolicy(mode="ask")
    agent_session.policy = policy
    return types.SimpleNamespace(
        manager=manager,
        agent_session=agent_session,
        approval_policy=policy,
        config=Config(model="test-model"),
        memory_store=types.SimpleNamespace(list=lambda: []),
        compactor=types.SimpleNamespace(compact_now=lambda session: (False, "noop")),
        context_builder=types.SimpleNamespace(list_available_skills=lambda: []),
        schedule_store=None,
        mcp_host=types.SimpleNamespace(
            summary=lambda: types.SimpleNamespace(
                config_path=None,
                configured_servers=0,
                enabled_servers=0,
                connected_servers=0,
                failed_servers=0,
                tool_count=0,
            ),
            status_snapshot=lambda: [],
        ),
    )


_REPLY_SCRIPT = [
    ("model.request.started", {"iteration": 1, "model": "test-model"}),
    ("message.delta", {"iteration": 1, "channel": "reasoning", "text": "想一想"}),
    ("message.delta", {"iteration": 1, "channel": "text", "text": "你好"}),
    ("message.delta", {"iteration": 1, "channel": "text", "text": ",世界"}),
    (
        "model.request.completed",
        {"iteration": 1, "elapsed_ms": 5, "tool_call_count": 0, "usage": None},
    ),
    ("message.completed", {"iteration": 1, "content": "你好,**世界**"}),
]

_TOOL_SCRIPT = [
    ("model.request.started", {"iteration": 1, "model": "test-model"}),
    (
        "tool_call.started",
        {
            "tool_call_id": "call_1",
            "tool": "read_file",
            "display_name": "read_file",
            "args": {"path": "a.txt"},
        },
    ),
    (
        "tool_call.completed",
        {
            "tool_call_id": "call_1",
            "tool": "read_file",
            "display_name": "read_file",
            "ok": True,
            "summary": "读取成功",
        },
    ),
    ("message.completed", {"iteration": 2, "content": "文件里写着 42"}),
]


class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def _run_prompt(self, app: MinibotApp, pilot, text: str) -> None:
        await pilot.pause()
        prompt = app.query_one(Input)
        prompt.value = text
        await pilot.press("enter")
        await pilot.pause()  # let Input.Submitted dispatch and the worker start
        await app.workers.wait_for_complete()
        await pilot.pause(0.25)  # let the event wakeup and Markdown parse settle

    async def test_reply_streams_and_finalizes_with_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_REPLY_SCRIPT)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await self._run_prompt(app, pilot, "打个招呼")

                self.assertEqual(agent.prompts, ["打个招呼"])
                user_lines = [str(w.content) for w in app.query(".user-line")]
                self.assertTrue(any("打个招呼" in line for line in user_lines))
                replies = list(app.query(Markdown))
                # Exactly one: the streamed widget is finalized in place, not
                # duplicated by message.completed.
                self.assertEqual(len(replies), 1)
                collapsibles = list(app.query(Collapsible))
                self.assertTrue(
                    any("已思考" in (c.title or "") for c in collapsibles)
                )

    async def test_worker_deltas_are_batched_until_one_ui_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession()
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                emitter = RuntimeEventEmitter(
                    run_id="r_batch",
                    session_id=app.session_id,
                )
                for _ in range(100):
                    app._on_event_from_worker(
                        emitter.emit(
                            "message.delta",
                            {"iteration": 1, "channel": "text", "text": "x"},
                        )
                    )

                self.assertIsNone(app._reply_widget)
                app._flush_ui()

                self.assertIsNotNone(app._reply_widget)
                self.assertIsInstance(app._reply_widget, Markdown)
                self.assertEqual(app._reply_buffer, "x" * 100)
                self.assertEqual(app._pending_reply_chunks, [])

    async def test_streaming_markdown_is_finalized_without_widget_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession()
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                emitter = RuntimeEventEmitter(
                    run_id="r_markdown",
                    session_id=app.session_id,
                )
                app._on_event_from_worker(
                    emitter.emit(
                        "message.delta",
                        {
                            "iteration": 1,
                            "channel": "text",
                            "text": "**流式",
                        },
                    )
                )
                app._flush_ui()
                streamed = app._reply_widget
                self.assertIsInstance(streamed, Markdown)

                app._on_event_from_worker(
                    emitter.emit(
                        "message.completed",
                        {"iteration": 1, "content": "**流式 Markdown**"},
                    )
                )
                app._flush_ui()
                await pilot.pause(0.1)

                replies = list(app.query(Markdown))
                self.assertEqual(len(replies), 1)
                self.assertIs(replies[0], streamed)
                self.assertIsNone(app._reply_widget)

    async def test_worker_event_wakes_ui_without_polling_timer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession()
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                emitter = RuntimeEventEmitter(
                    run_id="r_wakeup",
                    session_id=app.session_id,
                )
                app._on_event_from_worker(
                    emitter.emit(
                        "message.delta",
                        {"iteration": 1, "channel": "text", "text": "now"},
                    )
                )

                await pilot.pause(0.05)

                self.assertEqual(app._reply_buffer, "now")
                self.assertIsInstance(app._reply_widget, Markdown)

    async def test_streaming_markdown_drops_superseded_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession()
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                emitter = RuntimeEventEmitter(
                    run_id="r_coalesce",
                    session_id=app.session_id,
                )
                for index in range(100):
                    app._on_event_from_worker(
                        emitter.emit(
                            "message.delta",
                            {
                                "iteration": 1,
                                "channel": "text",
                                "text": str(index % 10),
                            },
                        )
                    )
                    app._flush_ui()

                streamed = app._reply_widget
                self.assertIsNotNone(streamed)
                self.assertEqual(streamed._pending_source, app._reply_buffer)
                self.assertIsNotNone(streamed._update_task)

                await pilot.pause(0.2)

                self.assertEqual(streamed._rendered_source, app._reply_buffer)
                self.assertIsNone(streamed._pending_source)

    async def test_tool_call_line_updates_on_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_TOOL_SCRIPT)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await self._run_prompt(app, pilot, "读文件")

                collapsibles = list(app.query(Collapsible))
                titles = [c.title or "" for c in collapsibles]
                self.assertTrue(
                    any("✓ read_file" in t and "读取成功" in t for t in titles),
                    titles,
                )

    async def test_failed_run_is_rendered_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = MinibotApp(_runtime(tmpdir, _FailingAgentSession()))
            async with app.run_test() as pilot:
                await self._run_prompt(app, pilot, "fail")

                failures = [
                    str(widget.content)
                    for widget in app.query(".error-line")
                    if "运行失败" in str(widget.content)
                ]
                self.assertEqual(len(failures), 1, failures)

    async def test_approval_modal_approve_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(needs_approval=True)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one(Input)
                prompt.value = "跑个命令"
                await pilot.press("enter")
                await pilot.pause()

                for _ in range(50):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, ApprovalModal):
                        break
                self.assertIsInstance(app.screen, ApprovalModal)
                await pilot.press("y")
                await app.workers.wait_for_complete()

                self.assertTrue(agent.approval_result)

    async def test_approval_modal_deny_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(needs_approval=True)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                app.query_one(Input).value = "跑个命令"
                await pilot.press("enter")
                await pilot.pause()
                for _ in range(50):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, ApprovalModal):
                        break
                await pilot.press("n")
                await app.workers.wait_for_complete()

                self.assertFalse(agent.approval_result)

    async def test_slash_command_renders_notice_without_running_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_REPLY_SCRIPT)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                app.query_one(Input).value = "/help"
                await pilot.press("enter")
                await pilot.pause()

                self.assertEqual(agent.prompts, [])
                system_lines = [
                    str(w.content) for w in app.query(".system-line")
                ]
                self.assertTrue(any("/sessions" in line for line in system_lines))

    async def test_slash_menu_filters_and_tab_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_REPLY_SCRIPT)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one(Input)
                prompt.value = "/m"
                app._update_slash_menu(prompt.value)

                menu = app.query_one("#slash-menu", Static)
                self.assertTrue(menu.has_class("visible"))
                self.assertIn("/mcp", str(menu.content))

                await pilot.press("tab")
                await pilot.pause()

                self.assertEqual(prompt.value, "/mcp")
                self.assertEqual(agent.prompts, [])

    async def test_enter_selects_partial_slash_command_before_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_REPLY_SCRIPT)
            app = MinibotApp(_runtime(tmpdir, agent))
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one(Input)
                prompt.value = "/he"
                app._update_slash_menu(prompt.value)

                await pilot.press("enter")
                await pilot.pause()

                self.assertEqual(prompt.value, "/help")
                self.assertEqual(agent.prompts, [])

                await pilot.press("enter")
                await pilot.pause()

                system_lines = [str(w.content) for w in app.query(".system-line")]
                self.assertTrue(any("/sessions" in line for line in system_lines))
                self.assertEqual(agent.prompts, [])

    async def test_two_level_choice_menu_commits_and_dispatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_REPLY_SCRIPT)
            runtime = _runtime(tmpdir, agent)
            app = MinibotApp(runtime)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one(Input)
                prompt.value = "/per"
                app._update_slash_menu(prompt.value)

                await pilot.press("tab")
                await pilot.pause()

                menu = app.query_one("#slash-menu", Static)
                self.assertIn("ask", str(menu.content))
                self.assertIn("always", str(menu.content))

                await pilot.press("down", "enter")
                await pilot.pause()
                self.assertEqual(prompt.value, "/permission always")

                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(runtime.approval_policy.mode, "always")
                self.assertEqual(agent.prompts, [])

    async def test_session_menu_uses_full_id_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = _ScriptedAgentSession(_REPLY_SCRIPT)
            runtime = _runtime(tmpdir, agent)
            target_id = "s_target_123456789abcdef"
            runtime.manager.create_session(target_id, title="目标会话")
            runtime.manager.create_current_session("s_current_123456789abcdef")
            app = MinibotApp(runtime)
            async with app.run_test() as pilot:
                await pilot.pause()
                prompt = app.query_one(Input)
                prompt.value = "/res"
                app._update_slash_menu(prompt.value)

                await pilot.press("tab")
                await pilot.pause()
                target_index = next(
                    index
                    for index, option in enumerate(app._param_options)
                    if option.value == target_id
                )
                for _ in range(target_index):
                    await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause()

                self.assertEqual(prompt.value, f"/resume {target_id}")
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app.session_id, target_id)
                self.assertEqual(agent.prompts, [])


if __name__ == "__main__":
    unittest.main()
