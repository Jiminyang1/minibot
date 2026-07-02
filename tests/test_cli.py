from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from minibot import cli
from minibot.cli import CliRenderer, install_slash_completion, main, read_user_input, run_repl
from minibot.config import Config
from minibot.runtime.events import RuntimeEventEmitter
from minibot.runtime.agent_loop import TurnOutcome
from minibot.runtime.approval import ApprovalPolicy, ApprovalRequest
from minibot.session import SessionManager


class _AgentSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def prompt(self, session_id, user_input, *, run_id=None, event_handler=None):
        self.calls.append((session_id, user_input))
        emitter = RuntimeEventEmitter(
            run_id=run_id or "r_test",
            session_id=session_id,
            handler=event_handler,
        )
        emitter.emit("context.usage", {"current_tokens": 10, "budget": 100})
        emitter.emit("model.request.started", {"iteration": 1, "model": "test-model"})
        emitter.emit(
            "tool_call.started",
            {
                "tool_call_id": "call_1",
                "tool": "read_file",
                "display_name": "read_file",
                "args": {"path": "a.txt"},
            },
        )
        emitter.emit(
            "tool_call.completed",
            {
                "tool_call_id": "call_1",
                "tool": "read_file",
                "display_name": "read_file",
                "ok": True,
                "summary": "done",
            },
        )
        emitter.emit("message.completed", {"iteration": 1, "content": f"echo: {user_input}"})
        return TurnOutcome(reply=f"echo: {user_input}", did_compact=False)

    def abort(self, run_id):
        del run_id
        return False


class _Compactor:
    def compact_now(self, session):
        return True, f"compact {session.session_id}"


class _ContextBuilder:
    def list_available_skills(self):
        return []


class _Memory:
    def list(self):
        return []

    def clear(self):
        return 0

    def delete(self, memory_id):
        del memory_id
        return False


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class CliTests(unittest.TestCase):
    def _runtime(self, manager: SessionManager):
        return types.SimpleNamespace(
            config=Config(model="test-model"),
            manager=manager,
            memory_store=_Memory(),
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
            compactor=_Compactor(),
            context_builder=_ContextBuilder(),
            agent_session=_AgentSession(),
            approval_policy=ApprovalPolicy(mode="ask"),
        )

    def test_main_rejects_positional_prompt(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["hello"])

        self.assertEqual(code, 2)
        self.assertIn("只支持交互模式", stderr.getvalue())

    def test_main_help_lists_repl_commands(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        output = stdout.getvalue()
        self.assertIn("REPL commands", output)
        self.assertIn("/sessions", output)

    def test_renderer_no_color_outputs_plain_text(self) -> None:
        out = io.StringIO()
        renderer = CliRenderer(no_color=True, stdout=out)

        renderer.success("ok")

        self.assertEqual(out.getvalue(), "  ok\n")
        self.assertNotIn("\x1b", out.getvalue())

    def test_renderer_default_hides_model_and_context_events(self) -> None:
        out = io.StringIO()
        renderer = CliRenderer(no_color=True, stdout=out)
        emitter = RuntimeEventEmitter(run_id="r_test", session_id="s_test")

        renderer.render_event(emitter.emit("context.usage", {"current_tokens": 10, "budget": 100}))
        renderer.render_event(emitter.emit("model.request.started", {"iteration": 1, "model": "m"}))
        renderer.render_event(
            emitter.emit(
                "tool_call.started",
                {"tool": "read_file", "display_name": "read_file", "args": {"path": "a"}},
            )
        )

        output = out.getvalue()
        self.assertNotIn("context:", output)
        self.assertNotIn("请求模型", output)
        self.assertIn("工具: read_file", output)
        self.assertNotIn('"path"', output)

    def test_renderer_status_line_is_disabled_for_non_tty(self) -> None:
        out = io.StringIO()
        renderer = CliRenderer(no_color=True, stdout=out)

        renderer.start_status("thinking")
        renderer.update_status("running tool")
        renderer.clear_status()

        self.assertEqual(out.getvalue(), "")

    def test_renderer_status_line_updates_from_events_for_tty(self) -> None:
        out = _TtyStringIO()
        renderer = CliRenderer(no_color=True, stdout=out)
        emitter = RuntimeEventEmitter(run_id="r_test", session_id="s_test")

        renderer.start_status("thinking")
        renderer.render_event(
            emitter.emit(
                "tool_call.started",
                {"tool": "read_file", "display_name": "read_file", "args": {}},
            )
        )

        self.assertTrue(renderer._status_line._active)
        self.assertEqual(renderer._status_line._message, "running read_file")
        self.assertIn("工具: read_file", out.getvalue())
        renderer.stop_status()

    def test_renderer_verbose_shows_context_model_and_tool_args(self) -> None:
        out = io.StringIO()
        renderer = CliRenderer(verbose=True, no_color=True, stdout=out)
        emitter = RuntimeEventEmitter(run_id="r_test", session_id="s_test")

        renderer.render_event(emitter.emit("context.usage", {"current_tokens": 10, "budget": 100}))
        renderer.render_event(emitter.emit("model.request.started", {"iteration": 1, "model": "m"}))
        renderer.render_event(
            emitter.emit(
                "tool_call.started",
                {"tool": "read_file", "display_name": "read_file", "args": {"path": "a"}},
            )
        )

        output = out.getvalue()
        self.assertIn("context: 10/100", output)
        self.assertIn("第 1 轮请求模型", output)
        self.assertIn('"path": "a"', output)

    def test_read_user_input_merges_multiline_paste(self) -> None:
        out = io.StringIO()
        renderer = CliRenderer(no_color=True, stdout=out)
        with (
            patch("builtins.input", return_value="line 1"),
            patch("minibot.cli._drain_pending_stdin", return_value="line 2\nline 3\n"),
        ):
            value = read_user_input(renderer)

        self.assertEqual(value, "line 1\nline 2\nline 3")
        self.assertIn("检测到 3 行粘贴内容", out.getvalue())

    def test_slash_completion_matches_repl_commands_and_restores_state(self) -> None:
        class _FakeReadline:
            def __init__(self) -> None:
                self.completer = "old"
                self.delims = "old-delims"
                self.bindings: list[str] = []

            def get_completer(self):
                return self.completer

            def set_completer(self, completer):
                self.completer = completer

            def get_completer_delims(self):
                return self.delims

            def set_completer_delims(self, delims):
                self.delims = delims

            def parse_and_bind(self, binding):
                self.bindings.append(binding)

            def get_line_buffer(self):
                return "/per"

            def get_begidx(self):
                return 0

        fake = _FakeReadline()
        fake.__doc__ = "GNU readline"
        with patch.object(cli, "readline", fake):
            state = install_slash_completion()
            self.assertEqual(fake.completer("/per", 0), "/permission ")
            self.assertIsNone(fake.completer("/per", 1))
            self.assertIn("tab: complete", fake.bindings)
            state.restore()

        self.assertEqual(fake.completer, "old")
        self.assertEqual(fake.delims, "old-delims")

    def test_slash_completion_uses_libedit_binding_on_macos(self) -> None:
        class _FakeReadline:
            __doc__ = "EditLine wrapper"

            def __init__(self) -> None:
                self.completer = None
                self.delims = ""
                self.bindings: list[str] = []

            def get_completer(self):
                return None

            def set_completer(self, completer):
                self.completer = completer

            def get_completer_delims(self):
                return ""

            def set_completer_delims(self, delims):
                self.delims = delims

            def parse_and_bind(self, binding):
                self.bindings.append(binding)

        fake = _FakeReadline()
        with patch.object(cli, "readline", fake):
            install_slash_completion()

        self.assertIn("bind ^I rl_complete", fake.bindings)

    def test_prompt_approval_reprompts_and_uses_safe_default(self) -> None:
        out = io.StringIO()
        renderer = CliRenderer(no_color=True, stdout=out)
        request = ApprovalRequest(
            run_id="r_test",
            session_id="s_test",
            approval_id="ap_test",
            tool_call_id="call_1",
            tool_name="exec",
            args={"cmd": "ls"},
        )

        with patch("builtins.input", side_effect=["bad", ""]):
            approved = renderer.prompt_approval(request, None)

        self.assertFalse(approved)
        self.assertIn("请输入 y 或 n", out.getvalue())

    def test_repl_routes_normal_messages_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            runtime = self._runtime(manager)
            out = io.StringIO()
            renderer = CliRenderer(no_color=True, stdout=out)

            with patch("minibot.cli.read_user_input", side_effect=["/permission always", "hello", "exit"]):
                run_repl(runtime, renderer)

            self.assertEqual(runtime.approval_policy.mode, "always")
            self.assertEqual(len(runtime.agent_session.calls), 1)
            self.assertEqual(runtime.agent_session.calls[0][1], "hello")
            output = out.getvalue()
            self.assertIn("MiniBot", output)
            self.assertIn("MiniBot › echo: hello", output)


if __name__ == "__main__":
    unittest.main()
