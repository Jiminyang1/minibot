from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest

from minibot.config import Config
from minibot.interaction.commands import CommandContext, dispatch_command
from minibot.mcp_host.models import MCPHostSummary, MCPServerStatus
from minibot.runtime.approval import ApprovalPolicy
from minibot.session import SessionManager


class _Memory:
    def __init__(self) -> None:
        self.items = []
        self.deleted = []

    def list(self):
        return list(self.items)

    def clear(self):
        count = len(self.items)
        self.items = []
        return count

    def delete(self, memory_id):
        self.deleted.append(memory_id)
        return memory_id == "m_1"


class _Compactor:
    def __init__(self) -> None:
        self.compacted = False

    def compact_now(self, session):
        self.compacted = True
        return True, f"compact {session.session_id}"


class _ContextBuilder:
    def list_available_skills(self):
        return [("mail", "Mail helper", ("read_mail",))]


class _MCPHost:
    def summary(self):
        return MCPHostSummary(
            config_path="/tmp/mcp.json",
            configured_servers=1,
            enabled_servers=1,
            connected_servers=1,
            failed_servers=0,
            tool_count=1,
        )

    def status_snapshot(self):
        return [
            MCPServerStatus(
                name="sqlite",
                transport="stdio",
                enabled=True,
                trusted=True,
                connected=True,
                tool_count=1,
                tool_names=("query",),
            )
        ]


class InteractionCommandTests(unittest.TestCase):
    def _context(self, manager):
        return CommandContext(
            sessions=manager,
            compactor=_Compactor(),  # type: ignore[arg-type]
            context_builder=_ContextBuilder(),  # type: ignore[arg-type]
            memory_store=_Memory(),  # type: ignore[arg-type]
            approval_policy=ApprovalPolicy(mode="ask"),
            mcp_host=_MCPHost(),  # type: ignore[arg-type]
            config=Config(model="test-model"),
        )

    def test_startup_session_prefers_current_then_latest_then_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            created, resumed = manager.startup_session()
            self.assertFalse(resumed)
            self.assertEqual(manager.get_current_session_id(), created.session_id)

            loaded, resumed = manager.startup_session()
            self.assertTrue(resumed)
            self.assertEqual(loaded.session_id, created.session_id)

    def test_session_commands_switch_create_resume_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            first = manager.create_session("s_first")
            manager.set_current_session(first.session_id)
            ctx = self._context(manager)

            new_result = dispatch_command("/new", first.session_id, ctx)
            self.assertTrue(new_result.handled)
            self.assertNotEqual(new_result.current_session_id, first.session_id)

            resume_result = dispatch_command("/resume s_first", new_result.current_session_id or "", ctx)
            self.assertEqual(resume_result.current_session_id, "s_first")

            delete_result = dispatch_command("/delete current", "s_first", ctx)
            self.assertNotEqual(delete_result.current_session_id, "s_first")
            self.assertIsNone(manager.load("s_first"))

    def test_existing_slash_commands_return_structured_notices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.create_session("s_test")
            ctx = self._context(manager)

            for command in [
                "/help",
                "/sessions",
                "/compact",
                "/mcp",
                "/mcp tools",
                "/skills",
                "/permission",
                "/permission always",
                "/config",
                "/memory",
                "/memory clear",
                "/memory forget m_1",
            ]:
                result = dispatch_command(command, session.session_id, ctx)
                self.assertTrue(result.handled, command)
                self.assertTrue(result.notices, command)

            self.assertEqual(ctx.approval_policy.mode, "always")

    def test_normal_prompt_is_not_handled_and_exit_is_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(Path(tmpdir))
            session = manager.create_session("s_test")
            ctx = self._context(manager)

            self.assertFalse(dispatch_command("hello", session.session_id, ctx).handled)
            exit_result = dispatch_command("exit", session.session_id, ctx)
            self.assertTrue(exit_result.handled)
            self.assertTrue(exit_result.should_exit)


if __name__ == "__main__":
    unittest.main()
