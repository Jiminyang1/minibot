from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import resolve_state_home
from minibot.migrate import migrate_workspace_state
from minibot.session import MessageEvent, Session, SessionManager
from minibot.tools.base import ToolExecutionContext
from minibot.tools.search_history import SearchHistoryTool


def _message(role: str, content: str, created_at: str) -> MessageEvent:
    return MessageEvent(
        id=f"e_{abs(hash((role, content, created_at))) % 10**10}",
        role=role,
        content=content,
        created_at=created_at,
    )


def _make_session(
    manager: SessionManager,
    session_id: str,
    messages: list[tuple[str, str, str]],
    *,
    workspace: str | None = None,
    summary: str | None = None,
) -> Session:
    session = manager.create_session(session_id)
    session.workspace = workspace
    for role, content, created_at in messages:
        session.add_message(_message(role, content, created_at))
    if summary is not None:
        session.compact_with_summary(
            summary,
            first_kept_entry_id=session.entries[-1].id if session.entries else None,
        )
    manager.save(session)
    return session


class StateHomeTests(unittest.TestCase):
    def test_env_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MINIBOT_HOME": tmpdir}):
                self.assertEqual(resolve_state_home(), Path(tmpdir).resolve())

    def test_default_is_global_dot_minibot(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIBOT_HOME", None)
            self.assertEqual(
                resolve_state_home(), (Path.home() / ".minibot").resolve()
            )

    def test_sessions_are_stamped_with_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            workspace = Path(tmpdir) / "project"
            workspace.mkdir()
            manager = SessionManager(home, default_workspace=workspace)

            created = manager.create_session("s_test")
            reloaded = manager.load("s_test")

            assert reloaded is not None
            self.assertEqual(created.workspace, str(workspace.resolve()))
            self.assertEqual(reloaded.workspace, str(workspace.resolve()))
            listed = manager.list_sessions()[0]
            self.assertEqual(listed.workspace, str(workspace.resolve()))


class MigrationTests(unittest.TestCase):
    def _seed_workspace(self, workspace: Path, session_id: str, marker: str) -> None:
        manager = SessionManager(workspace / ".minibot")
        session = manager.create_session(session_id)
        session.add_message(
            _message("user", marker, "2026-01-01T00:00:00Z")
        )
        manager.save(session)
        (workspace / ".minibot" / "runs.jsonl").write_text(
            json.dumps({"run_id": f"r_{marker}"}) + "\n", encoding="utf-8"
        )

    def test_collision_renames_and_merges_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            ws1, ws2 = root / "ws1", root / "ws2"
            self._seed_workspace(ws1, "s_dup", "第一处")
            self._seed_workspace(ws2, "s_dup", "第二处")

            report1 = migrate_workspace_state(ws1, home)
            report2 = migrate_workspace_state(ws2, home)

            self.assertEqual(report1.moved_sessions, ["s_dup"])
            self.assertEqual(report2.renamed_sessions, [("s_dup", "s_dup_m1")])
            manager = SessionManager(home)
            ids = sorted(s.session_id for s in manager.list_sessions())
            self.assertEqual(ids, ["s_dup", "s_dup_m1"])
            renamed = manager.load("s_dup_m1")
            assert renamed is not None
            self.assertEqual(renamed.messages[0].content, "第二处")
            self.assertEqual(renamed.workspace, str(ws2.resolve()))
            runs = (home / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(runs), 2)

    def test_missing_source_is_reported_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = migrate_workspace_state(root / "nowhere", root / "home")
            self.assertIn("目录不存在", report.skipped[0])


class SearchHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manager = SessionManager(Path(self._tmp.name))
        self.tool = SearchHistoryTool(self.manager)
        self.context = ToolExecutionContext(session_id="s_current")
        _make_session(
            self.manager,
            "s_current",
            [("user", "现在正在聊的餐厅话题", "2026-07-05T10:00:00Z")],
        )
        _make_session(
            self.manager,
            "s_food",
            [
                ("user", "帮我找一家好吃的川菜餐厅", "2026-07-01T10:00:00Z"),
                ("assistant", "推荐眉州东坡，川菜口碑不错", "2026-07-01T10:01:00Z"),
            ],
            workspace="/Users/jimin/Desktop/Projects/minibot",
        )
        _make_session(
            self.manager,
            "s_old",
            [("user", "很久以前聊过的餐厅", "2026-01-01T10:00:00Z")],
        )
        _make_session(
            self.manager,
            "s_compacted",
            [("user", "无关内容", "2026-06-20T10:00:00Z")],
            summary="用户决定周五去吃眉州东坡的川菜",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, **kwargs):
        return self.tool.execute(context=self.context, **kwargs)

    def test_finds_matches_and_excludes_current_session(self) -> None:
        output = self._run(query="餐厅")

        self.assertTrue(output.ok)
        session_ids = {m["session_id"] for m in output.data["matches"]}
        self.assertIn("s_food", session_ids)
        self.assertIn("s_old", session_ids)
        self.assertNotIn("s_current", session_ids)

    def test_terms_are_and_semantics(self) -> None:
        output = self._run(query="川菜 眉州东坡")

        kinds = {(m["session_id"], m["kind"]) for m in output.data["matches"]}
        self.assertIn(("s_food", "assistant"), kinds)
        self.assertNotIn(("s_old", "user"), kinds)

    def test_compaction_summaries_are_searched(self) -> None:
        output = self._run(query="周五 川菜")

        matches = output.data["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["session_id"], "s_compacted")
        self.assertEqual(matches[0]["kind"], "summary")

    def test_days_filter_drops_old_sessions(self) -> None:
        output = self._run(query="餐厅", days=30)

        session_ids = {m["session_id"] for m in output.data["matches"]}
        self.assertIn("s_food", session_ids)
        self.assertNotIn("s_old", session_ids)

    def test_workspace_filter(self) -> None:
        output = self._run(query="餐厅", workspace="minibot")

        session_ids = {m["session_id"] for m in output.data["matches"]}
        self.assertEqual(session_ids, {"s_food"})

    def test_limit_and_ordering_newest_first(self) -> None:
        output = self._run(query="餐厅", limit=1)

        matches = output.data["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["session_id"], "s_food")
        self.assertGreater(output.data["total_matches"], 1)

    def test_no_match_is_a_helpful_success(self) -> None:
        output = self._run(query="不存在的关键词xyz")

        self.assertTrue(output.ok)
        self.assertEqual(output.data["matches"], [])
        self.assertIn("未找到匹配", output.summary)

    def test_empty_query_is_rejected(self) -> None:
        output = self._run(query="   ")

        self.assertFalse(output.ok)
        self.assertEqual(output.code, "invalid_args")

    def test_snippet_windows_around_first_term(self) -> None:
        long_content = "开头" + "x" * 500 + " 目标关键词 " + "y" * 500
        _make_session(
            self.manager,
            "s_long",
            [("user", long_content, "2026-07-02T10:00:00Z")],
        )

        output = self._run(query="目标关键词")

        match = next(
            m for m in output.data["matches"] if m["session_id"] == "s_long"
        )
        self.assertIn("目标关键词", match["snippet"])
        self.assertLess(len(match["snippet"]), 300)
        self.assertTrue(match["snippet"].startswith("…"))


if __name__ == "__main__":
    unittest.main()
