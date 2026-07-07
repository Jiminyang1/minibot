from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.agent_loop import TurnOutcome
from minibot.schedule_store import (
    ScheduleStore,
    cron_next,
    parse_cron,
    parse_local_time,
)
from minibot.scheduler import Scheduler
from minibot.session import SessionManager
from minibot.tools.base import ToolExecutionContext
from minibot.tools.schedule_tools import (
    CancelScheduledTaskTool,
    ListScheduledTasksTool,
    ScheduleTaskTool,
)


def _local(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone()


class CronTests(unittest.TestCase):
    def test_daily_at_eight(self) -> None:
        after = _local("2026-07-06T09:30:00")
        self.assertEqual(
            cron_next("0 8 * * *", after),
            _local("2026-07-07T08:00:00"),
        )

    def test_step_every_fifteen_minutes(self) -> None:
        after = _local("2026-07-06T10:07:00")
        self.assertEqual(
            cron_next("*/15 * * * *", after),
            _local("2026-07-06T10:15:00"),
        )

    def test_weekly_friday(self) -> None:
        # 2026-07-06 is a Monday; next Friday 18:00 is 07-10.
        after = _local("2026-07-06T00:00:00")
        self.assertEqual(
            cron_next("0 18 * * 5", after),
            _local("2026-07-10T18:00:00"),
        )

    def test_sunday_seven_equals_zero(self) -> None:
        parsed = parse_cron("0 8 * * 7")
        self.assertIn(0, parsed[4])
        self.assertNotIn(7, parsed[4])

    def test_dom_dow_or_semantics(self) -> None:
        # Both restricted: fires on the 15th OR on Fridays (vixie cron).
        after = _local("2026-07-06T00:00:00")
        first = cron_next("0 8 15 * 5", after)
        self.assertEqual(first, _local("2026-07-10T08:00:00"))  # Friday
        second = cron_next("0 8 15 * 5", first)
        self.assertEqual(second, _local("2026-07-15T08:00:00"))  # the 15th

    def test_invalid_expressions_raise(self) -> None:
        for expr in ["0 8 * *", "61 * * * *", "* * * * mon", "*/0 * * * *"]:
            with self.assertRaises(ValueError):
                parse_cron(expr)


class ScheduleStoreTests(unittest.TestCase):
    def test_roundtrip_and_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ScheduleStore(Path(tmpdir))
            task = store.add(
                title="简报", prompt="生成简报", kind="cron", expr="0 8 * * *"
            )

            listed = store.list()
            self.assertEqual([t.id for t in listed], [task.id])
            self.assertEqual(listed[0].title, "简报")

            updated = store.update(task, last_status="success")
            self.assertEqual(store.get(task.id).last_status, "success")
            self.assertEqual(updated.last_status, "success")

            self.assertTrue(store.remove(task.id))
            self.assertFalse(store.remove(task.id))
            self.assertEqual(store.list(), [])

    def test_add_validates_expressions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ScheduleStore(Path(tmpdir))
            with self.assertRaises(ValueError):
                store.add(title="x", prompt="p", kind="cron", expr="not cron")
            with self.assertRaises(ValueError):
                store.add(title="x", prompt="p", kind="once", expr="not a time")

    def test_once_next_run_then_never_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ScheduleStore(Path(tmpdir))
            task = store.add(
                title="提醒", prompt="提醒我", kind="once", expr="2026-07-07T09:00:00"
            )
            self.assertEqual(task.next_run(), parse_local_time("2026-07-07T09:00:00"))
            fired = store.update(task, last_run_at="2026-07-07T09:00:05", enabled=False)
            self.assertIsNone(fired.next_run())


class _FakeAgentSession:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        reply: str = "简报生成完毕",
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._error = error
        self.reply = reply

    def prompt(self, session_id, user_input, **kwargs):
        self.calls.append((session_id, user_input))
        if self._error is not None:
            raise self._error
        return TurnOutcome(reply=self.reply)


class SchedulerTickTests(unittest.TestCase):
    def _scheduler(
        self,
        tmpdir: str,
        *,
        error: Exception | None = None,
        reply: str = "简报生成完毕",
    ):
        store = ScheduleStore(Path(tmpdir) / "home")
        manager = SessionManager(Path(tmpdir) / "home")
        agent = _FakeAgentSession(error=error, reply=reply)
        notifications: list[tuple[str, str]] = []
        scheduler = Scheduler(
            store=store,
            agent_session=agent,
            session_manager=manager,
            notifier=lambda title, body: notifications.append((title, body)),
        )
        return scheduler, store, manager, agent, notifications

    def test_due_cron_task_fires_once_within_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, store, manager, agent, notes = self._scheduler(tmpdir)
            task = store.add(
                title="每日简报", prompt="生成今天的简报", kind="cron", expr="0 8 * * *"
            )
            due = task.next_run()
            assert due is not None

            fired = scheduler.tick(due + timedelta(minutes=5))

            self.assertEqual(fired, [task.id])
            self.assertEqual(len(agent.calls), 1)
            self.assertIn("生成今天的简报", agent.calls[0][1])
            self.assertIn("无人值守", agent.calls[0][1])
            self.assertEqual(store.get(task.id).last_status, "success")
            self.assertEqual(len(notes), 1)
            self.assertIn("每日简报", notes[0][0])
            titles = [s.title for s in manager.list_sessions()]
            self.assertTrue(any(t.startswith("[定时] 每日简报") for t in titles))

            # Same moment again: anchor advanced, no double fire.
            self.assertEqual(scheduler.tick(due + timedelta(minutes=6)), [])

    def test_not_due_does_not_fire(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, store, _, agent, _ = self._scheduler(tmpdir)
            task = store.add(
                title="每日简报", prompt="p", kind="cron", expr="0 8 * * *"
            )
            due = task.next_run()
            assert due is not None

            self.assertEqual(scheduler.tick(due - timedelta(minutes=5)), [])
            self.assertEqual(agent.calls, [])

    def test_miss_beyond_grace_is_recorded_not_fired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, store, _, agent, notes = self._scheduler(tmpdir)
            task = store.add(
                title="每日简报", prompt="p", kind="cron", expr="0 8 * * *"
            )
            due = task.next_run()
            assert due is not None

            fired = scheduler.tick(due + timedelta(hours=3))

            self.assertEqual(fired, [])
            self.assertEqual(agent.calls, [])
            self.assertEqual(store.get(task.id).last_status, "missed")
            self.assertEqual(notes, [])
            # Anchor advanced: the same stale moment stays quiet.
            self.assertEqual(scheduler.tick(due + timedelta(hours=3, minutes=1)), [])

    def test_once_task_fires_then_disables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, store, _, agent, _ = self._scheduler(tmpdir)
            task = store.add(
                title="提醒", prompt="提醒我交周报", kind="once",
                expr="2026-07-07T09:00:00",
            )
            due = parse_local_time("2026-07-07T09:00:00")

            fired = scheduler.tick(due + timedelta(minutes=1))

            self.assertEqual(fired, [task.id])
            stored = store.get(task.id)
            self.assertFalse(stored.enabled)
            self.assertEqual(scheduler.tick(due + timedelta(minutes=2)), [])
            self.assertEqual(len(agent.calls), 1)

    def test_failed_run_records_status_and_notifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, store, _, _, notes = self._scheduler(
                tmpdir, error=RuntimeError("llm down")
            )
            task = store.add(
                title="每日简报", prompt="p", kind="cron", expr="0 8 * * *"
            )
            due = task.next_run()
            assert due is not None

            scheduler.tick(due + timedelta(minutes=1))

            self.assertEqual(store.get(task.id).last_status, "failed: RuntimeError")
            self.assertEqual(len(notes), 1)
            self.assertIn("任务失败", notes[0][0])


class HeartbeatTests(unittest.TestCase):
    def _fire_once(self, tmpdir: str, *, reply: str, checklist: str | None = None):
        store = ScheduleStore(Path(tmpdir) / "home")
        manager = SessionManager(Path(tmpdir) / "home")
        agent = _FakeAgentSession(reply=reply)
        notes: list[tuple[str, str]] = []
        scheduler = Scheduler(
            store=store,
            agent_session=agent,
            session_manager=manager,
            notifier=lambda title, body: notes.append((title, body)),
        )
        if checklist is not None:
            (store.state_dir / "HEARTBEAT.md").write_text(
                checklist, encoding="utf-8"
            )
        task = store.add(
            title="巡逻", prompt="", kind="heartbeat", expr="*/30 * * * *"
        )
        due = task.next_run()
        assert due is not None
        fired = scheduler.tick(due + timedelta(minutes=1))
        return scheduler, store, manager, agent, notes, task, fired, due

    def test_quiet_heartbeat_does_not_notify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, store, _, agent, notes, task, fired, _ = self._fire_once(
                tmpdir,
                reply="一切正常,HEARTBEAT_OK",
                checklist="- 检查邮件\n",
            )

            self.assertEqual(fired, [task.id])
            self.assertEqual(notes, [])
            self.assertEqual(store.get(task.id).last_status, "ok-quiet")
            self.assertIn("检查邮件", agent.calls[0][1])
            self.assertIn("HEARTBEAT_OK", agent.calls[0][1])

    def test_attention_heartbeat_notifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, store, _, _, notes, task, _, _ = self._fire_once(
                tmpdir,
                reply="你有一封重要邮件需要回复",
                checklist="- 检查邮件\n",
            )

            self.assertEqual(store.get(task.id).last_status, "attention")
            self.assertEqual(len(notes), 1)
            self.assertIn("心跳", notes[0][0])
            self.assertIn("重要邮件", notes[0][1])

    def test_heartbeat_reuses_one_persistent_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler, store, manager, agent, _, task, _, due = self._fire_once(
                tmpdir,
                reply="HEARTBEAT_OK",
                checklist="- 检查邮件\n",
            )
            stored = store.get(task.id)
            self.assertIsNotNone(stored.session_id)

            next_due = stored.next_run()
            assert next_due is not None
            scheduler.tick(next_due + timedelta(minutes=1))

            self.assertEqual(len(agent.calls), 2)
            self.assertEqual(agent.calls[0][0], agent.calls[1][0])
            heartbeat_sessions = [
                s for s in manager.list_sessions() if s.title.startswith("[心跳]")
            ]
            self.assertEqual(len(heartbeat_sessions), 1)
            # Heartbeats never disable themselves.
            self.assertTrue(store.get(task.id).enabled)

    def test_missing_checklist_creates_template_and_stays_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, store, _, agent, notes, task, _, _ = self._fire_once(
                tmpdir, reply="HEARTBEAT_OK"
            )

            template = (store.state_dir / "HEARTBEAT.md").read_text(encoding="utf-8")
            self.assertIn("巡逻清单", template)
            self.assertIn("清单为空", agent.calls[0][1])
            self.assertEqual(notes, [])

    def test_comment_lines_are_stripped_from_checklist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, agent, _, _, _, _ = self._fire_once(
                tmpdir,
                reply="HEARTBEAT_OK",
                checklist="# 注释行\n- 真实检查项\n\n# 又一条注释\n",
            )

            prompt = agent.calls[0][1]
            self.assertIn("真实检查项", prompt)
            self.assertNotIn("注释行", prompt)


class ScheduleToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ScheduleStore(Path(self._tmp.name))
        self.context = ToolExecutionContext(session_id="s_test")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_schedule_task_requires_exactly_one_time_form(self) -> None:
        tool = ScheduleTaskTool(self.store)
        self.assertTrue(tool.requires_approval)

        both = tool.execute(
            context=self.context, title="x", prompt="p",
            cron="0 8 * * *", at="2026-07-07T09:00",
        )
        neither = tool.execute(context=self.context, title="x", prompt="p")

        self.assertEqual(both.code, "invalid_args")
        self.assertEqual(neither.code, "invalid_args")
        self.assertEqual(self.store.list(), [])

    def test_schedule_list_cancel_flow(self) -> None:
        schedule = ScheduleTaskTool(self.store)
        listing = ListScheduledTasksTool(self.store)
        cancel = CancelScheduledTaskTool(self.store)

        created = schedule.execute(
            context=self.context, title="每日简报",
            prompt="汇总邮件和日程", cron="0 8 * * *",
        )
        self.assertTrue(created.ok)
        task_id = created.data["task_id"]
        self.assertIsNotNone(created.data["next_run"])

        listed = listing.execute(context=self.context)
        self.assertEqual(len(listed.data["tasks"]), 1)
        self.assertEqual(listed.data["tasks"][0]["task_id"], task_id)

        gone = cancel.execute(context=self.context, task_id=task_id)
        self.assertTrue(gone.ok)
        self.assertEqual(
            cancel.execute(context=self.context, task_id=task_id).code,
            "not_found",
        )

    def test_invalid_cron_is_rejected(self) -> None:
        tool = ScheduleTaskTool(self.store)
        output = tool.execute(
            context=self.context, title="x", prompt="p", cron="nope"
        )
        self.assertEqual(output.code, "invalid_args")

    def test_heartbeat_creation_requires_cron_and_allows_empty_prompt(self) -> None:
        tool = ScheduleTaskTool(self.store)

        with_at = tool.execute(
            context=self.context, title="巡逻", prompt="",
            at="2026-07-08T09:00", heartbeat=True,
        )
        self.assertEqual(with_at.code, "invalid_args")

        created = tool.execute(
            context=self.context, title="巡逻", prompt="",
            cron="*/30 * * * *", heartbeat=True,
        )
        self.assertTrue(created.ok)
        self.assertEqual(created.data["kind"], "heartbeat")
        self.assertEqual(self.store.list()[0].kind, "heartbeat")


if __name__ == "__main__":
    unittest.main()
