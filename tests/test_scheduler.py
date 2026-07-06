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
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._error = error

    def prompt(self, session_id, user_input, **kwargs):
        self.calls.append((session_id, user_input))
        if self._error is not None:
            raise self._error
        return TurnOutcome(reply="简报生成完毕")


class SchedulerTickTests(unittest.TestCase):
    def _scheduler(self, tmpdir: str, *, error: Exception | None = None):
        store = ScheduleStore(Path(tmpdir) / "home")
        manager = SessionManager(Path(tmpdir) / "home")
        agent = _FakeAgentSession(error=error)
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


if __name__ == "__main__":
    unittest.main()
