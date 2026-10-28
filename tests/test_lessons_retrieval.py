"""Lessons retrieval and its injection into model requests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.bootstrap import _build_lesson_retriever
from minibot.config import Config
from minibot.lessons.retrieve import LessonRetriever
from minibot.lessons.review import approve, retract
from minibot.lessons.store import Evidence, LessonStore
from minibot.llm import LLMClient, LLMResponse
from minibot.runtime.context_builder import ContextBuilder
from minibot.runtime.events import RuntimeEvent
from minibot.session import MessageEvent, SessionContextProjector, SessionEntry
from minibot.tools.registry import ToolRegistry

from loop_harness import build_loop, run_turn
from test_lessons import HashEmbedder

_FILE_TRIGGER = "用户要求读取文件内容"
_MAIL_TRIGGER = "用户要求发送邮件"
_CAL_TRIGGER = "用户要求创建日历日程"


def _approved_store(path: Path, *rules: tuple[str, str]) -> LessonStore:
    store = LessonStore(path)
    for trigger, guidance in rules:
        approve(store, store.add_draft(trigger, guidance).id)
    return store


class LessonRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _approved_store(
            Path(self._tmp.name) / "lessons.json",
            (_FILE_TRIGGER, "先确认路径"),
            (_MAIL_TRIGGER, "先确认收件人"),
            (_CAL_TRIGGER, "先确认时区"),
        )
        self.embedder = HashEmbedder()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_ranks_by_trigger_similarity_with_floor_and_top_k(self) -> None:
        retriever = LessonRetriever(self.store, self.embedder, top_k=3, min_score=0.3)
        hits = retriever.retrieve("请读取文件内容 notes.md")
        self.assertEqual(hits[0].lesson.trigger, _FILE_TRIGGER)
        self.assertTrue(all(h.score >= 0.3 for h in hits))
        self.assertEqual(hits, sorted(hits, key=lambda h: -h.score))
        self.assertEqual(len(LessonRetriever(self.store, self.embedder, top_k=1, min_score=-1).retrieve("用户要求")), 1)
        self.assertEqual(LessonRetriever(self.store, self.embedder, min_score=0.99).retrieve("天气"), [])
        self.assertEqual(retriever.retrieve("   "), [])

    def test_evidence_requests_are_matched_like_triggers(self) -> None:
        evidence = Evidence(
            incident_id="inc_1",
            kind="tool_error",
            session_id="s_1",
            excerpt="[把端口改成 8080] edit_file 失败",
            request="把端口改成 8080",
        )
        lesson = self.store.add_draft("用户要求修改配置项", "先读文件再改", evidence=[evidence])
        approve(self.store, lesson.id)
        retriever = LessonRetriever(self.store, self.embedder, top_k=1, min_score=-1)

        hits = retriever.retrieve("把端口改成 9090")

        self.assertEqual(hits[0].lesson.id, lesson.id)
        reloaded = LessonStore(self.store.path).get(lesson.id)
        self.assertEqual(reloaded.evidence[0].request, "把端口改成 8080")

    def test_only_approved_lessons(self) -> None:
        self.store.add_draft("用户要求读取文件内容并总结", "草稿不注入")
        retriever = LessonRetriever(self.store, self.embedder, top_k=5, min_score=-1)
        self.assertEqual({h.lesson.status for h in retriever.retrieve("读取文件内容")}, {"approved"})
        file_lesson = next(item for item in self.store.list() if item.trigger == _FILE_TRIGGER)
        retract(self.store, file_lesson.id)
        ids = [h.lesson.id for h in retriever.retrieve("读取文件内容")]
        self.assertNotIn(file_lesson.id, ids)
        self.assertEqual(len(ids), 2)

    def test_caches_lesson_vectors_and_query_vectors(self) -> None:
        retriever = LessonRetriever(self.store, self.embedder, min_score=-1)
        retriever.retrieve("读取文件")
        retriever.retrieve("读取文件")
        retriever.retrieve("发送邮件")
        self.assertEqual(
            self.embedder.calls,
            [sorted([_FILE_TRIGGER, _MAIL_TRIGGER, _CAL_TRIGGER]), ["读取文件"], ["发送邮件"]],
        )
        # A new approved lesson: only its trigger is embedded.
        approve(self.store, self.store.add_draft("用户要求删除文件", "先确认").id)
        self.embedder.calls.clear()
        retriever.retrieve("读取文件")
        self.assertEqual(self.embedder.calls, [["用户要求删除文件"]])

    def test_model_change_invalidates_everything(self) -> None:
        retriever = LessonRetriever(self.store, self.embedder, min_score=-1)
        retriever.retrieve("读取文件")
        self.embedder.model_name = "hash-v2"
        self.embedder.calls.clear()
        retriever.retrieve("读取文件")
        self.assertEqual(len(self.embedder.calls), 2)
        self.assertEqual(len(self.embedder.calls[0]), 3)

    def test_fail_open_disables_instead_of_raising(self) -> None:
        class Broken:
            model_name = "broken"

            def embed(self, texts):
                raise RuntimeError("no model")

        with self.assertRaises(RuntimeError):
            LessonRetriever(self.store, Broken()).retrieve("读取文件")
        retriever = LessonRetriever(self.store, Broken(), fail_open=True)
        self.assertEqual(retriever.retrieve("读取文件"), [])
        self.assertEqual(retriever.retrieve("读取文件"), [])


_FIXED_NOW = datetime(2026, 10, 27, 9, 30, tzinfo=timezone(timedelta(hours=8)))


class ContextInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _approved_store(
            Path(self._tmp.name) / "lessons.json",
            (_FILE_TRIGGER, "先确认路径"),
            (_MAIL_TRIGGER, "先确认收件人"),
        )
        self.retriever = LessonRetriever(self.store, HashEmbedder(), top_k=1, min_score=0.3)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _builder(self, retriever: LessonRetriever | None) -> ContextBuilder:
        return ContextBuilder(
            base_system_prompt="BASE",
            memory_store=None,
            skill_registry=None,
            tool_registry=ToolRegistry(),
            now_provider=lambda: _FIXED_NOW,
            lesson_retriever=retriever,
        )

    def test_block_goes_on_latest_user_message_before_the_time_stamp(self) -> None:
        history = [
            MessageEvent.create(role="user", content="发送邮件给小王"),
            MessageEvent.create(role="assistant", content="好的"),
            MessageEvent.create(role="user", content="读取文件内容 a.txt"),
        ]
        history[0].created_at = "2026-10-27T01:20:00Z"
        history[2].created_at = "2026-10-27T01:30:00Z"
        built = self._builder(self.retriever).build(history)
        file_id = next(item.id for item in self.store.list() if item.trigger == _FILE_TRIGGER)
        self.assertEqual(built.lesson_ids, (file_id,))
        latest = built.messages[-1].content
        self.assertTrue(latest.startswith("读取文件内容 a.txt\n\n[相关经验"))
        self.assertIn(f"- {file_id} 情境: {_FILE_TRIGGER}；做法: 先确认路径", latest)
        self.assertTrue(latest.endswith("[发送时间 2026-10-27 09:30 星期二 UTC+08:00]"))
        self.assertLess(latest.index("[相关经验"), latest.index("[发送时间"))
        # Request-only: earlier turns get only their own send time, the system
        # prompt and the session stay clean.
        self.assertEqual(
            built.messages[1].content,
            "发送邮件给小王\n\n[发送时间 2026-10-27 09:20 星期二 UTC+08:00]",
        )
        self.assertNotIn("相关经验", built.messages[0].content)
        self.assertEqual(history[-1].content, "读取文件内容 a.txt")

    def test_skips_projected_summary(self) -> None:
        entries = [
            SessionEntry.from_message(MessageEvent.create(role="user", content="读取文件内容")),
            SessionEntry.compaction(summary="读取文件内容的总结", first_kept_entry_id=None),
        ]
        projected = SessionContextProjector.project_messages(entries)
        self.assertTrue(SessionContextProjector.is_summary_message(projected[0]))
        built = self._builder(self.retriever).build(projected)
        self.assertEqual(built.lesson_ids, ())
        self.assertNotIn("相关经验", built.messages[1].content)

    def test_no_retriever_or_no_hit_leaves_request_unchanged(self) -> None:
        history = [MessageEvent.create(role="user", content="读取文件内容")]
        plain = self._builder(None).build(history)
        self.assertEqual(plain.lesson_ids, ())
        miss = self._builder(
            LessonRetriever(self.store, HashEmbedder(), min_score=0.99)
        ).build([MessageEvent.create(role="user", content="今天天气")])
        self.assertEqual(miss.lesson_ids, ())
        self.assertNotIn("相关经验", miss.messages[-1].content)

    def test_builds_within_a_turn_are_identical(self) -> None:
        builder = self._builder(self.retriever)
        history = [MessageEvent.create(role="user", content="读取文件内容")]
        # The loop builds several times per turn (usage, each iteration);
        # an unchanged block keeps the prompt cache warm within the turn.
        self.assertEqual(builder.build(history).messages, builder.build(history).messages)


class _FinalReply(LLMClient):
    def chat(self, messages, tools=None, model=None) -> LLMResponse:
        return LLMResponse(content="完成")


class LoopPayloadTests(unittest.TestCase):
    def test_model_request_started_names_the_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            store = _approved_store(workspace / "lessons.json", (_FILE_TRIGGER, "先确认路径"))
            loop, manager = build_loop(_FinalReply(), ToolRegistry(), workspace)
            loop.context_builder.lesson_retriever = LessonRetriever(
                store, HashEmbedder(), min_score=0.3
            )
            events: list[RuntimeEvent] = []
            run_turn(loop, manager, "读取文件内容 a.txt", events=events)
            started = [e for e in events if e.type == "model.request.started"]
            self.assertEqual(started[0].payload["lessons"], [store.list()[0].id])
            # The exact request carries the block too, so traces show it.
            self.assertIn("[相关经验", started[0].payload["messages"][-1]["content"])

            events.clear()
            run_turn(loop, manager, "今天天气怎么样", events=events, run_id="r_2")
            started = [e for e in events if e.type == "model.request.started"]
            self.assertNotIn("lessons", started[0].payload)


class LessonsConfigTests(unittest.TestCase):
    def test_defaults_and_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = Config.from_env()
        self.assertEqual(
            (config.lessons_mode, config.lessons_top_k, config.lessons_min_score, config.lessons_path),
            ("auto", 3, 0.55, None),
        )
        self.assertEqual(config.resolve_lessons_path(Path("/h")), Path("/h/lessons.json"))
        env = {
            "MINIBOT_LESSONS": "ON",
            "MINIBOT_LESSONS_TOP_K": "5",
            "MINIBOT_LESSONS_MIN_SCORE": "0.6",
            "MINIBOT_LESSONS_PATH": "/tmp/x/lessons.json",
        }
        with patch.dict("os.environ", env, clear=True):
            config = Config.from_env()
        self.assertEqual(config.lessons_mode, "on")
        self.assertEqual(config.lessons_top_k, 5)
        self.assertAlmostEqual(config.lessons_min_score, 0.6)
        self.assertEqual(config.resolve_lessons_path(Path("/h")), Path("/tmp/x/lessons.json").resolve())

    def test_invalid_values(self) -> None:
        for env in (
            {"MINIBOT_LESSONS": "maybe"},
            {"MINIBOT_LESSONS_TOP_K": "0"},
            {"MINIBOT_LESSONS_MIN_SCORE": "high"},
            {"MINIBOT_LESSONS_MIN_SCORE": "1.5"},
        ):
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(ValueError, msg=env):
                    Config.from_env()


class BootstrapRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, mode: str, *, available: bool = True, path: Path | None = None):
        config = Config(lessons_mode=mode, lessons_path=path)  # type: ignore[arg-type]
        with patch("minibot.lessons.embed.fastembed_available", return_value=available):
            return _build_lesson_retriever(config, self.home)

    def test_modes(self) -> None:
        self.assertIsNone(self._build("off"))
        self.assertIsNone(self._build("auto", available=False))
        with self.assertRaisesRegex(RuntimeError, "fastembed"):
            self._build("on", available=False)
        # auto with no approved lesson stays off; on builds anyway.
        self.assertIsNone(self._build("auto"))
        strict = self._build("on")
        self.assertIsNotNone(strict)
        self.assertFalse(strict.fail_open)
        self.assertEqual(strict.store.path, (self.home / "lessons.json").resolve())

        custom = self.home / "elsewhere" / "lessons.json"
        _approved_store(custom, ("t", "g"))
        retriever = self._build("auto", path=custom)
        self.assertIsNotNone(retriever)
        self.assertTrue(retriever.fail_open)
        self.assertEqual(retriever.store.path, custom.resolve())
        self.assertEqual((retriever.top_k, retriever.min_score), (3, 0.55))


class ImportCostTests(unittest.TestCase):
    def test_importing_lessons_does_not_import_fastembed(self) -> None:
        code = (
            "import sys; import minibot.lessons, minibot.lessons.retrieve, "
            "minibot.lessons.pipeline, minibot.lessons.review, minibot.bootstrap; "
            "from minibot.lessons.embed import FastEmbedEmbedder; FastEmbedEmbedder(); "
            "print('fastembed' in sys.modules)"
        )
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
