from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from minibot.tools import ForgetTool, RememberTool
from minibot.tools.base import ToolExecutionContext
from minibot.user_memory import MemoryFileDamaged, UserMemoryStore


class UserMemoryDurabilityTests(unittest.TestCase):
    def test_a_damaged_file_stops_writes_instead_of_being_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = UserMemoryStore(root)
            for fact in ["用户叫小明", "用户在极光实验室工作", "用户喜欢简短回答"]:
                store.add(fact)
            whole = store.memory_path.read_text(encoding="utf-8")
            torn = whole[: len(whole) // 2]
            store.memory_path.write_text(torn, encoding="utf-8")

            later = UserMemoryStore(root)  # the next process

            # A request still builds: reading is tolerant.
            self.assertEqual(later.list(), [])
            self.assertIn(str(later.memory_path), later.read_error() or "")
            # Writing is not: the three facts are still in the file's first half.
            with self.assertRaises(MemoryFileDamaged):
                later.add("新事实")
            with self.assertRaises(MemoryFileDamaged):
                later.delete("mem_1")
            with self.assertRaises(MemoryFileDamaged):
                later.clear()
            self.assertEqual(later.memory_path.read_text(encoding="utf-8"), torn)

    def test_a_file_of_the_wrong_shape_is_damaged_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserMemoryStore(Path(tmpdir))
            store.memory_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaises(MemoryFileDamaged):
                store.add("新事实")

    def test_an_empty_file_is_an_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserMemoryStore(Path(tmpdir))
            store.memory_path.write_text("", encoding="utf-8")

            self.assertIsNone(store.read_error())
            self.assertEqual(store.add("新事实").id, "mem_1")
            self.assertEqual([item.content for item in store.list()], ["新事实"])

    def test_memory_tools_report_a_damaged_file_as_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserMemoryStore(Path(tmpdir))
            store.memory_path.write_text("{", encoding="utf-8")
            context = ToolExecutionContext(session_id="s_test")

            remembered = RememberTool(store).execute(context=context, content="新事实")
            forgotten = ForgetTool(store).execute(context=context, memory_id="mem_1")

            self.assertFalse(remembered.ok)
            self.assertFalse(forgotten.ok)
            self.assertEqual(store.memory_path.read_text(encoding="utf-8"), "{")

    def test_writers_on_the_same_file_do_not_lose_each_others_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # One store per writer, as the CLI and the daemon each have one.
            stores = [UserMemoryStore(root) for _ in range(8)]

            def work(index: int) -> None:
                for step in range(25):
                    stores[index].add(f"fact {index}-{step}")

            threads = [
                threading.Thread(target=work, args=(index,)) for index in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            items = UserMemoryStore(root).list()
            self.assertEqual(len(items), 200)
            self.assertEqual(len({item.id for item in items}), 200)
            self.assertEqual(len({item.content for item in items}), 200)

    def test_an_id_forgotten_through_another_store_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cli, daemon = UserMemoryStore(root), UserMemoryStore(root)
            cli.add("第一条")
            second = daemon.add("第二条")
            daemon.delete(second.id)

            third = cli.add("第三条")

            self.assertEqual(second.id, "mem_2")
            self.assertEqual(third.id, "mem_3")

    def test_a_failed_save_leaves_the_old_file_and_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = UserMemoryStore(root)
            store.add("第一条")
            before = store.memory_path.read_text(encoding="utf-8")

            with mock.patch("minibot.user_memory.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    store.add("第二条")

            self.assertEqual(store.memory_path.read_text(encoding="utf-8"), before)
            self.assertEqual(
                sorted(name for name in os.listdir(root) if name != "locks"),
                ["user_memory.json"],
            )
            # The index was not consumed by the failed write.
            self.assertEqual(store.add("第二条").id, "mem_2")
            self.assertEqual(json.loads(store.memory_path.read_text())["next_index"], 3)


if __name__ == "__main__":
    unittest.main()
