"""The threshold benchmark runs end to end on its data with any embedder."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.lessons.benchmark import DATA_PATH, format_report, measure

from test_lessons import HashEmbedder


class BenchmarkTests(unittest.TestCase):
    def test_measures_every_threshold_on_the_shipped_data(self) -> None:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

        numbers = measure(data, HashEmbedder().embed)

        retrieval = numbers["retrieval"]
        self.assertEqual(set(retrieval), {"trigger_only", "trigger_and_evidence"})
        self.assertTrue(retrieval["trigger_and_evidence"]["top1"].endswith("/10"))
        self.assertEqual(numbers["clustering"]["groups"], 4)
        self.assertEqual(len(numbers["dedupe"]["same"]), 3)
        self.assertIn("手写的小样本", format_report(numbers))


if __name__ == "__main__":
    unittest.main()
