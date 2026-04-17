from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.tools.result import ArtifactRef, ToolResult


class ToolResultTests(unittest.TestCase):
    def test_to_model_content_excludes_meta(self) -> None:
        result = ToolResult.success(
            "done",
            data={"value": 1},
            artifact=ArtifactRef(id="a_test", kind="text", name="sample"),
            truncated=True,
            meta={"secret": "hidden"},
        )

        payload = json.loads(result.to_model_content())

        self.assertEqual(
            payload,
            {
                "ok": True,
                "code": "success",
                "summary": "done",
                "data": {"value": 1},
                "artifact": {
                    "id": "a_test",
                    "kind": "text",
                    "name": "sample",
                },
                "truncated": True,
            },
        )
        self.assertNotIn("meta", payload)


if __name__ == "__main__":
    unittest.main()
