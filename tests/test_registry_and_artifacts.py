from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.artifacts import ArtifactStore
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.read_artifact import ReadArtifactTool
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, value: str) -> ToolOutput:
        del context
        return ToolOutput.success("echo ok", data={"value": value})


class KernelEchoTool(EchoTool):
    @property
    def name(self) -> str:
        return "kernel_echo"

    @property
    def layer(self) -> str:
        return "kernel"


class ExplodingTool(Tool):
    @property
    def name(self) -> str:
        return "explode"

    @property
    def description(self) -> str:
        return "explode"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context
        raise RuntimeError("boom")


class TypeErrorTool(Tool):
    @property
    def name(self) -> str:
        return "type_error"

    @property
    def description(self) -> str:
        return "type_error"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        raise TypeError("internal type error")


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())
        self.registry.register(KernelEchoTool())
        self.registry.register(ExplodingTool())
        self.registry.register(TypeErrorTool())
        self.context = ToolExecutionContext(session_id="s_test")

    def test_unknown_tool_returns_not_found(self) -> None:
        result = self.registry.execute("missing", {}, context=self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "not_found")

    def test_non_dict_args_returns_invalid_args(self) -> None:
        result = self.registry.execute("echo", "bad", context=self.context)  # type: ignore[arg-type]
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "invalid_args")

    def test_bad_signature_returns_invalid_args(self) -> None:
        result = self.registry.execute("echo", {}, context=self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "invalid_args")

    def test_extra_args_return_invalid_args(self) -> None:
        result = self.registry.execute(
            "echo",
            {"value": "ok", "extra": "nope"},
            context=self.context,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "invalid_args")

    def test_exception_returns_error(self) -> None:
        result = self.registry.execute("explode", {}, context=self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "error")

    def test_internal_type_error_returns_error(self) -> None:
        result = self.registry.execute("type_error", {}, context=self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "error")

    def test_list_tools_can_filter_by_layer(self) -> None:
        self.assertEqual(
            [tool.name for tool in self.registry.list_tools(layer="kernel")],
            ["kernel_echo"],
        )
        self.assertEqual(
            [tool.name for tool in self.registry.list_tools(layer="extension")],
            ["echo", "explode", "type_error"],
        )

    def test_get_definitions_can_filter_by_layer(self) -> None:
        kernel_names = [
            item["function"]["name"]
            for item in self.registry.get_definitions(layer="kernel")
        ]
        extension_names = [
            item["function"]["name"]
            for item in self.registry.get_definitions(layer="extension")
        ]

        self.assertEqual(kernel_names, ["kernel_echo"])
        self.assertEqual(extension_names, ["echo", "explode", "type_error"])


class ReadArtifactToolTests(unittest.TestCase):
    def test_read_artifact_paginates_by_character(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            ref = store.put_text("s_test", "abcdef", name="sample")
            tool = ReadArtifactTool(store)

            result = tool.execute(
                context=context,
                artifact_id=ref.id,
                offset=2,
                limit=3,
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["content"], "cde")
            self.assertEqual(result.data["next_offset"], 5)
            self.assertTrue(result.data["has_more"])

    def test_read_artifact_large_page_is_shrunk_to_fit_result_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            ref = store.put_text("s_test", "x" * 12000, name="large")
            tool = ReadArtifactTool(store)

            result = tool.execute(
                context=context,
                artifact_id=ref.id,
                offset=0,
                limit=8000,
            )

            self.assertTrue(result.ok)
            self.assertLess(len(result.data["content"]), 8000)
            self.assertEqual(result.data["returned_chars"], len(result.data["content"]))
            self.assertEqual(result.data["next_offset"], len(result.data["content"]))
            self.assertTrue(result.data["has_more"])
            self.assertTrue(result.truncated)

if __name__ == "__main__":
    unittest.main()
