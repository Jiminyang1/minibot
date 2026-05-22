from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.llm_providers.openai_compatible import (
    model_tool_definition_to_openai,
    model_tool_definitions_to_openai,
)
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.definitions import ModelToolDefinition
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
        return ToolOutput.success("ok", data={"value": value})


class ToolDefinitionConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())

    def test_registry_returns_internal_tool_definition(self) -> None:
        definition = self.registry.get_definitions()[0]

        self.assertIsInstance(definition, ModelToolDefinition)
        self.assertEqual(definition.name, "echo")

    def test_provider_conversion_keeps_internal_definition_immutable(self) -> None:
        definition = self.registry.get_definitions()[0]
        converted = model_tool_definition_to_openai(definition)
        batch = model_tool_definitions_to_openai([definition])

        self.assertEqual(converted["function"]["name"], "echo")
        self.assertEqual(batch[0]["function"]["name"], "echo")
        self.assertEqual(definition.name, "echo")
        self.assertEqual(definition.description, "echo")


if __name__ == "__main__":
    unittest.main()
