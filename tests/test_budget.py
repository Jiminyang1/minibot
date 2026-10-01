from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.budget import TokenBudget
from minibot.runtime.messages import ModelMessage
from minibot.runtime.token_budget import estimate_request_token_breakdown
from minibot.tools.definitions import ModelToolDefinition


class TokenBudgetPolicyTests(unittest.TestCase):
    def test_model_limit_output_ceiling_and_compaction_trigger_are_separate(self) -> None:
        budget = TokenBudget(
            context_window_tokens=1_048_576,
            max_output_tokens=32_000,
            compact_token_threshold=500_000,
        )

        self.assertEqual(budget.input_budget, 1_016_576)
        self.assertEqual(budget.compaction_trigger_tokens, 500_000)
        self.assertFalse(budget.should_compact(500_000))
        self.assertTrue(budget.should_compact(500_001))

    def test_default_trigger_is_the_model_hard_input_limit(self) -> None:
        budget = TokenBudget(
            context_window_tokens=400_000,
            model_max_input_tokens=272_000,
            max_output_tokens=4_096,
        )

        self.assertEqual(budget.input_budget, 272_000)
        self.assertEqual(budget.compaction_trigger_tokens, 272_000)

    def test_trigger_cannot_exceed_hard_input_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能超过模型硬输入上限"):
            TokenBudget(
                context_window_tokens=400_000,
                max_output_tokens=128_000,
                compact_token_threshold=300_000,
            )

    def test_request_estimate_reports_tool_schema_separately(self) -> None:
        estimate = estimate_request_token_breakdown(
            [ModelMessage.create(role="user", content="hello")],
            [
                ModelToolDefinition(
                    name="echo",
                    description="Echo text",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                )
            ],
        )

        self.assertGreater(estimate.message_tokens, 0)
        self.assertGreater(estimate.tool_definition_tokens, 0)
        self.assertEqual(
            estimate.total_tokens,
            estimate.message_tokens + estimate.tool_definition_tokens,
        )


if __name__ == "__main__":
    unittest.main()
