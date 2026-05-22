from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.runtime.hooks import (
    HookContext,
    ModelRequest,
    RuntimeHook,
    RuntimeHookManager,
    ToolExecuteDecision,
    ToolPrepareRequest,
)
from minibot.runtime.hooks_builtin import ApprovalHook, ApprovalPolicy
from minibot.runtime.messages import ModelMessage
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.registry import PreparedToolCall
from minibot.tools.result import ToolOutput


class _DummyTool(Tool):
    def __init__(self, *, requires_approval: bool = False) -> None:
        super().__init__()
        self._requires_approval = requires_approval

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @property
    def requires_approval(self) -> bool:
        return self._requires_approval

    def execute(self, *, context: ToolExecutionContext) -> ToolOutput:
        del context
        return ToolOutput.success("ok")


class _RewriteHook(RuntimeHook):
    priority = 20

    def before_model_request(
        self,
        context: HookContext,
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        return ModelRequest(
            model="rewritten",
            messages=[*request.messages, ModelMessage.create(role="user", content="extra")],
            tool_definitions=request.tool_definitions,
        )

    def before_tool_prepare(
        self,
        context: HookContext,
        request: ToolPrepareRequest,
    ) -> ToolPrepareRequest:
        del context
        return ToolPrepareRequest(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            args={"value": "rewritten"},
            tool=request.tool,
        )

    def after_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
        output: ToolOutput,
    ) -> ToolOutput:
        del context, call, output
        return ToolOutput.success("rewritten result", data={"hooked": True})


class _BlockHook(RuntimeHook):
    priority = 10

    def before_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
    ) -> ToolExecuteDecision:
        del context, call
        return ToolExecuteDecision(ToolOutput.failure("denied", "blocked"))


class _FailingHook(RuntimeHook):
    def before_tool_prepare(
        self,
        context: HookContext,
        request: ToolPrepareRequest,
    ) -> ToolPrepareRequest:
        del context, request
        raise RuntimeError("prepare failed")

    def before_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
    ) -> ToolExecuteDecision:
        del context, call
        raise RuntimeError("execute failed")

    def after_tool_execute(
        self,
        context: HookContext,
        call: PreparedToolCall,
        output: ToolOutput,
    ) -> ToolOutput:
        del context, call, output
        raise RuntimeError("after failed")


class _FailingModelHook(RuntimeHook):
    def before_model_request(
        self,
        context: HookContext,
        request: ModelRequest,
    ) -> ModelRequest:
        del context, request
        raise RuntimeError("model failed")


class RuntimeHookManagerTests(unittest.TestCase):
    def _context(self) -> HookContext:
        return HookContext(
            run_id="r_test",
            session_id="s_test",
            workspace=Path(tempfile.gettempdir()),
        )

    def _prepared_call(self) -> PreparedToolCall:
        return PreparedToolCall(
            tool=_DummyTool(),
            args={},
            context=ToolExecutionContext(session_id="s_test"),
            tool_call_id="call_1",
        )

    def test_pipeline_rewrites_model_request_tool_args_and_tool_output(self) -> None:
        manager = RuntimeHookManager([_RewriteHook()])
        context = self._context()

        model_request = manager.before_model_request(
            context,
            ModelRequest(model="original", messages=[], tool_definitions=[]),
        )
        self.assertEqual(model_request.model, "rewritten")
        self.assertEqual(model_request.messages[-1].content, "extra")

        prepare_request = manager.before_tool_prepare(
            context,
            ToolPrepareRequest(
                tool_call_id="call_1",
                tool_name="dummy",
                args={},
                tool=_DummyTool(),
            ),
        )
        self.assertIsInstance(prepare_request, ToolPrepareRequest)
        assert isinstance(prepare_request, ToolPrepareRequest)
        self.assertEqual(prepare_request.args, {"value": "rewritten"})

        output = manager.after_tool_execute(
            context,
            self._prepared_call(),
            ToolOutput.success("original"),
        )
        self.assertEqual(output.summary, "rewritten result")
        self.assertEqual(output.data, {"hooked": True})

    def test_hooks_run_in_priority_order(self) -> None:
        calls: list[str] = []

        class _OrderHook(RuntimeHook):
            def __init__(self, name: str, priority: int) -> None:
                self.name = name
                self.priority = priority

            def before_model_request(
                self,
                context: HookContext,
                request: ModelRequest,
            ) -> ModelRequest:
                del context
                calls.append(self.name)
                return request

        manager = RuntimeHookManager(
            [
                _OrderHook("late", 50),
                _OrderHook("early", 10),
            ]
        )
        manager.before_model_request(
            self._context(),
            ModelRequest(model="original", messages=[], tool_definitions=[]),
        )

        self.assertEqual(calls, ["early", "late"])

    def test_before_tool_execute_can_block_execution(self) -> None:
        manager = RuntimeHookManager([_BlockHook()])
        decision = manager.before_tool_execute(self._context(), self._prepared_call())

        self.assertTrue(decision.blocked)
        assert decision.output is not None
        self.assertEqual(decision.output.code, "denied")

    def test_tool_hook_failures_return_error_outputs(self) -> None:
        manager = RuntimeHookManager([_FailingHook()])
        context = self._context()

        prepare_result = manager.before_tool_prepare(
            context,
            ToolPrepareRequest(
                tool_call_id="call_1",
                tool_name="dummy",
                args={},
                tool=_DummyTool(),
            ),
        )
        self.assertIsInstance(prepare_result, ToolOutput)
        assert isinstance(prepare_result, ToolOutput)
        self.assertFalse(prepare_result.ok)
        self.assertEqual(prepare_result.data["hook_phase"], "before_tool_prepare")

        execute_decision = manager.before_tool_execute(
            context,
            self._prepared_call(),
        )
        self.assertTrue(execute_decision.blocked)
        assert execute_decision.output is not None
        self.assertFalse(execute_decision.output.ok)
        self.assertEqual(
            execute_decision.output.data["hook_phase"],
            "before_tool_execute",
        )

        after_result = manager.after_tool_execute(
            context,
            self._prepared_call(),
            ToolOutput.success("original"),
        )
        self.assertFalse(after_result.ok)
        self.assertEqual(after_result.data["hook_phase"], "after_tool_execute")

    def test_non_tool_hook_failure_stops_pipeline(self) -> None:
        manager = RuntimeHookManager([_FailingModelHook()])

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            manager.before_model_request(
                self._context(),
                ModelRequest(model="original", messages=[], tool_definitions=[]),
            )

    def test_approval_hook_denies_sensitive_tool_without_invoking_tool(self) -> None:
        calls = 0

        def _handler(request) -> bool:
            nonlocal calls
            calls += 1
            self.assertEqual(request.tool_call_id, "call_1")
            return False

        manager = RuntimeHookManager(
            [ApprovalHook(ApprovalPolicy(handler=_handler, mode="ask"))]
        )
        call = PreparedToolCall(
            tool=_DummyTool(requires_approval=True),
            args={"danger": True},
            context=ToolExecutionContext(session_id="s_test"),
            tool_call_id="call_1",
        )
        decision = manager.before_tool_execute(self._context(), call)

        self.assertEqual(calls, 1)
        self.assertTrue(decision.blocked)
        assert decision.output is not None
        self.assertEqual(decision.output.code, "denied")


if __name__ == "__main__":
    unittest.main()
