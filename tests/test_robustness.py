from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.llm import (
    LLMClient,
    LLMResponse,
    LLMStreamEvent,
    is_retryable_llm_error,
)
from minibot.runtime.cancel import RunCancelled
from minibot.runtime.events import RuntimeEvent
from minibot.runtime.messages import ModelMessage
from minibot.session import MessageEvent
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput

from loop_harness import build_loop, run_turn


class _RateLimited(Exception):
    status_code = 429


class _ServerError(Exception):
    status_code = 503


class _BadRequest(Exception):
    status_code = 400


class _FlakyLLM(LLMClient):
    """Raises the scripted failures first, then streams a normal reply."""

    def __init__(self, failures: list[Exception], reply: str = "ok") -> None:
        self._failures = list(failures)
        self._reply = reply
        self.calls = 0

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        raise AssertionError("must be consumed via chat_stream")

    def chat_stream(self, messages, tools=None, model=None):
        del messages, tools, model
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        yield LLMStreamEvent.completed(LLMResponse(content=self._reply))


class _FailsAfterDeltaLLM(LLMClient):
    def chat(self, messages, tools=None, model=None) -> LLMResponse:
        raise AssertionError("must be consumed via chat_stream")

    def chat_stream(self, messages, tools=None, model=None):
        del messages, tools, model
        yield LLMStreamEvent.text_delta("partial ")
        raise _ServerError("stream broke mid-way")


def _fast_retry_loop(llm, workspace: Path, *, max_retries: int = 3):
    loop, manager = build_loop(llm, ToolRegistry(), workspace)
    loop.llm_max_retries = max_retries
    loop.retry_base_delay = 0.001
    return loop, manager


class RetryClassificationTests(unittest.TestCase):
    def test_status_codes_classify_as_documented(self) -> None:
        self.assertTrue(is_retryable_llm_error(_RateLimited()))
        self.assertTrue(is_retryable_llm_error(_ServerError()))
        self.assertFalse(is_retryable_llm_error(_BadRequest()))
        self.assertTrue(is_retryable_llm_error(ConnectionError("reset")))
        self.assertTrue(is_retryable_llm_error(TimeoutError("slow")))
        self.assertFalse(is_retryable_llm_error(RuntimeError("logic bug")))


class LoopRetryTests(unittest.TestCase):
    def test_transient_failures_retry_then_succeed_with_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = _FlakyLLM([_RateLimited("429"), _ServerError("503")], reply="done")
            loop, manager = _fast_retry_loop(llm, Path(tmpdir))
            events: list[RuntimeEvent] = []

            outcome, _ = run_turn(loop, manager, "hello", events=events)

            self.assertEqual(outcome.reply, "done")
            self.assertEqual(llm.calls, 3)
            retrying = [e for e in events if e.type == "model.request.retrying"]
            self.assertEqual(
                [(e.payload["attempt"], e.payload["error_type"]) for e in retrying],
                [(1, "_RateLimited"), (2, "_ServerError")],
            )
            self.assertEqual(retrying[0].payload["max_retries"], 3)

    def test_non_retryable_error_fails_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = _FlakyLLM([_BadRequest("bad schema")])
            loop, manager = _fast_retry_loop(llm, Path(tmpdir))
            events: list[RuntimeEvent] = []

            with self.assertRaises(_BadRequest):
                run_turn(loop, manager, "hello", events=events)

            self.assertEqual(llm.calls, 1)
            self.assertEqual(
                [e for e in events if e.type == "model.request.retrying"], []
            )

    def test_exhausted_retries_raise_the_last_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = _FlakyLLM([_RateLimited(f"try {i}") for i in range(5)])
            loop, manager = _fast_retry_loop(llm, Path(tmpdir), max_retries=2)
            events: list[RuntimeEvent] = []

            with self.assertRaises(_RateLimited):
                run_turn(loop, manager, "hello", events=events)

            self.assertEqual(llm.calls, 3)
            self.assertEqual(
                len([e for e in events if e.type == "model.request.retrying"]), 2
            )

    def test_no_retry_after_a_delta_reached_the_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            loop, manager = _fast_retry_loop(_FailsAfterDeltaLLM(), Path(tmpdir))
            events: list[RuntimeEvent] = []

            with self.assertRaises(_ServerError):
                run_turn(loop, manager, "hello", events=events)

            self.assertEqual(
                [e for e in events if e.type == "model.request.retrying"], []
            )

    def test_cancellation_during_backoff_raises_run_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cancel_event = threading.Event()

            class _FailAndCancel(LLMClient):
                def chat(self, messages, tools=None, model=None):
                    raise AssertionError

                def chat_stream(self, messages, tools=None, model=None):
                    del messages, tools, model
                    cancel_event.set()
                    raise _RateLimited("429")
                    yield  # pragma: no cover - marks this as a generator

            loop, manager = _fast_retry_loop(_FailAndCancel(), Path(tmpdir))
            loop.retry_base_delay = 5.0  # cancellation must win, not the sleep

            with self.assertRaises(RunCancelled):
                run_turn(loop, manager, "hello", cancel_event=cancel_event)


class SummarizerFallbackTests(unittest.TestCase):
    def test_failed_summarizer_degrades_to_truncated_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            def _broken_summarizer(request):
                raise _ServerError("summary llm down")

            loop, manager = build_loop(
                _FlakyLLM([], reply="unused"),
                ToolRegistry(),
                Path(tmpdir),
                summarizer=_broken_summarizer,
                compact_keep_recent_tokens=1,
            )
            session = manager.create_session("s_test")
            loop._append(
                session,
                MessageEvent.create(role="user", content="很旧的问题 " * 500),
            )
            loop._append(
                session, MessageEvent.create(role="assistant", content="旧回答")
            )
            loop._append(
                session, MessageEvent.create(role="user", content="当前问题")
            )

            message = loop.compactor.reduce(session, tokens_before=10**6)

            self.assertIn("已压缩", message)
            self.assertIn("摘要降级为截断: _ServerError", message)
            projected_summary = session.messages[0].content
            self.assertIn("自动摘要失败（_ServerError）", projected_summary)
            self.assertIn("很旧的问题", projected_summary)
            # The compaction entry persisted despite the summariser failure.
            self.assertEqual(session.entries[-1].type, "compaction")
            reloaded = manager.load("s_test")
            assert reloaded is not None
            self.assertEqual(reloaded.entries[-1].type, "compaction")


class _TypedTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.executions = 0

    @property
    def name(self) -> str:
        return "typed"

    @property
    def description(self) -> str:
        return "typed"

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def execute(
        self, *, context: ToolExecutionContext, path: str, limit: int = 10
    ) -> ToolOutput:
        del context
        self.executions += 1
        return ToolOutput.success("ok", data={"path": path, "limit": limit})


class SchemaValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.tool = _TypedTool()
        self.registry.register(self.tool)
        self.context = ToolExecutionContext(session_id="s_test")

    def _execute(self, args: dict) -> ToolOutput:
        return self.registry.execute("typed", args, context=self.context)

    def test_wrong_type_is_rejected_before_execution(self) -> None:
        output = self._execute({"path": 123})

        self.assertFalse(output.ok)
        self.assertEqual(output.code, "invalid_args")
        self.assertIn("$.path", output.summary)
        self.assertIn("is not of type 'string'", output.summary)
        self.assertEqual(self.tool.executions, 0)

    def test_missing_required_field_is_rejected(self) -> None:
        output = self._execute({"limit": 3})

        self.assertEqual(output.code, "invalid_args")
        self.assertIn("'path' is a required property", output.summary)
        self.assertEqual(self.tool.executions, 0)

    def test_extra_field_rejected_by_additional_properties(self) -> None:
        output = self._execute({"path": "a.txt", "bogus": True})

        self.assertEqual(output.code, "invalid_args")
        self.assertEqual(self.tool.executions, 0)

    def test_constraint_violation_is_rejected(self) -> None:
        output = self._execute({"path": "a.txt", "limit": 0})

        self.assertEqual(output.code, "invalid_args")
        self.assertIn("$.limit", output.summary)
        self.assertEqual(self.tool.executions, 0)

    def test_valid_args_execute(self) -> None:
        output = self._execute({"path": "a.txt", "limit": 3})

        self.assertTrue(output.ok)
        self.assertEqual(self.tool.executions, 1)

    def test_malformed_schema_degrades_to_signature_binding(self) -> None:
        class _LooseTool(_TypedTool):
            @property
            def name(self) -> str:
                return "loose"

            @property
            def parameters(self) -> dict[str, object]:
                return {"type": "objekt", "properties": 42}  # not a valid schema

        loose = _LooseTool()
        self.registry.register(loose)

        output = self.registry.execute(
            "loose", {"path": "a.txt"}, context=self.context
        )

        self.assertTrue(output.ok)
        self.assertEqual(loose.executions, 1)


if __name__ == "__main__":
    unittest.main()
