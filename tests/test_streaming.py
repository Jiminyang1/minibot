from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minibot.cli import CliRenderer
from minibot.llm import LLMClient, LLMResponse, LLMStreamEvent, TokenUsage, ToolCall
from minibot.llm_profile import LLMProfile, OpenAICompatibleCompat
from minibot.runtime.cancel import RunCancelled
from minibot.runtime.events import RuntimeEvent, RuntimeEventEmitter
from minibot.runtime.messages import ModelMessage
from minibot.tools.base import Tool, ToolExecutionContext
from minibot.tools.definitions import ModelToolDefinition
from minibot.tools.registry import ToolRegistry
from minibot.tools.result import ToolOutput

from loop_harness import build_loop, run_turn


def _ns(**kwargs) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kwargs)


def _delta_chunk(
    *,
    content=None,
    reasoning=None,
    tool_calls=None,
    finish_reason=None,
    usage=None,
) -> types.SimpleNamespace:
    delta = _ns(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return _ns(
        choices=[_ns(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


class _FakeStream:
    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


def _provider_client(*, supports_streaming: bool = True):
    from minibot.llm_providers.openai_compatible import OpenAICompatibleClient

    profile = LLMProfile(
        provider="openai",
        api="openai_chat_completions",
        model="test-model",
        base_url=None,
        api_key="sk-test",
        compat=OpenAICompatibleCompat(supports_streaming=supports_streaming),
    )
    return OpenAICompatibleClient(profile)


class ProviderStreamTests(unittest.TestCase):
    def test_stream_accumulates_deltas_tool_fragments_and_usage(self) -> None:
        client = _provider_client()
        stream = _FakeStream(
            [
                _delta_chunk(content="Hel"),
                _delta_chunk(content="lo"),
                _delta_chunk(reasoning="think "),
                _delta_chunk(
                    tool_calls=[
                        _ns(
                            index=0,
                            id="call_1",
                            function=_ns(name="echo", arguments='{"va'),
                        )
                    ]
                ),
                _delta_chunk(
                    tool_calls=[
                        _ns(
                            index=0,
                            id=None,
                            function=_ns(name=None, arguments='lue":"hi"}'),
                        )
                    ]
                ),
                _delta_chunk(finish_reason="tool_calls"),
                _ns(
                    choices=[],
                    usage=_ns(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ),
            ]
        )
        seen_kwargs: dict = {}

        def _create(**kwargs):
            seen_kwargs.update(kwargs)
            return stream

        client._client = _ns(chat=_ns(completions=_ns(create=_create)))

        events = list(
            client.chat_stream([ModelMessage.create(role="user", content="hi")])
        )

        self.assertTrue(seen_kwargs["stream"])
        self.assertEqual(seen_kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(
            [event.kind for event in events],
            ["text", "text", "reasoning", "response"],
        )
        self.assertEqual([e.text for e in events[:2]], ["Hel", "lo"])
        terminal = events[-1].response
        assert terminal is not None
        self.assertEqual(terminal.content, "Hello")
        self.assertEqual(terminal.reasoning_content, "think ")
        self.assertEqual(
            terminal.tool_calls,
            [ToolCall(id="call_1", name="echo", arguments='{"value":"hi"}')],
        )
        assert terminal.usage is not None
        self.assertEqual(terminal.usage.input_tokens, 10)
        self.assertEqual(terminal.usage.output_tokens, 5)
        self.assertEqual(terminal.usage.total_tokens, 15)
        assert terminal.debug is not None
        self.assertTrue(terminal.debug["streamed"])
        self.assertTrue(stream.closed)

    def test_streaming_disabled_falls_back_to_chat(self) -> None:
        client = _provider_client(supports_streaming=False)
        resp = _ns(
            choices=[
                _ns(message=_ns(content="ok", tool_calls=None), finish_reason="stop")
            ],
            usage=None,
        )
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            return resp

        client._client = _ns(chat=_ns(completions=_ns(create=_create)))

        events = list(
            client.chat_stream([ModelMessage.create(role="user", content="hi")])
        )

        self.assertEqual([event.kind for event in events], ["response"])
        assert events[0].response is not None
        self.assertEqual(events[0].response.content, "ok")
        self.assertNotIn("stream", calls[0])

    def test_default_chat_stream_wraps_chat_in_one_terminal(self) -> None:
        class _Plain(LLMClient):
            def chat(self, messages, tools=None, model=None) -> LLMResponse:
                del messages, tools, model
                return LLMResponse(content="done")

        events = list(_Plain().chat_stream([]))

        self.assertEqual([event.kind for event in events], ["response"])
        assert events[0].response is not None
        self.assertEqual(events[0].response.content, "done")


class _EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo"

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    def execute(self, *, context: ToolExecutionContext, **kwargs: object) -> ToolOutput:
        del context, kwargs
        return ToolOutput.success("ok")


class _StreamingLLM(LLMClient):
    """Scripted native-streaming client; chat() must never be called."""

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = list(scripts)
        self.closed_streams = 0

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        raise AssertionError("streaming client must be consumed via chat_stream")

    def chat_stream(self, messages, tools=None, model=None):
        del messages, tools, model
        script = self._scripts.pop(0)
        try:
            yield from script
        finally:
            self.closed_streams += 1


class _CancellingLLM(_StreamingLLM):
    def __init__(self, scripts, cancel_event: threading.Event) -> None:
        super().__init__(scripts)
        self._cancel_event = cancel_event

    def chat_stream(self, messages, tools=None, model=None):
        del messages, tools, model
        try:
            yield LLMStreamEvent.text_delta("partial ")
            self._cancel_event.set()
            yield LLMStreamEvent.text_delta("more")
            yield LLMStreamEvent.completed(LLMResponse(content="never"))
        finally:
            self.closed_streams += 1


class LoopStreamingTests(unittest.TestCase):
    def test_deltas_become_events_and_terminal_drives_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            registry.register(_EchoTool())
            llm = _StreamingLLM(
                [
                    [
                        LLMStreamEvent.text_delta("查一下"),
                        LLMStreamEvent.completed(
                            LLMResponse(
                                content="查一下",
                                tool_calls=[
                                    ToolCall(id="call_1", name="echo", arguments="{}")
                                ],
                                usage=TokenUsage(
                                    input_tokens=10, output_tokens=2, total_tokens=12
                                ),
                            )
                        ),
                    ],
                    [
                        LLMStreamEvent.reasoning_delta("hmm "),
                        LLMStreamEvent.text_delta("答案"),
                        LLMStreamEvent.text_delta("是 42"),
                        LLMStreamEvent.completed(
                            LLMResponse(
                                content="答案是 42",
                                usage=TokenUsage(
                                    input_tokens=20, output_tokens=4, total_tokens=24
                                ),
                            )
                        ),
                    ],
                ]
            )
            loop, manager = build_loop(llm, registry, Path(tmpdir))
            events: list[RuntimeEvent] = []

            outcome, session = run_turn(loop, manager, "问题", events=events)

            self.assertEqual(outcome.reply, "答案是 42")
            assert outcome.usage is not None
            self.assertEqual(outcome.usage.total_tokens, 36)

            deltas = [e for e in events if e.type == "message.delta"]
            self.assertEqual(
                [(e.payload["iteration"], e.payload["channel"], e.payload["text"]) for e in deltas],
                [
                    (1, "text", "查一下"),
                    (2, "reasoning", "hmm "),
                    (2, "text", "答案"),
                    (2, "text", "是 42"),
                ],
            )
            # Terminal response, not deltas, is what persists.
            self.assertEqual(
                [(m.role, m.content) for m in session.messages],
                [
                    ("user", "问题"),
                    ("assistant", "查一下"),
                    ("tool", session.messages[2].content),
                    ("assistant", "答案是 42"),
                ],
            )
            # Delta events precede the iteration's model.request.completed.
            types_in_order = [e.type for e in events]
            self.assertLess(
                types_in_order.index("message.delta"),
                types_in_order.index("model.request.completed"),
            )
            self.assertEqual(llm.closed_streams, 2)

    def test_cancel_mid_stream_leaves_clean_session_and_closes_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            cancel_event = threading.Event()
            llm = _CancellingLLM([], cancel_event)
            loop, manager = build_loop(llm, registry, Path(tmpdir))

            with self.assertRaises(RunCancelled):
                run_turn(
                    loop,
                    manager,
                    "hello",
                    cancel_event=cancel_event,
                )

            session = manager.load("s_test")
            assert session is not None
            self.assertEqual(
                [message.role for message in session.messages],
                ["user"],
            )
            self.assertEqual(llm.closed_streams, 1)

    def test_missing_terminal_event_is_a_contract_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            llm = _StreamingLLM([[LLMStreamEvent.text_delta("only delta")]])
            loop, manager = build_loop(llm, registry, Path(tmpdir))

            with self.assertRaisesRegex(RuntimeError, "缺少终局事件"):
                run_turn(loop, manager, "hello")


class TransientEventStoreTests(unittest.TestCase):
    def test_deltas_broadcast_live_but_never_replay(self) -> None:
        from minibot.server import RunEventStore

        store = RunEventStore()
        store.create("r_1")
        emitter = RuntimeEventEmitter(
            run_id="r_1", session_id="s_1", handler=store.append
        )
        emitter.emit("run.started", {})

        live = store.subscribe("r_1")
        assert live is not None
        self.assertEqual(live.get_nowait().type, "run.started")

        emitter.emit("message.delta", {"channel": "text", "text": "hi"})
        emitter.emit("message.completed", {"content": "hi"})

        self.assertEqual(live.get_nowait().type, "message.delta")
        self.assertEqual(live.get_nowait().type, "message.completed")

        replay = store.subscribe("r_1")
        assert replay is not None
        replayed_types = []
        while not replay.empty():
            replayed_types.append(replay.get_nowait().type)
        self.assertEqual(replayed_types, ["run.started", "message.completed"])


class CliTypewriterTests(unittest.TestCase):
    def _renderer(self) -> tuple[CliRenderer, io.StringIO]:
        out = io.StringIO()
        return CliRenderer(no_color=True, stdout=out), out

    def _event(self, event_type: str, payload: dict, seq: int = 1) -> RuntimeEvent:
        return RuntimeEvent(
            id=f"r_test:{seq}",
            run_id="r_test",
            session_id="s_test",
            seq=seq,
            type=event_type,
            created_at="2026-07-03T00:00:00Z",
            payload=payload,
        )

    def test_deltas_typewrite_once_and_reply_is_not_repeated(self) -> None:
        renderer, out = self._renderer()

        renderer.render_event(
            self._event("message.delta", {"channel": "text", "text": "你好"}, 1)
        )
        renderer.render_event(
            self._event("message.delta", {"channel": "text", "text": "世界"}, 2)
        )
        renderer.render_event(
            self._event("message.completed", {"iteration": 1, "content": "你好世界"}, 3)
        )
        renderer.print_reply("你好世界", run_id="r_test")

        output = out.getvalue()
        self.assertEqual(output.count("你好世界"), 1)
        self.assertIn("MiniBot › 你好世界", output.replace("\n", ""))

    def test_reasoning_deltas_are_not_rendered(self) -> None:
        renderer, out = self._renderer()

        renderer.render_event(
            self._event("message.delta", {"channel": "reasoning", "text": "思考中"}, 1)
        )

        self.assertEqual(out.getvalue(), "")

    def test_tool_event_breaks_the_stream_line(self) -> None:
        renderer, out = self._renderer()

        renderer.render_event(
            self._event("message.delta", {"channel": "text", "text": "先查一下"}, 1)
        )
        renderer.render_event(
            self._event(
                "tool_call.started",
                {"tool_call_id": "c1", "tool": "read_file", "display_name": "read_file", "args": {}},
                2,
            )
        )

        output = out.getvalue()
        self.assertIn("先查一下\n", output)
        self.assertIn("工具: read_file", output)

    def test_unstreamed_reply_still_prints(self) -> None:
        renderer, out = self._renderer()

        renderer.print_reply("plain answer", run_id="r_test")

        self.assertIn("MiniBot › plain answer", out.getvalue())


class CliReasoningPreviewTests(unittest.TestCase):
    def _renderer(self, *, preview_enabled: bool = True) -> tuple[CliRenderer, io.StringIO]:
        out = io.StringIO()
        renderer = CliRenderer(no_color=True, stdout=out)
        renderer._reasoning.enabled = preview_enabled
        return renderer, out

    def _delta(self, channel: str, text: str, seq: int) -> RuntimeEvent:
        return RuntimeEvent(
            id=f"r_test:{seq}",
            run_id="r_test",
            session_id="s_test",
            seq=seq,
            type="message.delta",
            created_at="2026-07-05T00:00:00Z",
            payload={"iteration": 1, "channel": channel, "text": text},
        )

    def test_reasoning_previews_then_collapses_when_answer_starts(self) -> None:
        renderer, out = self._renderer()

        renderer.render_event(self._delta("reasoning", "先想想这个问题\n", 1))
        renderer.render_event(self._delta("reasoning", "应该回答你好\n", 2))
        renderer.render_event(self._delta("text", "你好", 3))

        output = out.getvalue()
        self.assertIn("思考中", output)
        self.assertIn("先想想这个问题", output)
        self.assertIn("已思考", output)
        # The erase sequence proves the region was collapsed, not appended.
        self.assertIn("\x1b[", output)
        self.assertIn("MiniBot › 你好", output.replace("\n", "").split("已思考")[-1])
        self.assertFalse(renderer._reasoning.active)

    def test_disabled_preview_prints_summary_but_never_reasoning_text(self) -> None:
        renderer, out = self._renderer(preview_enabled=False)

        renderer.render_event(self._delta("reasoning", "内部推理内容", 1))
        renderer.render_event(self._delta("text", "答案", 2))

        output = out.getvalue()
        self.assertNotIn("内部推理内容", output)
        self.assertIn("已思考", output)
        self.assertIn("答案", output)

    def test_tool_event_collapses_reasoning_first(self) -> None:
        renderer, out = self._renderer()

        renderer.render_event(self._delta("reasoning", "需要查文件", 1))
        renderer.render_event(
            RuntimeEvent(
                id="r_test:2",
                run_id="r_test",
                session_id="s_test",
                seq=2,
                type="tool_call.started",
                created_at="2026-07-05T00:00:00Z",
                payload={
                    "tool_call_id": "c1",
                    "tool": "read_file",
                    "display_name": "read_file",
                    "args": {},
                },
            )
        )

        output = out.getvalue()
        self.assertLess(output.index("已思考"), output.index("工具: read_file"))
        self.assertFalse(renderer._reasoning.active)

    def test_truncate_display_counts_cjk_as_double_width(self) -> None:
        from minibot.cli import _truncate_display

        self.assertEqual(_truncate_display("中文很宽", 6), "中文…")
        self.assertEqual(_truncate_display("ascii", 10), "ascii")


if __name__ == "__main__":
    unittest.main()
