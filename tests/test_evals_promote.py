"""Trace → golden case: draft from a recorded trace, then replay it.

The "recorded trace" is real: a sandboxed turn's events go through the
production ``LangfuseFold`` into a fake client, and the observations it
records are reshaped like the v2 observations API returns them (input and
output as raw JSON strings). No network, no real model.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.config import Config
from minibot.evals import __main__ as cli
from minibot.evals import sandbox
from minibot.evals.promote import draft_case, format_traces, summarize_traces
from minibot.evals.runner import load_cases
from minibot.evals.sandbox import run_case
from minibot.evals.scoring import score_deterministic
from minibot.langfuse_tracing import LangfuseFold
from minibot.llm import LLMClient, LLMResponse, ToolCall
from minibot.runtime.messages import ModelMessage
from minibot.tools.definitions import ModelToolDefinition

CREATE = "mcp__macos_system__reminders_create"
TODAY = date.today()
DUE = f"{TODAY + timedelta(days=1):%Y-%m-%d}T09:00:00"
T0 = datetime(2026, 10, 1, 1, 0, tzinfo=UTC)


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[list[ModelMessage]] = []

    def chat(
        self,
        messages: list[ModelMessage],
        tools: list[ModelToolDefinition] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.requests.append(list(messages))
        return self._responses.pop(0)


def _call(call_id: str, tool: str, **args) -> ToolCall:
    return ToolCall(id=call_id, name=tool, arguments=json.dumps(args, ensure_ascii=False))


def _sandbox_env():
    return mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "MINIBOT_LANGFUSE": "0"})


class _Recorder:
    """A fake Langfuse client for LangfuseFold that keeps v2-shaped rows."""

    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []

    @contextmanager
    def propagate_attributes(self, **attributes):
        yield

    def start_observation(self, *, parent=None, **kwargs):
        row = SimpleNamespace(
            id=f"o{len(self.rows) + 1}",
            trace_id="trace-abc12345",
            parent_observation_id=parent.id if parent else None,
            type=str(kwargs.get("as_type", "span")).upper(),
            name=kwargs.get("name"),
            start_time=T0 + timedelta(seconds=len(self.rows)),
            input=_raw(kwargs.get("input")),
            output=None,
            metadata=kwargs.get("metadata") or {},
            level="DEFAULT",
            status_message=None,
        )
        self.rows.append(row)
        recorder = self

        class _Handle:
            id = row.id

            def start_observation(self, **child):
                return recorder.start_observation(parent=row, **child)

            def create_event(self, *, name, metadata=None, level=None):
                recorder.start_observation(parent=row, name=name, as_type="event", metadata=metadata)
                recorder.rows[-1].level = level or "DEFAULT"

            def update(self, **changes):
                if changes.get("output") is not None:
                    row.output = _raw(changes["output"])
                if changes.get("level"):
                    row.level = changes["level"]
                if isinstance(changes.get("metadata"), dict):
                    row.metadata = {**row.metadata, **changes["metadata"]}
                return self

            def end(self):
                pass

        return _Handle()


def _raw(value):
    # The v2 API returns input/output as strings: JSON for structures.
    return value if value is None or isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _record(case_input: dict, llm: LLMClient) -> tuple[list[SimpleNamespace], dict]:
    """Run a sandboxed turn and return its trace as the API would."""
    captured: list = []
    real = sandbox.summarize_events

    def capture(events):
        captured.extend(events)
        return real(events)

    with _sandbox_env(), mock.patch.object(sandbox, "summarize_events", side_effect=capture):
        output = run_case(case_input, config=Config(model="deepseek-v4-pro"), llm=llm)
    recorder = _Recorder()
    fold = LangfuseFold(recorder, propagate_attributes=recorder.propagate_attributes)
    for event in captured:
        fold(event)
    return recorder.rows, output


def _recovery_script() -> list[LLMResponse]:
    return [
        LLMResponse(
            content="",
            tool_calls=[_call("c1", "read_skill", name="reminders"), _call("c2", "read_file", path="notes.txt")],
        ),
        LLMResponse(content="", tool_calls=[_call("c3", CREATE, title="给妈妈打电话", due_at=DUE)]),
        LLMResponse(content="", tool_calls=[_call("c4", "exec", command="open -a Reminders")]),
        LLMResponse(content="", tool_calls=[_call("c5", CREATE, title="给妈妈打电话", due_at=DUE)]),
        LLMResponse(
            content="",
            tool_calls=[
                _call("c6", "web_search", query="提醒事项 用法"),
                _call("c7", "fetch_url", url="https://example.com/a"),
            ],
        ),
        LLMResponse(content="已创建提醒：明天 9:00 给妈妈打电话。"),
    ]


_ORIGINAL_INPUT = {
    "prompt": "明天早上九点提醒我给妈妈打电话。",
    "files": {"notes.txt": "妈妈电话 123\n"},
    "memory": ["用户在星海科技工作。"],
    "tool_results": {
        CREATE: {
            "sequence": [
                {"error": "Reminders got an error: Application isn’t running. (-600)"},
                {"id": "r-1", "title": "给妈妈打电话"},
            ]
        },
        "exec": {"stdout": "opened\n", "stderr": "", "exit_code": 0},
        "web_search": [{"title": "Reminders", "url": "https://example.com/a", "snippet": "s"}],
        "fetch_url": {"https://example.com/a": "page body"},
    },
}


class DraftFromTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_llm = _ScriptedLLM(_recovery_script())
        self.rows, self.original = _record(_ORIGINAL_INPUT, self.original_llm)

    def test_draft_reconstructs_prompt_state_and_recorded_results(self) -> None:
        draft = draft_case(self.rows, trace_id="trace-abc12345")
        case = draft.case

        self.assertEqual(case["id"], "promoted-trace-ab")
        self.assertEqual(case["source"], {"trace_id": "trace-abc12345"})
        self.assertEqual(draft.reply, "已创建提醒：明天 9:00 给妈妈打电话。")
        self.assertFalse(draft.run_failed)
        case_input = case["input"]
        self.assertEqual(case_input["prompt"], _ORIGINAL_INPUT["prompt"])
        self.assertEqual(case_input["files"], _ORIGINAL_INPUT["files"])
        self.assertEqual(case_input["memory"], _ORIGINAL_INPUT["memory"])
        # Recorded results in the sandbox's own format, failure first.
        self.assertEqual(case_input["tool_results"], _ORIGINAL_INPUT["tool_results"])
        self.assertNotIn("approval", case_input)

        expected = case["expected"]
        self.assertEqual(expected["skill"], "reminders")
        self.assertEqual(
            expected["tools_required"], ["read_file", CREATE, "exec", "web_search", "fetch_url"]
        )
        self.assertEqual(
            expected["args"][CREATE],
            {"title": {"contains": "给妈妈打电话"}, "due_at": {"date": "+1d", "time": "09:00"}},
        )
        self.assertEqual(expected["args"]["exec"], {"command": {"contains": "open -a Reminders"}})
        self.assertEqual(expected["args"]["read_file"], {"path": {"contains": "notes.txt"}})
        self.assertIn("有失败的工具调用", " ".join(draft.warnings))

    def test_replaying_the_draft_reproduces_the_run(self) -> None:
        draft = draft_case(self.rows, trace_id="trace-abc12345")
        replay_llm = _ScriptedLLM(_recovery_script())

        with _sandbox_env():
            replay = run_case(draft.case["input"], config=Config(model="deepseek-v4-pro"), llm=replay_llm)

        self.assertEqual(
            [(c["name"], c["ok"]) for c in replay["tool_calls"]],
            [(c["name"], c["ok"]) for c in self.original["tool_calls"]],
        )
        # The model saw byte-identical tool results in the replay.
        def tool_messages(llm):
            return [m.content for m in llm.requests[-1] if m.role == "tool"]

        self.assertEqual(tool_messages(replay_llm), tool_messages(self.original_llm))
        scores = score_deterministic(replay, draft.case["expected"], today=TODAY)
        self.assertEqual({s.name: s.value for s in scores},
                         {"skill_selection": 1.0, "tool_selection": 1.0, "tool_args": 1.0})

    def test_denied_approval_and_no_tool_turns(self) -> None:
        llm = _ScriptedLLM(
            [
                LLMResponse(content="", tool_calls=[_call("c1", "exec", command="rm -rf .")]),
                LLMResponse(content="没有执行删除。"),
            ]
        )
        rows, _ = _record({"prompt": "删掉所有文件", "approval": "deny"}, llm)
        draft = draft_case(rows, trace_id="t-deny", case_id="deny-case")

        self.assertEqual(draft.case["id"], "deny-case")
        self.assertEqual(draft.case["input"]["approval"], "deny")
        # A denied call never ran, so there is nothing to replay for it.
        self.assertNotIn("tool_results", draft.case["input"])
        self.assertEqual(draft.case["expected"]["tools_required"], ["exec"])

        rows, _ = _record({"prompt": "hi"}, _ScriptedLLM([LLMResponse(content="你好")]))
        self.assertEqual(draft_case(rows, trace_id="t-chat").case["expected"], {"tools_forbidden": ["*"]})

    def test_unreplayable_parts_become_warnings(self) -> None:
        def obs(id, type, name, parent="a", **kw):
            return SimpleNamespace(
                id=id, type=type, name=name, parent_observation_id=parent,
                start_time=T0 + timedelta(seconds=int(id[1:]) if id[1:].isdigit() else 0),
                input=kw.get("input"), output=kw.get("output"), metadata=kw.get("metadata") or {},
                level=kw.get("level", "DEFAULT"),
            )

        system = "当前工作目录: /work/proj\n- id: mem_3; fact: 喜欢咖啡\n- id: mem_7; fact: 住在上海"
        request = [{"role": "system", "content": system}, {"role": "user", "content": "q\n\n[发送时间 2026-10-01 10:00 星期四 UTC+08:00]"}]
        envelope = {"ok": True, "code": "success", "summary": "", "data": {"content": "x"}, "truncated": False}
        rows = [
            obs("a", "AGENT", "minibot.turn", parent=None, input="q", level="ERROR"),
            obs("g1", "GENERATION", "model", input=json.dumps(request), output=json.dumps({
                "tool_calls": [{"id": "c1", "name": "read_file"}, {"id": "c2", "name": "mcp__github__search"},
                               {"id": "c3", "name": "forget"}]})),
            obs("t1", "TOOL", "read_file", input='{"path": "/etc/hosts"}', metadata={"tool_call_id": "c1"}),
            obs("t2", "TOOL", "mcp__github__search", input='{"q": "x"}', metadata={"tool_call_id": "c2"}),
            obs("t3", "TOOL", "forget", input='{"memory_id": "mem_7"}', metadata={"tool_call_id": "c3"}),
            obs("g2", "GENERATION", "model", input=json.dumps(request + [
                {"role": "tool", "tool_call_id": "c1", "content": json.dumps(envelope)}])),
            # Outside the turn (e.g. an experiment's evaluator span): ignored.
            obs("x1", "TOOL", "exec", parent="elsewhere", input='{"command": "ls"}'),
        ]

        draft = draft_case(rows, trace_id="t-edge")

        warnings = " ".join(draft.warnings)
        self.assertTrue(draft.run_failed)
        self.assertIn("/etc/hosts", warnings)
        self.assertIn("mcp__github__search", warnings)
        self.assertIn("重新编号", warnings)
        self.assertEqual(draft.case["input"]["memory"], ["喜欢咖啡", "住在上海"])
        # mem_7 is the second fact, so it is mem_2 in the sandbox.
        self.assertEqual(draft.case["expected"]["args"]["forget"], {"memory_id": {"equals": "mem_2"}})
        self.assertNotIn("exec", draft.case["expected"]["tools_required"])
        self.assertNotIn("files", draft.case["input"])

        with self.assertRaises(ValueError):
            draft_case([obs("g1", "GENERATION", "model")], trace_id="t-none")


class TraceListingTests(unittest.TestCase):
    def test_summarize_flags_failed_runs_and_failed_tools(self) -> None:
        agents = [
            SimpleNamespace(trace_id="t1", start_time=T0, level="DEFAULT", tags=["tui"], input="你好"),
            SimpleNamespace(trace_id="t2", start_time=T0 + timedelta(minutes=1), level="ERROR",
                            tags=["cli"], input="帮我" * 40),
        ]
        tools = [SimpleNamespace(trace_id="t1"), SimpleNamespace(trace_id="t1")]

        rows = summarize_traces(agents, tools)

        self.assertEqual([r["trace_id"] for r in rows], ["t2", "t1"])  # newest first
        self.assertTrue(rows[0]["run_failed"])
        self.assertEqual(rows[1]["failed_tools"], 2)
        text = format_traces(rows)
        self.assertIn("运行失败", text)
        self.assertIn("2 个工具失败", text)
        self.assertIn("…", text)


class PromoteCommandTests(unittest.TestCase):
    def test_promote_appends_to_the_given_cases_file_and_syncs(self) -> None:
        rows, _ = _record(_ORIGINAL_INPUT, _ScriptedLLM(_recovery_script()))
        page = SimpleNamespace(data=rows, meta=SimpleNamespace(cursor=None))
        langfuse = mock.MagicMock()
        langfuse.api.observations.get_many.return_value = page
        langfuse.get_trace_url.return_value = "https://lf/trace/trace-abc12345"
        langfuse.get_dataset.return_value = SimpleNamespace(items=[])
        judge = _ScriptedLLM([LLMResponse(content="「确认已创建明天 9:00 给妈妈打电话的提醒。」")])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(load_cases(), ensure_ascii=False), encoding="utf-8")
            with (
                mock.patch.object(cli, "load_env"),
                mock.patch.object(cli, "langfuse_configured", return_value=True),
                mock.patch("langfuse.get_client", return_value=langfuse),
                mock.patch.object(cli, "build_judge", return_value=judge),
                mock.patch("builtins.print"),
            ):
                code = cli.main(["promote", "trace-abc12345", "--id", "mom-call", "--yes", "--cases", str(path)])

            saved = load_cases(path=path)

        self.assertEqual(code, 0)
        case = saved[-1]
        self.assertEqual(case["id"], "mom-call")
        self.assertEqual(case["expected"]["answer_rubric"], "确认已创建明天 9:00 给妈妈打电话的提醒。")
        self.assertEqual(case["source"]["trace_url"], "https://lf/trace/trace-abc12345")
        synced = {c.kwargs["id"] for c in langfuse.create_dataset_item.call_args_list}
        self.assertIn("minibot-golden--mom-call", synced)
        # The real cases.json is untouched.
        self.assertNotIn("mom-call", {c["id"] for c in load_cases()})



class RunDateLabelTests(unittest.TestCase):
    def test_both_stamp_labels_give_the_run_day(self) -> None:
        from minibot.evals.promote import _LOCAL_DATE_RE

        for text in ("q\n\n[发送时间 2026-10-01 10:00 星期四 UTC+08:00]", "q\n\n[当前本地时间 2026-10-01 10:00 星期四]"):
            with self.subTest(text=text):
                self.assertEqual(_LOCAL_DATE_RE.search(text).group(1), "2026-10-01")

if __name__ == "__main__":
    unittest.main()
