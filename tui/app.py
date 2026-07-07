"""The Textual TUI: a fifth caller of AgentSession, a new event subscriber.

The runtime stays synchronous and untouched. Turns run in a worker thread;
runtime events are marshalled onto the UI thread with ``call_from_thread``.
Deltas buffer and flush on a 10Hz timer so streaming Markdown stays cheap.
Approval blocks the worker on a ``threading.Event`` while a modal asks the
user — the same rendezvous shape as the web ApprovalBroker.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Footer, Input, Label, Markdown, Static

from ..interaction.commands import CommandContext, dispatch_command
from ..run_log import make_run_id, preview_text
from ..runtime.agent_session import RunCancelled
from ..runtime.approval import ApprovalRequest
from ..runtime.events import RuntimeEvent

if TYPE_CHECKING:
    from ..bootstrap import MiniBotRuntime


class ApprovalModal(ModalScreen[bool]):
    """Ask the user to approve one sensitive tool call."""

    BINDINGS = [
        Binding("y", "approve", "批准"),
        Binding("n,escape", "deny", "拒绝"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        args = json.dumps(self._request.args, ensure_ascii=False, indent=2)
        yield Static(
            f"批准执行 [b]{self._request.tool_name}[/b] ?\n\n{preview_text(args, 600)}",
            id="approval-text",
        )
        with Horizontal(id="approval-actions"):
            yield Button("批准 (y)", id="approve", variant="success")
            yield Button("拒绝 (n)", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class MinibotApp(App):
    """Chat TUI over the MiniBot runtime."""

    TITLE = "MiniBot"

    CSS = """
    #header {
        dock: top;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #chat {
        padding: 0 1;
    }
    #prompt {
        dock: bottom;
    }
    .user-line {
        color: $accent;
        margin: 1 0 0 0;
    }
    .system-line {
        color: $text-muted;
    }
    .error-line {
        color: $error;
    }
    .reply {
        margin: 0;
    }
    Collapsible {
        border: none;
        padding: 0;
        margin: 0;
    }
    Collapsible > Contents {
        padding: 0 0 0 2;
    }
    ApprovalModal {
        align: center middle;
    }
    #approval-text {
        width: 70;
        max-height: 16;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    #approval-actions {
        width: 70;
        height: 3;
        align: center middle;
        background: $surface;
    }
    #approval-actions Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_run", "取消运行"),
        Binding("ctrl+n", "new_session", "新会话"),
        Binding("ctrl+d", "quit", "退出"),
    ]

    def __init__(self, runtime: "MiniBotRuntime") -> None:
        super().__init__()
        self.runtime = runtime
        self.session_id: str = ""
        self._run_id: str | None = None
        self._turn_running = False
        # Streaming state, only touched on the UI thread.
        self._reply_widget: Markdown | None = None
        self._reply_buffer = ""
        self._reply_dirty = False
        self._reasoning: Collapsible | None = None
        self._reasoning_body: Static | None = None
        self._reasoning_buffer = ""
        self._reasoning_dirty = False
        self._reasoning_started = 0.0
        self._tool_lines: dict[str, tuple[Collapsible, str]] = {}

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield VerticalScroll(id="chat")
        yield Input(placeholder="输入消息,/ 开头是命令,exit 退出…", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.runtime.approval_policy.handler = self._approval_handler
        session, resumed = self.runtime.manager.startup_session()
        self.session_id = session.session_id
        self._refresh_header("ready")
        state = "已恢复" if resumed else "新会话"
        self._system_line(f"{state} {session.session_id} · /help 查看命令")
        self.set_interval(0.1, self._flush_stream_buffers)
        self.query_one(Input).focus()

    # ── input & commands ──────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one(Input).value = ""
        if not text:
            return
        if self._turn_running:
            self._system_line("当前有运行中的 turn,按 Esc 取消后再发。")
            return
        result = dispatch_command(text, self.session_id, self._command_context())
        if result.handled:
            if result.current_session_id:
                self.session_id = result.current_session_id
            for notice in result.notices:
                body = f"{notice.title}" + (f"\n{notice.body}" if notice.body else "")
                self._system_line(body, error=notice.kind == "error")
            self._refresh_header("ready")
            if result.should_exit:
                self.exit()
            return
        self._add_user_line(text)
        self._run_turn(text)

    def _command_context(self) -> CommandContext:
        return CommandContext(
            sessions=self.runtime.manager,
            compactor=self.runtime.compactor,
            context_builder=self.runtime.context_builder,
            memory_store=self.runtime.memory_store,
            approval_policy=self.runtime.approval_policy,
            mcp_host=self.runtime.mcp_host,
            config=self.runtime.config,
            schedule_store=self.runtime.schedule_store,
        )

    # ── the turn worker ───────────────────────────────────────────

    @work(thread=True)
    def _run_turn(self, text: str) -> None:
        run_id = make_run_id()
        self._run_id = run_id
        self._turn_running = True
        try:
            self.runtime.agent_session.prompt(
                self.session_id,
                text,
                run_id=run_id,
                event_handler=self._on_event_from_worker,
            )
        except RunCancelled:
            self.call_from_thread(self._system_line, "已取消。")
        except Exception as exc:
            self.call_from_thread(
                self._system_line, f"运行失败: {exc}", True
            )
        finally:
            self._turn_running = False
            self._run_id = None
            self.call_from_thread(self._finish_turn)

    def _on_event_from_worker(self, event: RuntimeEvent) -> None:
        self.call_from_thread(self._handle_event, event)

    def action_cancel_run(self) -> None:
        if self._run_id is not None:
            self.runtime.agent_session.abort(self._run_id)
            self._system_line("已请求取消…")

    def action_new_session(self) -> None:
        if self._turn_running:
            self._system_line("先等当前运行结束(或 Esc 取消)。")
            return
        session = self.runtime.manager.create_current_session()
        self.session_id = session.session_id
        self._system_line(f"已创建新会话 {session.session_id}")
        self._refresh_header("ready")

    # ── approval rendezvous (worker thread ↔ UI modal) ────────────

    def _approval_handler(
        self,
        request: ApprovalRequest,
        cancel_event: threading.Event | None,
    ) -> bool:
        done = threading.Event()
        decision = {"approved": False}

        def _ask() -> None:
            def _resolved(approved: bool | None) -> None:
                decision["approved"] = bool(approved)
                done.set()

            self.push_screen(ApprovalModal(request), _resolved)

        self.call_from_thread(_ask)
        while not done.wait(0.1):
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled("run cancelled while waiting for approval")
        return decision["approved"]

    # ── event → widgets ───────────────────────────────────────────

    def _handle_event(self, event: RuntimeEvent) -> None:
        payload = event.payload
        kind = event.type
        if kind == "message.delta":
            self._on_delta(payload)
        elif kind == "message.completed":
            self._finalize_reply(str(payload.get("content") or ""))
            self._refresh_header("ready")
        elif kind == "model.request.started":
            self._refresh_header(f"thinking · 第 {payload.get('iteration')} 轮")
        elif kind == "model.request.completed":
            self._finalize_reasoning()
            if int(payload.get("tool_call_count") or 0) > 0:
                # Tool calls follow: freeze this iteration's narration so the
                # next iteration streams into a fresh widget. The final reply
                # keeps its widget for message.completed to finalize in place.
                self._reset_reply_stream()
        elif kind == "model.request.retrying":
            self._refresh_header(
                f"retrying {payload.get('attempt')}/{payload.get('max_retries')}"
            )
        elif kind == "tool_call.started":
            self._on_tool_started(payload)
        elif kind in {"tool_call.completed", "tool_call.failed"}:
            self._on_tool_finished(payload, failed=kind == "tool_call.failed")
        elif kind == "context.compacted":
            self._system_line(f"已自动压缩: {payload.get('message') or ''}")
        elif kind == "approval.required":
            self._refresh_header(f"等待批准 · {payload.get('tool')}")
        elif kind == "run.failed":
            self._system_line(
                f"运行失败: {payload.get('error_type')}: {payload.get('message')}",
                error=True,
            )

    def _on_delta(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "")
        if not text:
            return
        if payload.get("channel") == "reasoning":
            if self._reasoning is None:
                self._reasoning_started = time.monotonic()
                self._reasoning_body = Static("")
                self._reasoning = Collapsible(
                    self._reasoning_body, title="⋯ 思考中", collapsed=True
                )
                self._mount(self._reasoning)
            self._reasoning_buffer += text
            self._reasoning_dirty = True
            return
        if payload.get("channel") != "text":
            return
        self._finalize_reasoning()
        if self._reply_widget is None:
            self._reply_widget = Markdown("", classes="reply")
            self._mount(self._reply_widget)
            self._refresh_header("replying")
        self._reply_buffer += text
        self._reply_dirty = True

    def _flush_stream_buffers(self) -> None:
        if self._reply_dirty and self._reply_widget is not None:
            self._reply_widget.update(self._reply_buffer)
            self._reply_dirty = False
            self._scroll_end()
        if self._reasoning_dirty and self._reasoning_body is not None:
            tail = self._reasoning_buffer[-1500:]
            self._reasoning_body.update(tail)
            if self._reasoning is not None:
                elapsed = time.monotonic() - self._reasoning_started
                self._reasoning.title = f"⋯ 思考中 · {elapsed:.0f}s"
            self._reasoning_dirty = False
            self._scroll_end()

    def _finalize_reasoning(self) -> None:
        if self._reasoning is None:
            return
        elapsed = time.monotonic() - self._reasoning_started
        self._reasoning.title = f"⋯ 已思考 · {elapsed:.1f}s(点击展开)"
        if self._reasoning_body is not None:
            self._reasoning_body.update(self._reasoning_buffer)
        self._reasoning = None
        self._reasoning_body = None
        self._reasoning_buffer = ""
        self._reasoning_dirty = False

    def _reset_reply_stream(self) -> None:
        """An iteration ended (tool calls follow): freeze this reply widget."""
        if self._reply_widget is not None and self._reply_buffer:
            self._reply_widget.update(self._reply_buffer)
        self._reply_widget = None
        self._reply_buffer = ""
        self._reply_dirty = False

    def _finalize_reply(self, authoritative: str) -> None:
        if self._reply_widget is None:
            self._reply_widget = Markdown("", classes="reply")
            self._mount(self._reply_widget)
        self._reply_widget.update(authoritative)
        self._reply_widget = None
        self._reply_buffer = ""
        self._reply_dirty = False
        self._scroll_end()

    def _on_tool_started(self, payload: dict[str, Any]) -> None:
        call_id = str(payload.get("tool_call_id") or "")
        label = payload.get("display_name") or payload.get("tool") or "tool"
        args = json.dumps(payload.get("args") or {}, ensure_ascii=False)
        body = Static(f"args: {preview_text(args, 500)}")
        line = Collapsible(body, title=f"⚙ {label} · 运行中…", collapsed=True)
        self._tool_lines[call_id] = (line, str(label))
        self._mount(line)

    def _on_tool_finished(self, payload: dict[str, Any], *, failed: bool) -> None:
        call_id = str(payload.get("tool_call_id") or "")
        entry = self._tool_lines.pop(call_id, None)
        if entry is None:
            return
        line, label = entry
        mark = "✗" if failed else "✓"
        summary = preview_text(str(payload.get("summary") or ""), 80)
        line.title = f"{mark} {label} · {summary}"
        self._refresh_header("thinking")

    def _finish_turn(self) -> None:
        self._finalize_reasoning()
        self._reset_reply_stream()
        for line, label in self._tool_lines.values():
            line.title = f"✗ {label} · 未完成"
        self._tool_lines.clear()
        self._refresh_header("ready")

    # ── small UI helpers ──────────────────────────────────────────

    def _add_user_line(self, text: str) -> None:
        self._mount(Static(f"› {text}", classes="user-line"))

    def _system_line(self, text: str, error: bool = False) -> None:
        classes = "error-line" if error else "system-line"
        self._mount(Static(text, classes=classes))

    def _mount(self, widget) -> None:
        self.query_one("#chat", VerticalScroll).mount(widget)
        self._scroll_end()

    def _scroll_end(self) -> None:
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    def _refresh_header(self, state: str) -> None:
        summary = self.runtime.mcp_host.summary()
        mcp = (
            "mcp off"
            if summary.enabled_servers == 0
            else f"mcp {summary.connected_servers}/{summary.enabled_servers}"
        )
        self.query_one("#header", Static).update(
            f"[b magenta]MiniBot[/] · {self.runtime.config.model} · "
            f"{self.session_id} · {mcp} · "
            f"approval {self.runtime.approval_policy.mode} · [dim]{state}[/]"
        )


def run_tui(runtime: "MiniBotRuntime") -> None:
    MinibotApp(runtime).run()
