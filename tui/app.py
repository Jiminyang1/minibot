"""The Textual TUI: a fifth caller of AgentSession, a new event subscriber.

The runtime stays synchronous and untouched. Turns run in a worker thread;
runtime events cross into the UI through a thread-safe queue. One edge-triggered
message wakes the UI, while streaming paints are coalesced to 60Hz so model
deltas neither wait for a polling tick nor flood Textual's message pump.
Approval blocks the worker on a ``threading.Event`` while a modal asks the
user — the same rendezvous shape as the web ApprovalBroker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import threading
import time
from typing import TYPE_CHECKING, Any, TypedDict

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Markdown, Static

from ..interaction.commands import (
    CommandContext,
    CommandDef,
    command_defs,
    dispatch_command,
)
from ..run_log import make_run_id, preview_text
from ..runtime.agent_session import RunCancelled
from ..runtime.approval import ApprovalRequest
from ..runtime.events import RuntimeEvent

if TYPE_CHECKING:
    from ..bootstrap import MiniBotRuntime


_UI_FRAME_INTERVAL_SECONDS = 0.016
_MAX_EVENTS_PER_TICK = 512


class _SlashCommandItem(TypedDict):
    display: str
    completion: str
    description: str
    definition: CommandDef


@dataclass(frozen=True)
class _ParamOption:
    display: str
    value: str


class RuntimeEventsReady(Message):
    """Wake the UI once after one or more worker events arrive."""


class _StreamingMarkdown(Markdown):
    """A Markdown document that drops superseded parses while streaming.

    Textual parses Markdown asynchronously and serializes concurrent updates.
    Calling ``update`` for every frame can therefore build an unbounded queue.
    This wrapper keeps only the newest source while one parse is in flight.
    """

    def __init__(
        self,
        *,
        classes: str,
    ) -> None:
        super().__init__("", classes=classes)
        self._rendered_source = ""
        self._pending_source: str | None = None
        self._update_task: asyncio.Task[None] | None = None

    def set_source(self, source: str) -> None:
        """Render the latest source without queueing obsolete intermediate text."""
        if self._pending_source is not None:
            if self._pending_source == source:
                return
        elif source == self._rendered_source:
            return

        self._pending_source = source
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(
                self._drain_updates(),
                name="minibot-streaming-markdown",
            )

    async def _drain_updates(self) -> None:
        try:
            while self._pending_source is not None:
                source = self._pending_source
                self._pending_source = None
                await super().update(source)
                self._rendered_source = source
        finally:
            self._update_task = None

    def on_unmount(self) -> None:
        if self._update_task is not None and not self._update_task.done():
            self._update_task.cancel()


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
    Screen {
        background: #18181e;
        color: #d4d4d4;
    }

    #header {
        dock: top;
        height: 1;
        background: #18181e;
        color: #808080;
        padding: 0 1;
    }

    #chat {
        background: #18181e;
        padding: 1 1 0 1;
    }

    #composer {
        dock: bottom;
        height: 5;
        background: #18181e;
        padding: 0 1 1 1;
    }

    #composer.has-menu {
        height: 14;
    }

    #prompt {
        height: 3;
        background: #1e1e24;
        color: #d4d4d4;
        border: solid #505050;
        padding: 0 1;
    }

    #prompt:focus {
        border: solid #00d7ff;
    }

    #slash-menu {
        display: none;
        max-height: 8;
        margin: 0 0 1 0;
        padding: 0 1;
        background: #1e1e24;
        border-left: solid #00d7ff;
    }

    #slash-menu.visible {
        display: block;
    }

    #footer-bar {
        height: 1;
        color: #666666;
        padding: 0 1;
    }

    .user-line {
        margin: 1 0 0 0;
        padding: 1 2;
        background: #343541;
        color: #d4d4d4;
        border-left: solid #8abeb7;
    }

    .system-line {
        margin: 1 0 0 0;
        padding: 0 1;
        color: #808080;
        border-left: solid #505050;
    }

    .error-line {
        margin: 1 0 0 0;
        padding: 0 1;
        color: #cc6666;
        border-left: solid #cc6666;
    }

    .reply {
        margin: 1 0 0 0;
        padding: 0 1 0 2;
        border-left: solid #5f87ff;
    }

    .reply-streaming {
        color: #d4d4d4;
    }

    .thinking-line {
        margin: 1 0 0 0;
        color: #808080;
    }

    .tool-line {
        margin: 1 0 0 0;
        padding: 0 1;
        background: #282832;
        color: #d4d4d4;
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
        background: #18181e 80%;
    }

    #approval-text {
        width: 70;
        max-height: 16;
        border: solid #ffff00;
        padding: 1 2;
        background: #1e1e24;
        color: #d4d4d4;
    }

    #approval-actions {
        width: 70;
        height: 3;
        align: center middle;
        background: #1e1e24;
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
        self._run_failed_rendered = False
        # Streaming state, only touched on the UI thread.
        self._reply_widget: _StreamingMarkdown | None = None
        self._reply_buffer = ""
        self._pending_reply_chunks: list[str] = []
        self._reply_dirty = False
        self._reasoning: Collapsible | None = None
        self._reasoning_body: Static | None = None
        self._reasoning_buffer = ""
        self._pending_reasoning_chunks: list[str] = []
        self._reasoning_dirty = False
        self._reasoning_started = 0.0
        self._tool_lines: dict[str, tuple[Collapsible, str]] = {}
        # Runtime callbacks happen on the worker thread. Queue them and post at
        # most one thread-safe wakeup until the UI has drained the current burst.
        self._event_queue: SimpleQueue[RuntimeEvent] = SimpleQueue()
        self._event_wakeup_lock = threading.Lock()
        self._event_wakeup_pending = False
        self._frame_scheduled = False
        self._last_frame_at = 0.0
        self._state = "ready"
        self._slash_commands = tuple(
            _SlashCommandItem(
                display=cmd.display,
                completion=_command_completion(cmd.display),
                description=cmd.description,
                definition=cmd,
            )
            for cmd in command_defs()
        )
        self._slash_matches: list[_SlashCommandItem] = []
        self._slash_index = 0
        # ── two-level slash menu ──────────────────────────────────
        self._param_mode = False
        self._param_options: list[_ParamOption] = []
        self._param_index = 0
        self._active_cmd_def: CommandDef | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield VerticalScroll(id="chat")
        with Container(id="composer"):
            yield Static("", id="slash-menu")
            yield Input(placeholder="Message MiniBot  (/help, Esc cancel)", id="prompt")
            yield Static(id="footer-bar")

    def on_mount(self) -> None:
        self.runtime.approval_policy.handler = self._approval_handler
        session, resumed = self.runtime.manager.startup_session()
        self.session_id = session.session_id
        self._refresh_header("ready")
        state = "已恢复" if resumed else "新会话"
        self._system_line(f"{state} {session.session_id} · /help 查看命令")
        self.query_one(Input).focus()

    # ── input & commands ──────────────────────────────────────────

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._slash_matches:
            return
        if event.key == "down":
            if self._param_mode:
                self._param_index = (self._param_index + 1) % len(self._param_options)
            else:
                self._slash_index = (self._slash_index + 1) % len(self._slash_matches)
            self._render_slash_menu()
            event.prevent_default()
            event.stop()
            return
        if event.key == "up":
            if self._param_mode:
                self._param_index = (self._param_index - 1) % len(self._param_options)
            else:
                self._slash_index = (self._slash_index - 1) % len(self._slash_matches)
            self._render_slash_menu()
            event.prevent_default()
            event.stop()
            return
        if event.key == "tab":
            self._complete_slash_selection()
            event.prevent_default()
            event.stop()
            return
        if event.key == "enter" and self._should_complete_slash_on_enter():
            self._complete_slash_selection()
            event.prevent_default()
            event.stop()
            return
        if event.key == "escape":
            self._hide_slash_menu()
            event.prevent_default()
            event.stop()
            return

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        self._update_slash_menu(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self._slash_matches and self._should_complete_slash_on_enter():
            self._complete_slash_selection()
            return
        self._hide_slash_menu()
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
        self._refresh_header("starting")
        self._start_turn(text)

    def _command_context(self) -> CommandContext:
        return CommandContext(
            sessions=self.runtime.manager,
            compactor=self.runtime.compactor,
            context_builder=self.runtime.context_builder,
            memory_store=self.runtime.memory_store,
            approval_policy=self.runtime.approval_policy,
            mcp_host=self.runtime.mcp_host,
            config=self.runtime.config,
            budget=getattr(self.runtime, "budget", None),
            schedule_store=self.runtime.schedule_store,
        )

    # ── slash command completion ──────────────────────────────────

    def _update_slash_menu(self, value: str) -> None:
        if self._turn_running:
            self._hide_slash_menu()
            return
        query = value.strip()
        if not query.startswith("/"):
            self._hide_slash_menu()
            return

        lowered = query.lower()
        matches = [
            command
            for command in self._slash_commands
            if command["display"].lower().startswith(lowered)
            or command["completion"].lower().startswith(lowered)
        ]
        if not matches:
            self._hide_slash_menu()
            return
        self._slash_matches = matches[:8]
        self._param_mode = False
        self._param_options = []
        self._param_index = 0
        self._active_cmd_def = None
        self._slash_index = min(self._slash_index, len(self._slash_matches) - 1)
        self._render_slash_menu()

    def _render_slash_menu(self) -> None:
        menu = self.query_one("#slash-menu", Static)
        if not self._slash_matches and not self._param_mode:
            self._hide_slash_menu()
            return
        if self._param_mode:
            self._render_param_menu(menu)
            return
        width = max(len(item["display"]) for item in self._slash_matches) + 2
        lines: list[str] = []
        for index, item in enumerate(self._slash_matches):
            selected = index == self._slash_index
            marker = "›" if selected else " "
            command = escape(item["display"].ljust(width))
            description = escape(item["description"])
            if selected:
                lines.append(
                    f"[#18181e on #3a3a4a]{marker} {command}[/]"
                    f"[#d4d4d4 on #3a3a4a]{description}[/]"
                )
            else:
                lines.append(f"[#8abeb7]{marker} {command}[/][#808080]{description}[/]")
        menu.update("\n".join(lines))
        menu.add_class("visible")
        self.query_one("#composer", Container).add_class("has-menu")

    def _hide_slash_menu(self) -> None:
        self._slash_matches = []
        self._slash_index = 0
        self._param_mode = False
        self._param_options = []
        self._param_index = 0
        self._active_cmd_def = None
        menu = self.query_one("#slash-menu", Static)
        menu.update("")
        menu.remove_class("visible")
        self.query_one("#composer", Container).remove_class("has-menu")

    def _complete_slash_selection(self) -> None:
        selected = self._selected_slash_command()
        if selected is None:
            return
        if self._param_mode:
            self._commit_param_selection()
            return
        cmd_def = selected["definition"]
        if cmd_def.param_type != "none":
            self._enter_param_mode(cmd_def)
            return
        prompt = self.query_one(Input)
        prompt.value = selected["completion"]
        prompt.cursor_position = len(prompt.value)
        self._update_slash_menu(prompt.value)

    def _selected_slash_command(self) -> _SlashCommandItem | None:
        if not self._slash_matches:
            return None
        return self._slash_matches[self._slash_index]

    def _should_complete_slash_on_enter(self) -> bool:
        if self._param_mode:
            return True
        selected = self._selected_slash_command()
        if selected is None:
            return False
        value = self.query_one(Input).value.strip()
        completion = selected["completion"].strip()
        return bool(value and value != completion)

    # ── two-level slash menu: param mode ─────────────────────────

    def _enter_param_mode(self, cmd_def: CommandDef) -> None:
        self._active_cmd_def = cmd_def
        self._param_mode = True
        self._param_index = 0
        if cmd_def.param_type == "choice":
            self._param_options = [
                _ParamOption(display=option, value=option)
                for option in cmd_def.param_options
            ]
        elif cmd_def.param_type in ("sessions", "sessions_delete"):
            sessions = self.runtime.manager.list_sessions()
            options: list[_ParamOption] = []
            if cmd_def.param_type == "sessions_delete":
                options.append(_ParamOption(display="current", value="current"))
            for s in sessions[:10]:
                short = s.session_id[:8]
                title = s.title or ""
                display = f"{short}  {title}" if title else short
                options.append(_ParamOption(display=display, value=s.session_id))
            self._param_options = options
        elif cmd_def.param_type == "text":
            prompt = self.query_one(Input)
            prompt.value = _command_completion(cmd_def.display)
            prompt.cursor_position = len(prompt.value)
            self._hide_slash_menu()
            return
        self._render_slash_menu()

    def _commit_param_selection(self) -> None:
        if not self._param_options or self._active_cmd_def is None:
            return
        value = self._param_options[self._param_index].value
        prefix = _command_completion(self._active_cmd_def.display).strip()
        assembled = f"{prefix} {value}"
        prompt = self.query_one(Input)
        prompt.value = assembled
        prompt.cursor_position = len(prompt.value)
        self._hide_slash_menu()

    def _render_param_menu(self, menu: Static) -> None:
        if not self._param_options:
            self._hide_slash_menu()
            return
        if self._active_cmd_def is None:
            return
        header = f"[#808080]{escape(self._active_cmd_def.display)}[/]"
        max_width = max(
            max(len(option.display) for option in self._param_options),
            len(self._active_cmd_def.display),
        ) + 4
        lines: list[str] = [header]
        for index, option in enumerate(self._param_options):
            selected = index == self._param_index
            marker = "›" if selected else " "
            padded = escape(option.display.ljust(max_width))
            if selected:
                lines.append(f"[#18181e on #3a3a4a]{marker} {padded}[/]")
            else:
                lines.append(f"[#d4d4d4]{marker} {padded}[/]")
        menu.update("\n".join(lines))
        menu.add_class("visible")
        self.query_one("#composer", Container).add_class("has-menu")

    # ── the turn worker ───────────────────────────────────────────

    def _start_turn(self, text: str) -> None:
        """Mark the run active synchronously, then hand work to a thread."""
        run_id = make_run_id()
        self._run_id = run_id
        self._turn_running = True
        self._run_failed_rendered = False
        self._run_turn(text, run_id)

    @work(thread=True)
    def _run_turn(self, text: str, run_id: str) -> None:
        notice: str | None = None
        notice_is_error = False
        try:
            self.runtime.agent_session.prompt(
                self.session_id,
                text,
                run_id=run_id,
                event_handler=self._on_event_from_worker,
                source="tui",
            )
        except RunCancelled:
            notice = "已取消。"
        except Exception as exc:
            notice = f"运行失败: {exc}"
            notice_is_error = True
        finally:
            self.call_from_thread(
                self._complete_worker_turn,
                notice,
                notice_is_error,
            )

    def _on_event_from_worker(self, event: RuntimeEvent) -> None:
        self._event_queue.put(event)
        self._request_event_wakeup()

    def _request_event_wakeup(self) -> None:
        with self._event_wakeup_lock:
            if self._event_wakeup_pending:
                return
            self._event_wakeup_pending = True
        if not self.post_message(RuntimeEventsReady()):
            with self._event_wakeup_lock:
                self._event_wakeup_pending = False

    def _complete_worker_turn(
        self,
        notice: str | None,
        notice_is_error: bool,
    ) -> None:
        """Finish on the UI thread after consuming every queued event."""
        self._drain_event_queue()
        self._flush_stream_buffers()
        if notice is not None and not (
            notice_is_error and self._run_failed_rendered
        ):
            self._system_line(notice, error=notice_is_error)
        self._turn_running = False
        self._run_id = None
        self._finish_turn()

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

    def on_runtime_events_ready(self, _message: RuntimeEventsReady) -> None:
        with self._event_wakeup_lock:
            self._event_wakeup_pending = False
        processed = self._drain_event_queue(max_events=_MAX_EVENTS_PER_TICK)
        if self._reply_dirty or self._reasoning_dirty:
            self._request_frame()
        if processed == _MAX_EVENTS_PER_TICK:
            self._request_event_wakeup()

    def _request_frame(self) -> None:
        if self._frame_scheduled:
            return
        elapsed = time.monotonic() - self._last_frame_at
        delay = max(0.0, _UI_FRAME_INTERVAL_SECONDS - elapsed)
        if delay == 0.0:
            self._render_frame()
            return
        self._frame_scheduled = True
        self.set_timer(delay, self._render_frame, name="stream-frame")

    def _render_frame(self) -> None:
        self._frame_scheduled = False
        self._last_frame_at = time.monotonic()
        if next(iter(self.query("#chat")), None) is not None:
            self._flush_stream_buffers()

    def _flush_ui(self) -> None:
        """Synchronously drain a bounded batch; primarily useful for teardown/tests."""
        if next(iter(self.query("#chat")), None) is None:
            return
        processed = self._drain_event_queue(max_events=_MAX_EVENTS_PER_TICK)
        self._render_frame()
        if processed == _MAX_EVENTS_PER_TICK:
            self._request_event_wakeup()

    def _drain_event_queue(self, *, max_events: int | None = None) -> int:
        processed = 0
        while max_events is None or processed < max_events:
            try:
                event = self._event_queue.get_nowait()
            except Empty:
                break
            self._handle_event(event)
            processed += 1
        return processed

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
            self._run_failed_rendered = True
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
                    self._reasoning_body,
                    title="⋯ 思考中",
                    collapsed=True,
                    classes="thinking-line",
                )
                self._mount(self._reasoning)
            self._pending_reasoning_chunks.append(text)
            self._reasoning_dirty = True
            return
        if payload.get("channel") != "text":
            return
        self._finalize_reasoning()
        if self._reply_widget is None:
            self._reply_widget = _StreamingMarkdown(
                classes="reply reply-streaming",
            )
            self._mount(self._reply_widget)
            self._refresh_header("replying")
        self._pending_reply_chunks.append(text)
        self._reply_dirty = True

    def _flush_stream_buffers(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        follow_output = chat.is_vertical_scroll_end
        if follow_output:
            chat.anchor()
        if self._reply_dirty and self._reply_widget is not None:
            self._consume_pending_reply_chunks()
            self._reply_widget.set_source(self._reply_buffer)
            self._reply_dirty = False
        if self._reasoning_dirty and self._reasoning_body is not None:
            self._consume_pending_reasoning_chunks()
            tail = self._reasoning_buffer[-1500:]
            self._reasoning_body.update(tail)
            if self._reasoning is not None:
                elapsed = time.monotonic() - self._reasoning_started
                self._reasoning.title = f"⋯ 思考中 · {elapsed:.0f}s"
            self._reasoning_dirty = False

    def _consume_pending_reply_chunks(self) -> None:
        if not self._pending_reply_chunks:
            return
        self._reply_buffer += "".join(self._pending_reply_chunks)
        self._pending_reply_chunks.clear()

    def _consume_pending_reasoning_chunks(self) -> None:
        if not self._pending_reasoning_chunks:
            return
        self._reasoning_buffer += "".join(self._pending_reasoning_chunks)
        self._pending_reasoning_chunks.clear()

    def _finalize_reasoning(self) -> None:
        if self._reasoning is None:
            return
        self._consume_pending_reasoning_chunks()
        elapsed = time.monotonic() - self._reasoning_started
        self._reasoning.title = f"⋯ 已思考 · {elapsed:.1f}s"
        if self._reasoning_body is not None:
            self._reasoning_body.update(self._reasoning_buffer)
        self._reasoning = None
        self._reasoning_body = None
        self._reasoning_buffer = ""
        self._pending_reasoning_chunks.clear()
        self._reasoning_dirty = False

    def _reset_reply_stream(self) -> None:
        """An iteration ended (tool calls follow): freeze this reply widget."""
        self._consume_pending_reply_chunks()
        if self._reply_widget is not None and self._reply_buffer:
            chat = self.query_one("#chat", VerticalScroll)
            if chat.is_vertical_scroll_end:
                chat.anchor()
            self._reply_widget.set_source(self._reply_buffer)
            self._reply_widget.remove_class("reply-streaming")
        self._reply_widget = None
        self._reply_buffer = ""
        self._pending_reply_chunks.clear()
        self._reply_dirty = False

    def _finalize_reply(self, authoritative: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        follow_output = chat.is_vertical_scroll_end
        if self._reply_widget is None:
            self._reply_widget = _StreamingMarkdown(
                classes="reply",
            )
            self._mount(self._reply_widget)
        if follow_output:
            chat.anchor()
        self._reply_widget.remove_class("reply-streaming")
        self._reply_widget.set_source(authoritative)
        self._reply_widget = None
        self._reply_buffer = ""
        self._pending_reply_chunks.clear()
        self._reply_dirty = False

    def _on_tool_started(self, payload: dict[str, Any]) -> None:
        call_id = str(payload.get("tool_call_id") or "")
        label = payload.get("display_name") or payload.get("tool") or "tool"
        args = json.dumps(payload.get("args") or {}, ensure_ascii=False)
        body = Static(f"args: {preview_text(args, 500)}")
        line = Collapsible(
            body,
            title=f"• {label} · running",
            collapsed=True,
            classes="tool-line",
        )
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
        self._mount(Static(escape(text), classes="user-line"))

    def _system_line(self, text: str, error: bool = False) -> None:
        classes = "error-line" if error else "system-line"
        self._mount(Static(text, classes=classes))

    def _mount(self, widget) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        follow_output = chat.is_vertical_scroll_end
        chat.mount(widget)
        if follow_output:
            chat.anchor()

    def _refresh_header(self, state: str) -> None:
        self._state = state
        self.query_one("#header", Static).update(
            f"[b #00d7ff]MiniBot[/] [#505050]·[/] "
            f"[#d4d4d4]{escape(state)}[/]"
        )
        self._refresh_footer()

    def _refresh_footer(self) -> None:
        summary = self.runtime.mcp_host.summary()
        mcp = (
            "mcp off"
            if summary.enabled_servers == 0
            else f"mcp {summary.connected_servers}/{summary.enabled_servers}"
        )
        self.query_one("#footer-bar", Static).update(
            f"{escape(self._workspace_label())} · "
            f"{escape(self.session_id or 'no-session')} · "
            f"{escape(self.runtime.config.model)} · "
            f"{escape(mcp)} · "
            f"approval {escape(self.runtime.approval_policy.mode)} · "
            "Esc cancel · Ctrl+N new"
        )

    def _workspace_label(self) -> str:
        workspace = getattr(self.runtime.context_builder, "workspace", None)
        if workspace is None:
            workspace = getattr(self.runtime.manager, "default_workspace", None)
        if not workspace:
            return "~"

        text = str(workspace)
        home = str(Path.home())
        if text == home:
            return "~"
        if text.startswith(home + "/"):
            return "~" + text[len(home):]
        return text


def _command_completion(command: str) -> str:
    """Turn a help/catalog command spec into the text inserted in the prompt."""
    parts: list[str] = []
    for part in command.split():
        if part.startswith("<") or part.startswith("["):
            break
        parts.append(part)
    completion = " ".join(parts) if parts else command
    return completion + (" " if len(parts) < len(command.split()) else "")


def run_tui(runtime: "MiniBotRuntime") -> None:
    MinibotApp(runtime).run()
