"""Lightweight interactive CLI for MiniBot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None  # type: ignore[assignment]

import json
import logging
import os
from pathlib import Path
import select
import shutil
import signal
import sys
import threading
import time
import unicodedata
from typing import Any, TextIO

try:
    import readline

    readline.parse_and_bind("set enable-bracketed-paste on")
except Exception:
    readline = None  # type: ignore[assignment]

from .bootstrap import MiniBotRuntime, build_runtime
from .config import Config, load_env
from .interaction.commands import (
    CommandContext,
    CommandNotice,
    _COMMANDS,
    dispatch_command,
)
from .run_log import make_run_id
from .runtime.agent_session import RunCancelled
from .runtime.approval import ApprovalRequest
from .runtime.events import RuntimeEvent


_PASTE_DRAIN_IDLE_SECONDS = 0.05
_PASTE_DRAIN_MAX_SECONDS = 0.35
_PASTE_DRAIN_CHUNK_SIZE = 4096
_PASTE_DRAIN_MAX_BYTES = 64 * 1024
_COMMAND_NAMES = tuple(dict.fromkeys(command.split()[0] for command, _ in _COMMANDS))
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass(frozen=True)
class CliOptions:
    verbose: bool = False
    no_color: bool = False


class CliRenderer:
    """Small ANSI renderer for the REPL and runtime events."""

    _STYLES = {
        "reset": "\x1b[0m",
        "bold": "\x1b[1m",
        "dim": "\x1b[2m",
        "cyan": "\x1b[36m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "magenta": "\x1b[35m",
        "red": "\x1b[31m",
        "gray": "\x1b[90m",
    }

    def __init__(
        self,
        *,
        verbose: bool = False,
        no_color: bool = False,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.verbose = verbose
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.use_color = (
            not no_color
            and os.environ.get("NO_COLOR") is None
            and self.stdout.isatty()
        )
        self._compacted_runs: set[str] = set()
        self._failed_runs: set[str] = set()
        self._status_line = StatusLine(self, enabled=self.stdout.isatty() and not verbose)
        # Typewriter state: True while a streamed line is open on the terminal.
        # While open, the spinner stays off and block output breaks the line first.
        self._stream_open = False
        self._streamed_replies: set[str] = set()
        self._reasoning = ReasoningPreview(
            self,
            enabled=self.stdout.isatty() and not verbose,
        )

    def c(self, text: str, *styles: str) -> str:
        if not self.use_color:
            return text
        codes = "".join(self._STYLES[s] for s in styles if s in self._STYLES)
        return f"{codes}{text}{self._STYLES['reset']}"

    def input_prompt(self) -> str:
        return self.c("\n› ", "bold", "cyan")

    def print_startup(
        self,
        runtime: MiniBotRuntime,
        *,
        session_id: str,
        resumed: bool,
        startup_logs: Sequence[str] = (),
    ) -> None:
        session_state = "resumed" if resumed else "new"
        summary = runtime.mcp_host.summary()
        mcp = (
            "mcp off"
            if summary.enabled_servers == 0
            else f"mcp {summary.connected_servers}/{summary.enabled_servers}"
        )
        print(
            (
                f"{self.c('MiniBot', 'bold', 'magenta')} "
                f"{self.c(runtime.config.model, 'gray')} "
                f"{self.c(session_id, 'cyan')} "
                f"{self.c(session_state, 'gray')} "
                f"{self.c(mcp, 'gray')} "
                f"{self.c('approval ' + runtime.approval_policy.mode, 'gray')}"
            ),
            file=self.stdout,
        )
        print(
            self.c("Tab 补全 slash 命令 · /help 查看命令 · Ctrl+D 退出", "gray"),
            file=self.stdout,
        )
        if self.verbose:
            for line in startup_logs:
                self.info(line)

    def info(self, message: str) -> None:
        self._flush_live_regions()
        print(self.c(f"  {message}", "gray"), file=self.stdout)

    def success(self, message: str) -> None:
        self._flush_live_regions()
        print(self.c(f"  {message}", "green"), file=self.stdout)

    def warn(self, message: str) -> None:
        self._flush_live_regions()
        print(self.c(f"  {message}", "yellow"), file=self.stdout)

    def error(self, message: str) -> None:
        self._flush_live_regions()
        print(self.c(f"  {message}", "red"), file=self.stdout)

    def _flush_live_regions(self) -> None:
        """Collapse the reasoning preview and close the typewriter line.

        Called before any block output so live regions never interleave
        with printed lines.
        """
        self._reasoning.collapse()
        self._end_stream_line()
        self.clear_status()

    def render_notice(self, notice: CommandNotice) -> None:
        printer = {
            "success": self.success,
            "warning": self.warn,
            "error": self.error,
            "info": self.info,
        }.get(notice.kind, self.info)
        if notice.body:
            printer(notice.title)
            print(_indent(notice.body), file=self.stdout)
            return
        printer(notice.title)

    def render_notices(self, notices: Sequence[CommandNotice]) -> None:
        for notice in notices:
            self.render_notice(notice)

    def render_event(self, event: RuntimeEvent) -> None:
        if event.type == "message.delta":
            self._render_stream_delta(event)
            return
        if event.type == "message.completed" and self._stream_open:
            # The reply just finished streaming; close the line and remember
            # the run so print_reply does not repeat the text.
            self._flush_live_regions()
            self._streamed_replies.add(event.run_id)
        message = self.format_event(event)
        if message:
            self._flush_live_regions()
            print(
                f"  {self.c('›', 'dim')} {self.c(message, 'dim')}",
                file=self.stdout,
            )
        self.update_status_from_event(event)

    def _render_stream_delta(self, event: RuntimeEvent) -> None:
        payload = event.payload
        text = str(payload.get("text") or "")
        if not text:
            return
        channel = payload.get("channel")
        if channel == "reasoning":
            # The preview needs whole lines to itself; close any open
            # typewriter line from a previous iteration first.
            self._end_stream_line()
            self.clear_status()
            self._reasoning.feed(text)
            return
        if channel != "text":
            return
        self._reasoning.collapse()
        self.clear_status()
        if not self._stream_open:
            self._stream_open = True
            self.stdout.write(f"\n{self.c('MiniBot ›', 'bold', 'magenta')} ")
        self.stdout.write(text)
        self.stdout.flush()

    def _end_stream_line(self) -> None:
        if not self._stream_open:
            return
        self._stream_open = False
        self.stdout.write("\n")
        self.stdout.flush()

    def format_event(self, event: RuntimeEvent) -> str | None:
        payload = event.payload
        if event.type == "context.compacted":
            self._compacted_runs.add(event.run_id)
            return f"已自动压缩: {payload.get('message') or ''}".rstrip()
        if event.type == "tool_call.started":
            label = payload.get("display_name") or payload.get("tool")
            if self.verbose:
                args = _json_preview(payload.get("args") or {}, limit=1200)
                return f"工具: {label}({args})"
            return f"工具: {label}"
        if event.type in {"tool_call.completed", "tool_call.failed"}:
            label = payload.get("display_name") or payload.get("tool")
            status = "失败" if event.type == "tool_call.failed" else "完成"
            summary = str(payload.get("summary") or "")
            if not self.verbose:
                summary = _truncate(summary, 180)
            bits = [f"工具{status}: {label}"]
            if summary:
                bits.append(summary)
            if self.verbose and payload.get("artifact"):
                bits.append(f"artifact={payload.get('artifact')}")
            if self.verbose and payload.get("truncated"):
                bits.append("truncated")
            return " · ".join(bits)
        if event.type == "model.request.retrying":
            return (
                f"模型请求重试 {payload.get('attempt')}/{payload.get('max_retries')}: "
                f"{payload.get('error_type')} · 等待 {payload.get('delay_seconds')}s"
            )
        if event.type == "approval.required":
            return f"等待批准: {payload.get('tool')}"
        if event.type == "approval.resolved":
            state = "已批准" if payload.get("approved") else "已拒绝"
            auto = " · auto" if payload.get("auto") else ""
            return f"{state}: {payload.get('tool')}{auto}"
        if event.type == "run.cancelled":
            return "运行已取消"
        if event.type == "run.failed":
            self._failed_runs.add(event.run_id)
            return f"运行失败: {payload.get('error_type')}: {payload.get('message')}"

        if not self.verbose:
            return None

        if event.type == "context.usage":
            return f"context: {payload.get('current_tokens')}/{payload.get('budget')} tokens"
        if event.type == "model.request.started":
            return f"第 {payload.get('iteration')} 轮请求模型: {payload.get('model')}"
        if event.type == "model.request.completed":
            if payload.get("empty_reply"):
                debug = _json_preview(payload.get("response_debug") or {}, limit=1200)
                return f"第 {payload.get('iteration')} 轮模型返回空回答: {debug}"
            usage = payload.get("usage") or {}
            return (
                f"第 {payload.get('iteration')} 轮模型返回 "
                f"{payload.get('tool_call_count', 0)} 个工具调用 · "
                f"{payload.get('elapsed_ms')}ms · usage={usage}"
            )
        return None

    def print_compaction_if_needed(
        self,
        *,
        run_id: str,
        did_compact: bool,
        compact_message: str | None,
    ) -> None:
        if not did_compact or not compact_message or run_id in self._compacted_runs:
            return
        self._compacted_runs.add(run_id)
        self.info(f"已自动压缩: {compact_message}")

    def failed_run_seen(self, run_id: str) -> bool:
        return run_id in self._failed_runs

    def print_reply(self, reply: str, *, run_id: str | None = None) -> None:
        self._flush_live_regions()
        if run_id is not None and run_id in self._streamed_replies:
            # Already on screen via the typewriter; printing again would dupe.
            self._streamed_replies.discard(run_id)
            return
        visible = reply if reply.strip() else "（模型返回空回复）"
        print(
            f"\n{self.c('MiniBot ›', 'bold', 'magenta')} {visible}",
            file=self.stdout,
        )

    def start_status(self, message: str = "thinking") -> None:
        self._status_line.start(message)

    def update_status(self, message: str) -> None:
        self._status_line.update(message)

    def clear_status(self) -> None:
        self._status_line.clear()

    def stop_status(self) -> None:
        self._status_line.stop()

    def update_status_from_event(self, event: RuntimeEvent) -> None:
        if self._stream_open or self._reasoning.active:
            # The typewriter or the reasoning preview owns the terminal;
            # a spinner redraw would corrupt it. Status resumes after close.
            return
        payload = event.payload
        if event.type == "run.started":
            self.update_status("thinking")
        elif event.type == "model.request.started":
            iteration = payload.get("iteration")
            self.update_status(f"thinking · round {iteration}")
        elif event.type == "model.request.completed":
            tool_count = payload.get("tool_call_count", 0)
            self.update_status("drafting" if tool_count == 0 else f"planning {tool_count} tool(s)")
        elif event.type == "model.request.retrying":
            self.update_status(
                f"retrying {payload.get('attempt')}/{payload.get('max_retries')}"
            )
        elif event.type == "tool_call.started":
            label = payload.get("display_name") or payload.get("tool") or "tool"
            self.update_status(f"running {label}")
        elif event.type in {"tool_call.completed", "tool_call.failed"}:
            self.update_status("thinking")
        elif event.type == "approval.required":
            self.update_status(f"waiting approval · {payload.get('tool')}")
        elif event.type == "approval.resolved":
            self.update_status("thinking")
        elif event.type == "context.compacted":
            self.update_status("compacting context")
        elif event.type == "message.completed":
            self.update_status("ready")
        elif event.type in {"run.completed", "run.failed", "run.cancelled"}:
            self.clear_status()

    def prompt_approval(
        self,
        request: ApprovalRequest,
        cancel_event: threading.Event | None,
    ) -> bool:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("run cancelled while waiting for approval")

        self._flush_live_regions()
        preview = ", ".join(f"{key}={value!r}" for key, value in request.args.items())
        prompt = (
            f"  {self.c('批准执行', 'yellow', 'bold')} "
            f"{self.c(request.tool_name, 'cyan')}({preview}) "
            f"{self.c('[y/N]', 'gray')} "
        )
        while True:
            answer = input(prompt).strip().lower()
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled("run cancelled while waiting for approval")
            if answer in {"y", "yes"}:
                return True
            if answer in {"", "n", "no"}:
                return False
            self.warn("请输入 y 或 n。")
            prompt = f"  {self.c('[y/N]', 'gray')} "


class StatusLine:
    """Single-line spinner that never persists in transcript output."""

    def __init__(self, renderer: CliRenderer, *, enabled: bool) -> None:
        self.renderer = renderer
        self.enabled = enabled
        self._message = ""
        self._active = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._message = message
            self._active = True
            self._stop_event.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop,
                    name="minibot-cli-status",
                    daemon=True,
                )
                self._thread.start()

    def update(self, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._message = message
            should_start = not self._active
        if should_start:
            self.start(message)

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            active = self._active
            self._active = False
            if active:
                self.renderer.stdout.write("\r\x1b[2K")
                self.renderer.stdout.flush()

    def stop(self) -> None:
        if not self.enabled:
            return
        self.clear()
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    def _loop(self) -> None:
        frame_index = 0
        while not self._stop_event.wait(0.12):
            with self._lock:
                active = self._active
                message = self._message
                if not active:
                    continue
                frame = _SPINNER_FRAMES[frame_index % len(_SPINNER_FRAMES)]
                frame_index += 1
                text = self.renderer.c(f"{frame} {message}", "gray")
                self.renderer.stdout.write(f"\r\x1b[2K{text}")
                self.renderer.stdout.flush()


class ReasoningPreview:
    """Live tail view of streamed reasoning that collapses to one dim line.

    The preview owns a small self-drawn region: a header plus the last few
    reasoning lines, each truncated to the terminal width. Because it always
    knows exactly how many physical lines it drew, erasing the region with
    cursor-up + clear-to-end is reliable — that is what makes the "collapse"
    possible on a plain terminal.
    """

    _MAX_TAIL_LINES = 3
    _REDRAW_INTERVAL = 0.08

    def __init__(self, renderer: CliRenderer, *, enabled: bool) -> None:
        self.renderer = renderer
        self.enabled = enabled
        self._buffer: list[str] = []
        self._drawn_lines = 0
        self._started_at: float | None = None
        self._last_draw = 0.0

    @property
    def active(self) -> bool:
        return self._started_at is not None

    def feed(self, text: str) -> None:
        """Accept one reasoning delta; redraw the tail view (throttled)."""
        if self._started_at is None:
            self._started_at = time.monotonic()
        self._buffer.append(text)
        if not self.enabled:
            return
        now = time.monotonic()
        if self._drawn_lines and now - self._last_draw < self._REDRAW_INTERVAL:
            return
        self._draw(now)

    def collapse(self) -> None:
        """Erase the live region, leave a single summary line, reset."""
        if self._started_at is None:
            return
        elapsed = time.monotonic() - self._started_at
        self._erase()
        self.renderer.stdout.write(
            self.renderer.c(f"  ⋯ 已思考 · {elapsed:.1f}s", "dim") + "\n"
        )
        self.renderer.stdout.flush()
        self._buffer.clear()
        self._drawn_lines = 0
        self._started_at = None
        self._last_draw = 0.0

    def _erase(self) -> None:
        if not self._drawn_lines:
            return
        self.renderer.stdout.write(f"\x1b[{self._drawn_lines}A\r\x1b[J")
        self._drawn_lines = 0

    def _draw(self, now: float) -> None:
        width = max(20, shutil.get_terminal_size().columns - 4)
        lines = ["⋯ 思考中"]
        lines.extend(self._tail_lines(width))
        self._erase()
        for line in lines:
            self.renderer.stdout.write(
                self.renderer.c(f"  {line}", "dim", "gray") + "\n"
            )
        self.renderer.stdout.flush()
        self._drawn_lines = len(lines)
        self._last_draw = now

    def _tail_lines(self, width: int) -> list[str]:
        text = "".join(self._buffer).replace("\t", "  ")
        logical = [line.strip() for line in text.splitlines() if line.strip()]
        return [
            _truncate_display(line, width)
            for line in logical[-self._MAX_TAIL_LINES :]
        ]


def _truncate_display(text: str, max_cols: int) -> str:
    """Truncate to a display width, counting East Asian wide chars as two."""
    used = 0
    kept: list[str] = []
    for ch in text:
        char_width = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + char_width > max_cols - 1:
            kept.append("…")
            break
        kept.append(ch)
        used += char_width
    return "".join(kept)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minibot",
        description="MiniBot interactive local agent.",
        epilog=_format_repl_commands(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show model rounds, context usage, and full tool previews",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable ANSI colors; NO_COLOR is also respected",
    )
    parser.add_argument(
        "--migrate",
        nargs="+",
        metavar="DIR",
        help="move per-workspace .minibot state from DIR(s) into the global home, then exit",
    )
    parser.add_argument("prompt", nargs="*", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.migrate:
        return _run_migration(args.migrate)
    if args.prompt:
        print(
            "MiniBot 只支持交互模式。请直接运行 `minibot` 进入 REPL。",
            file=sys.stderr,
        )
        return 2

    options = CliOptions(verbose=args.verbose, no_color=args.no_color)
    renderer = CliRenderer(verbose=options.verbose, no_color=options.no_color)

    load_env()
    try:
        config = Config.from_env()
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    startup_logs: list[str] = []
    try:
        with _quiet_startup(enabled=not options.verbose):
            runtime = build_runtime(
                config=config,
                log_handler=startup_logs.append if options.verbose else None,
            )
    except RuntimeError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    runtime.approval_policy.handler = renderer.prompt_approval
    try:
        run_repl(runtime, renderer, startup_logs=startup_logs)
    finally:
        runtime.close()
    return 0


def _run_migration(directories: Sequence[str]) -> int:
    from .config import resolve_state_home
    from .migrate import migrate_workspace_state

    state_home = resolve_state_home()
    print(f"全局状态目录: {state_home}")
    for raw in directories:
        report = migrate_workspace_state(Path(raw).expanduser(), state_home)
        for line in report.lines():
            print(line)
    return 0


def run_repl(
    runtime: MiniBotRuntime,
    renderer: CliRenderer,
    *,
    startup_logs: Sequence[str] = (),
) -> None:
    session, resumed = runtime.manager.startup_session()
    current_session_id = session.session_id
    completion_state = install_slash_completion()
    try:
        renderer.print_startup(
            runtime,
            session_id=current_session_id,
            resumed=resumed,
            startup_logs=startup_logs,
        )

        while True:
            try:
                raw = read_user_input(renderer)
            except EOFError:
                renderer.info("已退出。")
                return
            except KeyboardInterrupt:
                renderer.info("按 Ctrl+D 或输入 exit 退出。")
                continue

            text = raw.strip()
            if not text or _is_terminal_escape_sequence(text):
                continue

            result = dispatch_command(
                text,
                current_session_id,
                _command_context(runtime),
            )
            if result.handled:
                if result.current_session_id:
                    current_session_id = result.current_session_id
                renderer.render_notices(result.notices)
                if result.should_exit:
                    return
                continue

            current_session_id = _run_prompt(
                runtime,
                renderer,
                current_session_id=current_session_id,
                user_input=text,
            )
    finally:
        renderer.stop_status()
        completion_state.restore()


def read_user_input(renderer: CliRenderer) -> str:
    first_line = input(renderer.input_prompt())
    pending = _drain_pending_stdin()
    if pending:
        return _merge_pasted_input(renderer, f"{first_line}\n{pending}")
    return first_line.strip()


@dataclass
class CompletionState:
    old_completer: object | None = None
    old_delims: str | None = None

    def restore(self) -> None:
        if readline is None:
            return
        try:
            readline.set_completer(self.old_completer)
            if self.old_delims is not None:
                readline.set_completer_delims(self.old_delims)
        except Exception:
            pass


def install_slash_completion() -> CompletionState:
    if readline is None:
        return CompletionState()

    try:
        old_completer = readline.get_completer()
        old_delims = readline.get_completer_delims()
        readline.set_completer_delims(" \t\n")
        readline.set_completer(_slash_completer)
        readline_doc = (readline.__doc__ or "").lower()
        if "libedit" in readline_doc or "editline" in readline_doc:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        return CompletionState(old_completer=old_completer, old_delims=old_delims)
    except Exception:
        return CompletionState()


def _slash_completer(text: str, state: int) -> str | None:
    if readline is None:
        return None
    try:
        buffer = readline.get_line_buffer()
        begin = readline.get_begidx()
    except Exception:
        return None

    if begin != 0 or not buffer.startswith("/"):
        return None
    matches = [command + " " for command in _COMMAND_NAMES if command.startswith(text)]
    try:
        return matches[state]
    except IndexError:
        return None


def _run_prompt(
    runtime: MiniBotRuntime,
    renderer: CliRenderer,
    *,
    current_session_id: str,
    user_input: str,
) -> str:
    run_id = make_run_id()
    restore_sigint = _install_run_sigint_handler(runtime, renderer, run_id)
    renderer.start_status("thinking")
    try:
        result = runtime.agent_session.prompt(
            current_session_id,
            user_input,
            run_id=run_id,
            event_handler=renderer.render_event,
        )
    except RunCancelled:
        renderer.clear_status()
        return current_session_id
    except KeyboardInterrupt:
        runtime.agent_session.abort(run_id)
        renderer.clear_status()
        renderer.warn("已请求取消当前运行。")
        return current_session_id
    except Exception as exc:
        renderer.clear_status()
        if not renderer.failed_run_seen(run_id):
            renderer.error(f"运行失败: {exc}")
        return current_session_id
    finally:
        restore_sigint()

    renderer.clear_status()
    renderer.print_compaction_if_needed(
        run_id=run_id,
        did_compact=result.did_compact,
        compact_message=result.compact_message,
    )
    renderer.print_reply(result.reply, run_id=run_id)
    reloaded = runtime.manager.load(current_session_id)
    return current_session_id if reloaded is None else reloaded.session_id


def _install_run_sigint_handler(
    runtime: MiniBotRuntime,
    renderer: CliRenderer,
    run_id: str,
):
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    previous = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):  # type: ignore[no-untyped-def]
        del signum, frame
        if runtime.agent_session.abort(run_id):
            renderer.warn("已请求取消当前运行。")
        else:
            renderer.warn("当前没有可取消的运行。")

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except (ValueError, OSError):
        return lambda: None

    def _restore() -> None:
        try:
            signal.signal(signal.SIGINT, previous)
        except (ValueError, OSError):
            pass

    return _restore


@contextmanager
def _quiet_startup(*, enabled: bool):
    if not enabled:
        yield
        return

    previous_disabled = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
        try:
            yield
        finally:
            logging.disable(previous_disabled)


def _command_context(runtime: MiniBotRuntime) -> CommandContext:
    return CommandContext(
        sessions=runtime.manager,
        compactor=runtime.compactor,
        context_builder=runtime.context_builder,
        memory_store=runtime.memory_store,
        approval_policy=runtime.approval_policy,
        mcp_host=runtime.mcp_host,
        config=runtime.config,
    )


def _is_terminal_escape_sequence(text: str) -> bool:
    return text.startswith("\x1b") or text.startswith("^[")


def _drain_pending_stdin() -> str:
    if fcntl is None:
        return ""
    if not sys.stdin.isatty():
        return ""
    try:
        fd = sys.stdin.fileno()
        original_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    except (AttributeError, OSError, ValueError):
        return ""

    chunks: list[bytes] = []
    deadline = time.monotonic() + _PASTE_DRAIN_MAX_SECONDS
    total = 0

    try:
        fcntl.fcntl(fd, fcntl.F_SETFL, original_flags | os.O_NONBLOCK)
        while time.monotonic() < deadline and total < _PASTE_DRAIN_MAX_BYTES:
            timeout = min(
                _PASTE_DRAIN_IDLE_SECONDS,
                max(0.0, deadline - time.monotonic()),
            )
            ready, _, _ = select.select([fd], [], [], timeout)
            if not ready:
                break
            try:
                data = os.read(fd, _PASTE_DRAIN_CHUNK_SIZE)
            except BlockingIOError:
                continue
            if not data:
                break
            chunks.append(data)
            total += len(data)
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, original_flags)

    if not chunks:
        return ""
    encoding = sys.stdin.encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


def _merge_pasted_input(renderer: CliRenderer, text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    line_count = max(1, len(cleaned.splitlines()))
    renderer.info(f"检测到 {line_count} 行粘贴内容，已合并为一条请求。")
    return cleaned


def _format_repl_commands() -> str:
    width = max(len(command) for command, _ in _COMMANDS) + 2
    lines = ["REPL commands:"]
    lines.extend(f"  {command.ljust(width)}{description}" for command, description in _COMMANDS)
    return "\n".join(lines)


def _json_preview(value: Any, *, limit: int) -> str:
    return _truncate(json.dumps(value, ensure_ascii=False, default=str), limit)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" if line else "" for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
