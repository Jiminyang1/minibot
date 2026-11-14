"""Tool abstraction for MiniBot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Literal, TypeAlias
from urllib.parse import unquote_plus

from .definitions import ModelToolDefinition
from .result import ToolOutput


ToolLayer: TypeAlias = Literal["kernel", "extension"]
ToolSource: TypeAlias = Literal["local", "mcp"]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime-only execution context for one tool call."""

    session_id: str
    run_id: str | None = None
    cancel_event: threading.Event | None = None


def written_forms(text: str) -> tuple[str, ...]:
    """The forms in which model-written text is compared with a later echo.

    A tool rarely echoes its input byte for byte: ``web_search`` collapses
    whitespace, others trim, change case or percent-decode. Whitespace and
    case are dropped and the percent-decoded form is added, so those echoes
    still match what the model wrote.
    """
    squashed = "".join(text.split()).casefold()
    decoded = "".join(unquote_plus(text).split()).casefold()
    return (squashed,) if decoded == squashed else (squashed, decoded)


@dataclass(frozen=True)
class ShownText:
    """One successful result of an untrusted tool, in session order."""

    text: str
    # How much of ``UntrustedExposure.written`` the model had produced when
    # this result arrived.
    written_before: int


@dataclass(frozen=True)
class UntrustedExposure:
    """What the session has already shown the model from untrusted sources.

    ``tainted``: a tool whose output may carry third-party text (web pages,
    search results, MCP servers) has returned into this session, so the
    model may be following injected instructions.

    ``mentions`` answers "did someone other than the model put this text in
    front of the model?". ``seen_text`` is what the user wrote and always
    counts. ``shown`` holds untrusted tool results; one counts only if the
    model had not written the text before that result arrived (``written``:
    replies, tool-call arguments and summaries, in ``written_forms``).
    Otherwise the result is an echo of the model's own words, and a URL in
    it was built by the model, not found by it.
    """

    tainted: bool = False
    seen_text: str = ""
    shown: tuple[ShownText, ...] = ()
    written: str = ""

    def mentions(self, text: str) -> bool:
        if not text:
            return False
        if text in self.seen_text:
            return True
        forms = written_forms(text)
        return any(
            text in item.text
            and all(
                self.written.find(form, 0, item.written_before) == -1
                for form in forms
            )
            for item in self.shown
        )


class Tool(ABC):
    """A single agent capability exposed through function calling."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
    ) -> None:
        self._workspace = workspace.resolve() if workspace else None

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name used by the model."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description for the model."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool's parameters."""

    @property
    def layer(self) -> ToolLayer:
        """Architectural layer for the tool."""
        return "extension"

    @property
    def source(self) -> ToolSource:
        """High-level source category for UI/logging."""
        return "local"

    @property
    def display_name(self) -> str:
        """Human-friendly tool label for logs."""
        return self.name

    @property
    def requires_approval(self) -> bool:
        """Whether this tool needs user confirmation before execution."""
        return False

    @property
    def untrusted_output(self) -> bool:
        """Whether results may carry third-party text the model could obey."""
        return self.source == "mcp"

    def approval_reason(
        self,
        args: dict[str, Any],
        exposure: UntrustedExposure,
    ) -> str | None:
        """Why this call needs approval, or None to run it unasked.

        The default follows the static ``requires_approval`` flag. Tools
        override it to ask only once untrusted content is in the session.
        """
        del args, exposure
        return "该工具需要审批。" if self.requires_approval else None

    @property
    def read_only(self) -> bool:
        """Whether this tool is side-effect free and safe to batch."""
        return False

    @property
    def exclusive(self) -> bool:
        """Whether this tool should always run alone."""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """Whether this tool may run with other concurrency-safe tools."""
        return self.read_only and not self.exclusive

    @abstractmethod
    def execute(self, *, context: ToolExecutionContext, **kwargs: Any) -> ToolOutput:
        """Run the tool with named parameters."""

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace and verify it stays inside."""
        if self._workspace is None:
            return Path(path).resolve()
        resolved = (self._workspace / path).resolve()
        if not resolved.is_relative_to(self._workspace):
            raise PermissionError(
                f"路径 {path} 超出工作目录 {self._workspace}"
            )
        return resolved

    def to_model_definition(self) -> ModelToolDefinition:
        """Return a provider-agnostic runtime tool definition."""
        return ModelToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )
