"""Public webpage fetch tool."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
from typing import Any
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json
import re
import xml.etree.ElementTree as ET

from .base import Tool, ToolExecutionContext
from .result import ToolResult

if TYPE_CHECKING:
    from ..session import SessionManager


@dataclass(frozen=True)
class _FetchedPage:
    final_url: str
    status_code: int
    content_type: str
    body_text: str
    title: str = ""
    published_at: str | None = None
    extractor: str = "direct"
    byte_truncated: bool = False


class _ReadableHtmlParser(HTMLParser):
    """Extract readable text and a few common metadata fields from HTML."""

    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "section",
        "tr",
        "ul",
        "ol",
    }
    _SKIP_TAGS = {
        "aside",
        "footer",
        "head",
        "iframe",
        "menu",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
        "template",
    }
    _TITLE_META_NAMES = {
        "og:title",
        "twitter:title",
        "title",
    }
    _PUBLISHED_META_NAMES = {
        "article:published_time",
        "article:modified_time",
        "date",
        "dc.date",
        "og:pubdate",
        "parsely-pub-date",
        "pubdate",
        "publishdate",
        "published_time",
        "timestamp",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_chunks: list[str] = []
        self._parts: list[str] = []
        self.meta_title: str | None = None
        self.published_at: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {
            key.lower(): (value or "").strip()
            for key, value in attrs
            if key
        }

        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            self._handle_meta(attr_map)
            return

        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if tag in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self._title_chunks.append(text)
            return
        if self._skip_depth > 0:
            return

        if not self._parts:
            self._parts.append(text)
            return
        if self._parts[-1] == "\n":
            self._parts.append(text)
            return
        self._parts.append(" ")
        self._parts.append(text)

    @property
    def title(self) -> str:
        joined = " ".join(self._title_chunks).strip()
        return joined or (self.meta_title or "")

    @property
    def text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _append_break(self) -> None:
        if not self._parts:
            return
        if self._parts[-1] != "\n":
            self._parts.append("\n")

    def _handle_meta(self, attrs: dict[str, str]) -> None:
        key = (attrs.get("property") or attrs.get("name") or "").strip().lower()
        content = attrs.get("content", "").strip()
        if not key or not content:
            return
        if key in self._TITLE_META_NAMES and not self.meta_title:
            self.meta_title = content
        if key in self._PUBLISHED_META_NAMES and not self.published_at:
            self.published_at = content


class FetchUrlTool(Tool):
    """Fetch a public webpage and extract readable text."""

    _USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    _TIMEOUT = 20
    _MAX_BYTES = 1024 * 1024
    _INLINE_CHARS = 3000
    _JINA_URL = "https://r.jina.ai/{url}"
    _ALLOWED_CONTENT_TYPES = {
        "application/json",
        "application/xhtml+xml",
        "text/html",
        "text/plain",
    }

    def __init__(self, session_manager: SessionManager) -> None:
        super().__init__(workspace=None, session_manager=session_manager)

    @property
    def name(self) -> str:
        return "fetch_url"

    @property
    def description(self) -> str:
        return (
            "抓取公开网页并提取可读正文、标题和链接。"
            "适合在 `web_search` 找到候选链接后继续深入阅读页面内容。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取的公开网页 URL。",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        url: str,
        **kwargs: Any,
    ) -> ToolResult:
        normalized = url.strip()
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ToolResult.failure(
                "invalid_args",
                "抓取失败: 请输入有效的 http/https URL。",
                data={"url": normalized},
            )

        page = self._fetch_google_news_rss(normalized)
        if page is None:
            page = self._fetch_via_jina(normalized)
        if page is None:
            try:
                page = self._fetch_direct(normalized)
            except Exception as exc:
                return ToolResult.failure(
                    "error",
                    f"抓取失败: {exc}",
                    data={"url": normalized},
                )

        total_chars = len(page.body_text)
        data: dict[str, Any] = {
            "url": normalized,
            "final_url": page.final_url,
            "content_type": page.content_type,
            "status_code": page.status_code,
            "title": page.title,
            "extractor": page.extractor,
            "total_chars": total_chars,
            "byte_truncated": page.byte_truncated,
        }
        if page.published_at:
            data["published_at"] = page.published_at

        if total_chars <= self._INLINE_CHARS and not page.byte_truncated:
            data["content"] = page.body_text
            return ToolResult.success(
                f"已抓取网页 {page.final_url}（{total_chars} 字符）。",
                data=data,
            )

        artifact = self._require_session_manager().put_artifact_text(
            context.session_id,
            page.body_text,
            kind="text",
            name=page.title or page.final_url,
        )
        data["preview"] = page.body_text[: self._INLINE_CHARS]
        details: list[str] = []
        if page.byte_truncated:
            details.append("页面较大，仅提取前段内容")
        if total_chars > self._INLINE_CHARS:
            details.append("正文预览已截断")
        detail_text = "，".join(details) if details else "已截断预览"
        return ToolResult.success(
            f"已抓取网页 {page.final_url}（{total_chars} 字符，{detail_text}）。",
            data=data,
            artifact=artifact,
            truncated=True,
        )

    def _fetch_google_news_rss(self, url: str) -> _FetchedPage | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host != "news.google.com":
            return None
        if not (
            parsed.path.startswith("/topics/")
            or parsed.path.startswith("/rss/topics/")
        ):
            return None

        rss_path = parsed.path if parsed.path.startswith("/rss/") else f"/rss{parsed.path}"
        rss_url = parsed._replace(path=rss_path).geturl()
        request = Request(
            rss_url,
            headers={
                "User-Agent": self._USER_AGENT,
                "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._TIMEOUT) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                raw = response.read(self._MAX_BYTES + 1)
                byte_truncated = len(raw) > self._MAX_BYTES
                if byte_truncated:
                    raw = raw[: self._MAX_BYTES]
                charset = response.headers.get_content_charset() or "utf-8"
        except Exception:
            return None

        xml_text = raw.decode(charset, errors="ignore").strip()
        if not xml_text:
            return None
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None

        channel = root.find("./channel")
        if channel is None:
            return None

        channel_title = self._xml_text(channel.find("title")) or "Google News RSS"
        lines: list[str] = []
        for index, item in enumerate(channel.findall("item")[:20], start=1):
            title = self._xml_text(item.find("title"))
            link = self._xml_text(item.find("link"))
            source = self._xml_text(item.find("source"))
            pub_date = self._xml_text(item.find("pubDate"))
            description = self._xml_text(item.find("description"))
            description = self._strip_html_fallback(description).strip()

            if title or link:
                lines.append(f"{index}. {title or link}")
            if source:
                lines.append(f"来源: {source}")
            if pub_date:
                lines.append(f"时间: {pub_date}")
            if description:
                lines.append(f"摘要: {description[:280]}")
            if link:
                lines.append(f"链接: {link}")
            if title or link:
                lines.append("")

        body_text = "\n".join(lines).strip()
        if not body_text:
            return None

        return _FetchedPage(
            final_url=rss_url,
            status_code=status_code,
            content_type="application/rss+xml",
            body_text=body_text,
            title=channel_title,
            extractor="google_news_rss",
            byte_truncated=byte_truncated,
        )

    def _fetch_via_jina(self, url: str) -> _FetchedPage | None:
        if not self._can_use_jina(url):
            return None

        request = Request(
            self._JINA_URL.format(url=url),
            headers={
                "User-Agent": self._USER_AGENT,
                "Accept": "text/plain,text/markdown;q=0.9,*/*;q=0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._TIMEOUT) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                raw = response.read(self._MAX_BYTES + 1)
                byte_truncated = len(raw) > self._MAX_BYTES
                if byte_truncated:
                    raw = raw[: self._MAX_BYTES]
                charset = response.headers.get_content_charset() or "utf-8"
        except Exception:
            return None

        text = raw.decode(charset, errors="ignore").strip()
        if not text:
            return None

        title = ""
        final_url = url
        body_text = text
        published_at: str | None = None

        if "Markdown Content:" in text:
            header, _, content = text.partition("Markdown Content:")
            body_text = content.strip()
            title = self._match_prefixed_value(header, "Title:")
            final_url = self._match_prefixed_value(header, "URL Source:") or url
            published_at = self._match_prefixed_value(header, "Published Time:")

        body_text = body_text.strip()
        if not body_text:
            return None

        return _FetchedPage(
            final_url=final_url,
            status_code=status_code,
            content_type="text/markdown",
            body_text=body_text,
            title=title,
            published_at=published_at,
            extractor="jina",
            byte_truncated=byte_truncated,
        )

    def _fetch_direct(self, url: str) -> _FetchedPage:
        request = Request(
            url,
            headers={
                "User-Agent": self._USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1",
            },
        )
        with urlopen(request, timeout=self._TIMEOUT) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            if content_type not in self._ALLOWED_CONTENT_TYPES:
                raise ValueError(f"不支持的内容类型 {content_type}")

            raw = response.read(self._MAX_BYTES + 1)
            byte_truncated = len(raw) > self._MAX_BYTES
            if byte_truncated:
                raw = raw[: self._MAX_BYTES]
            charset = response.headers.get_content_charset() or "utf-8"

        text = raw.decode(charset, errors="ignore")
        title = ""
        published_at: str | None = None
        body_text = text
        extractor = "direct"

        if content_type in {"text/html", "application/xhtml+xml"}:
            parser = _ReadableHtmlParser()
            parser.feed(text)
            parser.close()
            title = parser.title
            published_at = parser.published_at
            body_text = self._clean_html_text(
                parser.text or self._strip_html_fallback(text)
            )
            extractor = "html_parser"
        elif content_type == "application/json":
            try:
                body_text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
                extractor = "json"
            except json.JSONDecodeError:
                body_text = text
                extractor = "text"

        body_text = body_text.strip()
        if not body_text:
            raise ValueError("抓取成功，但没有提取到可读正文")

        return _FetchedPage(
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            body_text=body_text,
            title=title,
            published_at=published_at,
            extractor=extractor,
            byte_truncated=byte_truncated,
        )

    @staticmethod
    def _strip_html_fallback(html: str) -> str:
        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _clean_html_text(text: str) -> str:
        cleaned = text
        cleaned = re.sub(
            r"@keyframes\s+[^{]+\{(?:[^{}]|\{[^{}]*\})*\}",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        for _ in range(4):
            updated = re.sub(
                r"(?<!\w)[#.@]?[A-Za-z0-9_,:%\-\s>*+\[\]\(\)=/]{1,120}\{[^{}]{1,500}\}",
                " ",
                cleaned,
            )
            if updated == cleaned:
                break
            cleaned = updated
        cleaned = re.sub(r"\b(?:var|let|const)\s+[A-Za-z_$][\w$]*\s*=\s*[^;]{1,200};", " ", cleaned)
        cleaned = re.sub(r"\bfunction\s+[A-Za-z_$][\w$]*\([^)]*\)\s*\{[^{}]{1,500}\}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _match_prefixed_value(text: str, prefix: str) -> str | None:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(prefix):
                value = line.removeprefix(prefix).strip()
                return value or None
        return None

    @staticmethod
    def _xml_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    @staticmethod
    def _can_use_jina(url: str) -> bool:
        host = (urlparse(url).hostname or "").strip().lower()
        if not host or host in {"localhost"} or host.endswith(".local"):
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True
        return not (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
        )
