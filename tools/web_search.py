"""Simple web search tool backed by DuckDuckGo HTML results."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
import re

from .base import Tool, ToolExecutionContext
from .result import ToolOutput


@dataclass(frozen=True)
class _SearchResult:
    title: str
    url: str
    snippet: str


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


class WebSearchTool(Tool):
    """Search the public web and return a compact result list."""

    _SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
    _USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    _TIMEOUT = 20
    _DEFAULT_MAX_RESULTS = 5
    _MAX_RESULTS = 8
    _MAX_TITLE_CHARS = 160
    _MAX_SNIPPET_CHARS = 280

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def read_only(self) -> bool:
        return True

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "搜索公开互联网并返回结果标题、摘要和链接。"
            "适合查询最新信息、新闻、公开网页和外部资料入口。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的关键词或问题。",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回多少条结果，默认 5，最大 8。",
                    "minimum": 1,
                    "maximum": self._MAX_RESULTS,
                },
                "allowed_domains": {
                    "type": "array",
                    "description": "可选，只保留这些域名下的结果，例如 ['openai.com', 'reuters.com']。",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(
        self,
        *,
        context: ToolExecutionContext,
        query: str,
        max_results: int = _DEFAULT_MAX_RESULTS,
        allowed_domains: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        del context
        query = _collapse_ws(query)
        if not query:
            return ToolOutput.failure("invalid_args", "搜索失败: query 不能为空。")

        max_results = max(1, min(int(max_results), self._MAX_RESULTS))
        normalized_domains = self._normalize_domains(allowed_domains or [])

        try:
            html = self._fetch_results_html(query)
        except Exception as exc:
            return ToolOutput.failure(
                "error",
                f"搜索失败: {exc}",
                data={"query": query},
            )

        results = self._parse_results(html)
        if normalized_domains:
            results = [
                item for item in results if self._matches_domains(item.url, normalized_domains)
            ]

        if not results:
            return ToolOutput.failure(
                "not_found",
                "未找到匹配结果。",
                data={
                    "query": query,
                    "allowed_domains": sorted(normalized_domains),
                    "results": [],
                },
            )

        selected = results[:max_results]
        return ToolOutput.success(
            f"找到 {len(selected)} 条网页结果。",
            data={
                "query": query,
                "allowed_domains": sorted(normalized_domains),
                "results": [
                    {
                        "title": item.title,
                        "url": item.url,
                        "snippet": item.snippet,
                    }
                    for item in selected
                ],
            },
        )

    def _fetch_results_html(self, query: str) -> str:
        url = self._SEARCH_URL.format(query=quote_plus(query))
        request = Request(url, headers={"User-Agent": self._USER_AGENT})
        with urlopen(request, timeout=self._TIMEOUT) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _parse_results(self, html: str) -> list[_SearchResult]:
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        results: list[_SearchResult] = []
        seen_urls: set[str] = set()

        for match in pattern.finditer(html):
            url = self._extract_result_url(match.group("href"))
            if not url or url in seen_urls:
                continue

            title = self._clean_text(match.group("title"), self._MAX_TITLE_CHARS)
            snippet = self._clean_text(match.group("snippet"), self._MAX_SNIPPET_CHARS)
            if not title:
                continue

            results.append(_SearchResult(title=title, url=url, snippet=snippet))
            seen_urls.add(url)

        return results

    def _extract_result_url(self, raw_href: str) -> str:
        href = unescape(raw_href).strip()
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            href = "https://html.duckduckgo.com" + href

        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc:
            query = parse_qs(parsed.query)
            uddg = query.get("uddg")
            if uddg:
                return self._clean_result_url(unquote(uddg[0]))
            return ""
        return self._clean_result_url(href)

    @staticmethod
    def _clean_text(text: str, max_chars: int) -> str:
        cleaned = _collapse_ws(unescape(_strip_tags(text)))
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _normalize_domains(domains: list[str]) -> set[str]:
        normalized: set[str] = set()
        for raw in domains:
            domain = _collapse_ws(raw).lower()
            if not domain:
                continue
            domain = domain.removeprefix("https://").removeprefix("http://")
            domain = domain.split("/", 1)[0].strip(".")
            if domain:
                normalized.add(domain)
        return normalized

    @staticmethod
    def _matches_domains(url: str, allowed_domains: set[str]) -> bool:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        if host.startswith("www."):
            host = host[4:]
        return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)

    @staticmethod
    def _clean_result_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        blocked_hosts = {"duckduckgo.com", "www.duckduckgo.com"}
        if parsed.netloc.lower() in blocked_hosts:
            return ""
        if parsed.netloc.lower() in {"bing.com", "www.bing.com"} and parsed.path.startswith("/aclick"):
            return ""
        return url
