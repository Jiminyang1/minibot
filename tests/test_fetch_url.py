from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import HTTPHandler, HTTPSHandler, Request
from urllib.response import addinfourl
import http.client
import io
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.tools.base import ToolExecutionContext
from minibot.tools.fetch_url import FetchUrlTool


def _response(url: str, code: int, headers: dict[str, str], body: bytes = b"") -> addinfourl:
    raw = "".join(f"{key}: {value}\r\n" for key, value in headers.items()) + "\r\n"
    response = addinfourl(
        io.BytesIO(body),
        http.client.parse_headers(io.BytesIO(raw.encode("latin-1"))),
        url,
        code,
    )
    # HTTPErrorProcessor reads `.msg` off real responses.
    response.msg = "OK" if code == 200 else "Found"
    return response


class _FakeWeb:
    """Stand-in for urllib's network layer and DNS.

    ``pages`` maps a full URL to a response factory; anything else (Jina
    included) fails like an unreachable host. Every request is recorded.
    """

    def __init__(self, *, dns: dict[str, list[str]], pages: dict[str, object]) -> None:
        self.dns = dns
        self.pages = pages
        self.requests: list[str] = []

    def open(self, request: Request) -> addinfourl:
        url = request.full_url
        self.requests.append(url)
        factory = self.pages.get(url)
        if factory is None:
            raise URLError("unreachable in test")
        return factory(url)  # type: ignore[operator]

    def getaddrinfo(self, host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        del port, args, kwargs
        if host not in self.dns:
            raise socket.gaierror(socket.EAI_NONAME, "not known")
        results = []
        for address in self.dns[host]:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            sockaddr = (address, 0, 0, 0) if family == socket.AF_INET6 else (address, 0)
            results.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return results


class FetchUrlPublicHostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolExecutionContext(session_id="s_test")

    def _run(self, web: _FakeWeb, url: str, tool: FetchUrlTool | None = None):
        tool = tool or FetchUrlTool()
        with (
            patch("minibot.tools.fetch_url.socket.getaddrinfo", side_effect=web.getaddrinfo),
            patch.object(HTTPHandler, "http_open", side_effect=web.open),
            patch.object(HTTPSHandler, "https_open", side_effect=web.open),
        ):
            return tool.execute(context=self.context, url=url)

    def _assert_blocked(self, result, url: str) -> None:
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "permission_denied")
        self.assertEqual(result.data["url"], url)

    def test_literal_private_addresses_are_rejected_without_requests(self) -> None:
        urls = [
            "http://127.0.0.1:8765/api/sessions",
            "http://[::1]/",
            "http://10.0.0.5/admin",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://0.0.0.0:8765/",
            "http://[::ffff:127.0.0.1]/",
            "http://224.0.0.1/",
            "http://100.100.100.100/",
        ]
        for url in urls:
            with self.subTest(url=url):
                web = _FakeWeb(dns={}, pages={})
                result = self._run(web, url)
                self._assert_blocked(result, url)
                self.assertEqual(web.requests, [])

    def test_local_hostnames_are_rejected_without_requests(self) -> None:
        for url in ["http://localhost:8765/", "http://LOCALHOST./", "http://app.localhost/", "http://printer.local/"]:
            with self.subTest(url=url):
                web = _FakeWeb(dns={}, pages={})
                result = self._run(web, url)
                self._assert_blocked(result, url)
                self.assertEqual(web.requests, [])

    def test_hostname_resolving_to_loopback_is_rejected_without_requests(self) -> None:
        url = "https://totally-public.example/page"
        web = _FakeWeb(dns={"totally-public.example": ["127.0.0.1"]}, pages={})

        result = self._run(web, url)

        self._assert_blocked(result, url)
        # Neither Jina nor a direct fetch was attempted.
        self.assertEqual(web.requests, [])

    def test_hostname_with_any_private_address_is_rejected(self) -> None:
        url = "https://mixed.example/"
        web = _FakeWeb(dns={"mixed.example": ["93.184.216.34", "fd00::1"]}, pages={})

        result = self._run(web, url)

        self._assert_blocked(result, url)
        self.assertEqual(web.requests, [])

    def test_redirect_from_public_host_to_private_address_is_rejected(self) -> None:
        url = "http://public.example/start"
        target = "http://169.254.169.254/latest/meta-data/"
        web = _FakeWeb(
            dns={"public.example": ["93.184.216.34"]},
            pages={url: lambda u: _response(u, 302, {"Location": target})},
        )

        result = self._run(web, url)

        self._assert_blocked(result, url)
        self.assertEqual(result.data["blocked_url"], target)
        self.assertNotIn(target, web.requests)
        # Jina was tried (and failed), then exactly one direct hop.
        self.assertEqual(web.requests[-1], url)

    def test_redirect_to_hostname_resolving_privately_is_rejected(self) -> None:
        url = "http://public.example/start"
        target = "http://internal.example/secret"
        web = _FakeWeb(
            dns={"public.example": ["93.184.216.34"], "internal.example": ["10.1.2.3"]},
            pages={url: lambda u: _response(u, 301, {"Location": target})},
        )

        result = self._run(web, url)

        self._assert_blocked(result, url)
        self.assertNotIn(target, web.requests)

    def test_public_host_is_fetched_directly_after_jina_fails(self) -> None:
        url = "http://public.example/start"
        final = "http://public.example/article"
        html = b"<html><head><title>Public Page</title></head><body><p>hello world</p></body></html>"
        web = _FakeWeb(
            dns={"public.example": ["93.184.216.34"]},
            pages={
                url: lambda u: _response(u, 302, {"Location": final}),
                final: lambda u: _response(u, 200, {"Content-Type": "text/html; charset=utf-8"}, html),
            },
        )

        result = self._run(web, url)

        self.assertTrue(result.ok, result.summary)
        self.assertEqual(result.data["final_url"], final)
        self.assertEqual(result.data["status_code"], 200)
        self.assertEqual(result.data["title"], "Public Page")
        self.assertEqual(result.data["extractor"], "html_parser")
        self.assertIn("hello world", result.content or "")
        self.assertTrue(web.requests[0].startswith("https://r.jina.ai/"))
        self.assertEqual(web.requests[1:], [url, final])

    def test_unresolvable_host_fails_before_direct_fetch(self) -> None:
        url = "http://nowhere.example/"
        web = _FakeWeb(dns={}, pages={})

        result = self._run(web, url)

        # Jina may try (it resolves remotely); the direct fetch must not run.
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "error")
        self.assertIn("无法解析主机", result.summary)
        self.assertNotIn(url, web.requests)

    def test_allow_private_hosts_opt_in_skips_the_check(self) -> None:
        url = "http://127.0.0.1:8765/page"
        web = _FakeWeb(
            dns={},
            pages={url: lambda u: _response(u, 200, {"Content-Type": "text/plain"}, b"local ok")},
        )

        result = self._run(web, url, tool=FetchUrlTool(allow_private_hosts=True))

        self.assertTrue(result.ok, result.summary)
        self.assertEqual(result.content, "local ok")


if __name__ == "__main__":
    unittest.main()
