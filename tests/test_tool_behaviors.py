from __future__ import annotations

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.session import SessionManager
from minibot.tools.base import ToolExecutionContext
from minibot.tools.exec_cmd import ExecTool
from minibot.tools.fetch_url import FetchUrlTool, _FetchedPage
from minibot.tools.read_artifact import ReadArtifactTool
from minibot.tools.read_file import ReadFileTool
from minibot.tools.search_files import SearchFilesTool


class _ArticleHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = (
            "<html><head>"
            "<title>China Tech Daily</title>"
            '<meta property="article:published_time" content="2026-04-17T12:30:00Z">'
            "</head><body><article><h1>China Tech Daily</h1><p>"
            + ("agent news " * 400)
            + "</p></article></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _HugeArticleHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = (
            "<html><head><title>Huge Feed</title></head><body><article><h1>Huge Feed</h1><p>"
            + ("topic item " * 120000)
            + "</p></article></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class ToolBehaviorTests(unittest.TestCase):
    def test_read_file_large_content_returns_preview_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            (workspace / "long.txt").write_text("a" * 2500, encoding="utf-8")

            result = ReadFileTool(
                workspace=workspace,
                session_manager=manager,
            ).execute(context=context, path="long.txt")

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertIsNotNone(result.artifact)
            self.assertEqual(len(result.data["preview"]), 2000)
            self.assertNotIn("content", result.data)

            artifact_result = ReadArtifactTool(manager).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertEqual(artifact_result.data["total_chars"], 2500)

    def test_exec_long_output_and_nonzero_exit_code_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            tool = ExecTool(workspace=workspace, session_manager=manager)
            command = (
                f"{shlex.quote(sys.executable)} -c "
                "\"import sys; print('x'*2500); sys.stderr.write('err'); sys.exit(1)\""
            )

            result = tool.execute(context=context, command=command)

            self.assertTrue(result.ok)
            self.assertEqual(result.data["exit_code"], 1)
            self.assertTrue(result.truncated)
            self.assertIn("stdout_preview", result.data)
            self.assertIsNotNone(result.artifact)

            artifact_result = ReadArtifactTool(manager).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("[stderr]\nerr", artifact_result.data["content"])

    def test_search_files_many_matches_returns_preview_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            content = "\n".join(
                f"needle line {idx}"
                for idx in range(60)
            )
            (workspace / "matches.txt").write_text(content, encoding="utf-8")
            tool = SearchFilesTool(workspace=workspace, session_manager=manager)

            result = tool.execute(context=context, pattern="needle", path=".")

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertEqual(result.data["total_matches"], 60)
            self.assertEqual(len(result.data["matches"]), 50)
            self.assertIsNotNone(result.artifact)

            artifact_result = ReadArtifactTool(manager).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("matches.txt:60: needle line 59", artifact_result.data["content"])

    def test_fetch_url_extracts_readable_text_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _ArticleHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                tool = FetchUrlTool(manager)
                result = tool.execute(
                    context=context,
                    url=f"http://127.0.0.1:{server.server_address[1]}/article",
                )
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertEqual(result.data["title"], "China Tech Daily")
            self.assertEqual(result.data["published_at"], "2026-04-17T12:30:00Z")
            self.assertIsNotNone(result.artifact)
            self.assertIn("agent news", result.data["preview"])

            artifact_result = ReadArtifactTool(manager).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("agent news", artifact_result.data["content"])

    def test_fetch_url_prefers_jina_for_public_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            tool = FetchUrlTool(manager)

            with (
                patch.object(
                    FetchUrlTool,
                    "_fetch_via_jina",
                    return_value=_FetchedPage(
                        final_url="https://example.com/post",
                        status_code=200,
                        content_type="text/markdown",
                        body_text="headline\n\n" + ("detail " * 800),
                        title="Example Headline",
                        extractor="jina",
                    ),
                ) as mock_jina,
                patch.object(
                    FetchUrlTool,
                    "_fetch_direct",
                    side_effect=AssertionError("should not fall back to direct fetch"),
                ),
            ):
                result = tool.execute(
                    context=context,
                    url="https://example.com/post",
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["extractor"], "jina")
            self.assertTrue(result.truncated)
            self.assertIsNotNone(result.artifact)
            mock_jina.assert_called_once()

    def test_fetch_url_prefers_google_news_rss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            tool = FetchUrlTool(manager)

            with (
                patch.object(
                    FetchUrlTool,
                    "_fetch_google_news_rss",
                    return_value=_FetchedPage(
                        final_url="https://news.google.com/rss/topics/test",
                        status_code=200,
                        content_type="application/rss+xml",
                        body_text="1. Headline\n来源: Example\n链接: https://example.com/story",
                        title="Google News RSS",
                        extractor="google_news_rss",
                    ),
                ) as mock_rss,
                patch.object(
                    FetchUrlTool,
                    "_fetch_via_jina",
                    side_effect=AssertionError("should not fall back to jina"),
                ),
                patch.object(
                    FetchUrlTool,
                    "_fetch_direct",
                    side_effect=AssertionError("should not fall back to direct fetch"),
                ),
            ):
                result = tool.execute(
                    context=context,
                    url="https://news.google.com/topics/test?hl=en-US&ceid=US:en",
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["extractor"], "google_news_rss")
            self.assertIn("content", result.data)
            mock_rss.assert_called_once()

    def test_fetch_url_large_page_returns_partial_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manager = SessionManager(workspace)
            context = ToolExecutionContext(session_id="s_test")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _HugeArticleHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                tool = FetchUrlTool(manager)
                result = tool.execute(
                    context=context,
                    url=f"http://127.0.0.1:{server.server_address[1]}/huge",
                )
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertTrue(result.data["byte_truncated"])
            self.assertEqual(result.data["title"], "Huge Feed")
            self.assertIsNotNone(result.artifact)
            self.assertIn("页面较大，仅提取前段内容", result.summary)

            artifact_result = ReadArtifactTool(manager).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("topic item", artifact_result.data["content"])


if __name__ == "__main__":
    unittest.main()
