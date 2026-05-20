from __future__ import annotations

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import shlex
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.artifacts import ArtifactStore
from minibot.runtime.tool_output_materializer import ToolOutputMaterializer
from minibot.tools.base import ToolExecutionContext
from minibot.tools.edit_file import EditFileTool
from minibot.tools.exec_cmd import ExecTool
from minibot.tools.fetch_url import FetchUrlTool, _FetchedPage
from minibot.tools.read_artifact import ReadArtifactTool
from minibot.tools.read_file import ReadFileTool
from minibot.tools.search_files import SearchFilesTool
from minibot.tools.write_file import WriteFileTool

_INLINE = ToolOutputMaterializer._INLINE_CONTENT_CHARS
_PREVIEW = ToolOutputMaterializer._PREVIEW_CHARS


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _ArticleHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = (
            "<html><head>"
            "<title>China Tech Daily</title>"
            '<meta property="article:published_time" content="2026-04-17T12:30:00Z">'
            "</head><body><article><h1>China Tech Daily</h1><p>"
            + ("agent news " * 1400)
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
    def _materialize(self, workspace: Path, context: ToolExecutionContext, output):
        return ToolOutputMaterializer(ArtifactStore(workspace)).materialize(
            output,
            context=context,
        )

    def test_read_file_large_content_returns_preview_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            (workspace / "long.txt").write_text("a" * (_INLINE + 500), encoding="utf-8")

            output = ReadFileTool(workspace=workspace).execute(context=context, path="long.txt")
            result = self._materialize(workspace, context, output)

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertIsNotNone(result.artifact)
            self.assertEqual(len(result.data["preview"]), _PREVIEW)
            self.assertNotIn("content", result.data)

            artifact_result = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertEqual(artifact_result.data["total_chars"], _INLINE + 500)

    def test_exec_long_output_and_nonzero_exit_code_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            tool = ExecTool(workspace=workspace)
            command = (
                f"{shlex.quote(sys.executable)} -c "
                "\"import sys; print('x'*13000); sys.stderr.write('err'); sys.exit(1)\""
            )

            result = self._materialize(workspace, context, tool.execute(context=context, command=command))

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "error")
            self.assertEqual(result.data["exit_code"], 1)
            self.assertTrue(result.truncated)
            self.assertIn("stdout_preview", result.data)
            self.assertIsNotNone(result.artifact)

            artifact_result = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
                limit=8000,
            )
            total = artifact_result.data["total_chars"]
            tail = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
                offset=max(0, total - 100),
                limit=100,
            )
            self.assertIn("[stderr]\nerr", tail.data["content"])

    def test_exec_zero_exit_code_remains_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = ExecTool(workspace=workspace)
            command = f"{shlex.quote(sys.executable)} -c \"print('ok')\""

            result = self._materialize(workspace, context, tool.execute(context=context, command=command))

            self.assertTrue(result.ok)
            self.assertEqual(result.code, "success")
            self.assertEqual(result.data["exit_code"], 0)
            self.assertEqual(result.data["stdout"], "ok\n")

    def test_search_files_many_matches_returns_preview_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            content = "\n".join(
                f"needle line {idx} " + ("x" * 40)
                for idx in range(200)
            )
            (workspace / "matches.txt").write_text(content, encoding="utf-8")
            tool = SearchFilesTool(workspace=workspace)

            result = self._materialize(workspace, context, tool.execute(context=context, pattern="needle", path="."))

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertEqual(result.data["total_matches"], 200)
            self.assertEqual(len(result.data["matches"]), 50)
            self.assertIsNotNone(result.artifact)

            artifact_result = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("matches.txt:1: needle line 0", artifact_result.data["content"])
            self.assertTrue(artifact_result.data["has_more"])

    def test_fetch_url_extracts_readable_text_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _ArticleHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                tool = FetchUrlTool()
                result = self._materialize(
                    workspace,
                    context,
                    tool.execute(
                        context=context,
                        url=f"http://127.0.0.1:{server.server_address[1]}/article",
                    ),
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

            artifact_result = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("agent news", artifact_result.data["content"])

    def test_fetch_url_prefers_jina_for_public_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = FetchUrlTool()

            with (
                patch.object(
                    FetchUrlTool,
                    "_fetch_via_jina",
                    return_value=_FetchedPage(
                        final_url="https://example.com/post",
                        status_code=200,
                        content_type="text/markdown",
                        body_text="headline\n\n" + ("detail " * 2000),
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
                result = self._materialize(
                    workspace,
                    context,
                    tool.execute(
                        context=context,
                        url="https://example.com/post",
                    ),
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["extractor"], "jina")
            self.assertTrue(result.truncated)
            self.assertIsNotNone(result.artifact)
            mock_jina.assert_called_once()

    def test_fetch_url_prefers_google_news_rss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = FetchUrlTool()

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
                result = self._materialize(
                    workspace,
                    context,
                    tool.execute(
                        context=context,
                        url="https://news.google.com/topics/test?hl=en-US&ceid=US:en",
                    ),
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["extractor"], "google_news_rss")
            self.assertIn("content", result.data)
            mock_rss.assert_called_once()

    def test_fetch_url_large_page_returns_partial_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _HugeArticleHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                tool = FetchUrlTool()
                result = self._materialize(
                    workspace,
                    context,
                    tool.execute(
                        context=context,
                        url=f"http://127.0.0.1:{server.server_address[1]}/huge",
                    ),
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
            self.assertIn("artifact", result.summary)

            artifact_result = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertIn("topic item", artifact_result.data["content"])


class FileHashTests(unittest.TestCase):
    """Cover the sha256 optimistic lock around ``write_file``."""

    def _materialize(self, workspace: Path, context: ToolExecutionContext, output):
        return ToolOutputMaterializer(ArtifactStore(workspace)).materialize(
            output,
            context=context,
        )

    def test_read_file_small_returns_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            body = "hello world\n"
            (workspace / "note.txt").write_text(body, encoding="utf-8")

            output = ReadFileTool(workspace=workspace).execute(
                context=context,
                path="note.txt",
            )
            result = self._materialize(workspace, context, output)

            self.assertTrue(result.ok)
            self.assertFalse(result.truncated)
            self.assertEqual(result.data["file_sha256"], _sha(body))

    def test_read_file_large_flows_hash_through_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            store = ArtifactStore(workspace)
            context = ToolExecutionContext(session_id="s_test")
            body = "a" * (_INLINE + 500)
            (workspace / "long.txt").write_text(body, encoding="utf-8")

            output = ReadFileTool(workspace=workspace).execute(
                context=context,
                path="long.txt",
            )
            result = self._materialize(workspace, context, output)

            self.assertTrue(result.truncated)
            self.assertEqual(result.data["file_sha256"], _sha(body))

            artifact_result = ReadArtifactTool(store).execute(
                context=context,
                artifact_id=result.artifact.id,
            )
            self.assertEqual(artifact_result.data["file_sha256"], _sha(body))

    def test_write_new_file_does_not_require_expected_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = WriteFileTool(workspace=workspace)

            new_body = "fresh\n"
            result = tool.execute(context=context, path="new.txt", content=new_body)

            self.assertTrue(result.ok)
            self.assertEqual(result.data["file_sha256"], _sha(new_body))
            self.assertTrue((workspace / "new.txt").exists())

    def test_write_existing_file_with_matching_hash_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = WriteFileTool(workspace=workspace)
            original = "one"
            (workspace / "a.txt").write_text(original, encoding="utf-8")

            new_body = "two"
            result = tool.execute(
                context=context,
                path="a.txt",
                content=new_body,
                expected_sha256=_sha(original),
            )

            self.assertTrue(result.ok)
            # The returned hash must be for the new content, not the old one.
            self.assertEqual(result.data["file_sha256"], _sha(new_body))
            self.assertEqual(
                (workspace / "a.txt").read_text(encoding="utf-8"),
                new_body,
            )

    def test_write_existing_file_with_wrong_hash_returns_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = WriteFileTool(workspace=workspace)
            original = "one"
            (workspace / "a.txt").write_text(original, encoding="utf-8")

            result = tool.execute(
                context=context,
                path="a.txt",
                content="two",
                expected_sha256="deadbeef" * 8,
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "conflict")
            self.assertEqual(result.data["expected_sha256"], "deadbeef" * 8)
            self.assertEqual(result.data["current_sha256"], _sha(original))
            # File must remain untouched.
            self.assertEqual(
                (workspace / "a.txt").read_text(encoding="utf-8"),
                original,
            )

    def test_write_existing_file_without_hash_returns_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = WriteFileTool(workspace=workspace)
            original = "one"
            (workspace / "a.txt").write_text(original, encoding="utf-8")

            result = tool.execute(
                context=context,
                path="a.txt",
                content="two",
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "conflict")
            self.assertEqual(result.data["current_sha256"], _sha(original))
            self.assertNotIn("expected_sha256", result.data)
            self.assertEqual(
                (workspace / "a.txt").read_text(encoding="utf-8"),
                original,
            )


class EditFileTests(unittest.TestCase):
    """Cover the hash-guarded, line-based ``edit_file`` contract."""

    def test_edit_file_replace_range_with_matching_hash_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = EditFileTool(workspace=workspace)
            original = "alpha\nbeta\nrepeat\nrepeat\ngamma\n"
            (workspace / "note.txt").write_text(original, encoding="utf-8")

            result = tool.execute(
                context=context,
                path="note.txt",
                expected_sha256=_sha(original),
                edits=[
                    {
                        "op": "replace",
                        "start_line": 4,
                        "end_line": 4,
                        "old_text": "repeat\n",
                        "new_text": "delta\n",
                    }
                ],
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["previous_sha256"], _sha(original))
            updated = "alpha\nbeta\nrepeat\ndelta\ngamma\n"
            self.assertEqual((workspace / "note.txt").read_text(encoding="utf-8"), updated)
            self.assertEqual(result.data["file_sha256"], _sha(updated))

    def test_edit_file_insert_and_append_in_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = EditFileTool(workspace=workspace)
            original = "first\nthird\n"
            (workspace / "flow.txt").write_text(original, encoding="utf-8")

            result = tool.execute(
                context=context,
                path="flow.txt",
                expected_sha256=_sha(original),
                edits=[
                    {
                        "op": "insert_after",
                        "line": 1,
                        "new_text": "second\n",
                    },
                    {
                        "op": "append",
                        "new_text": "fourth\n",
                    },
                ],
            )

            self.assertTrue(result.ok)
            self.assertEqual(
                (workspace / "flow.txt").read_text(encoding="utf-8"),
                "first\nsecond\nthird\nfourth\n",
            )

    def test_edit_file_with_stale_hash_returns_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = EditFileTool(workspace=workspace)
            original = "one\ntwo\n"
            path = workspace / "a.txt"
            path.write_text(original, encoding="utf-8")
            stale_sha = _sha(original)
            path.write_text("one\nchanged\n", encoding="utf-8")

            result = tool.execute(
                context=context,
                path="a.txt",
                expected_sha256=stale_sha,
                edits=[
                    {
                        "op": "replace",
                        "start_line": 2,
                        "end_line": 2,
                        "old_text": "two\n",
                        "new_text": "three\n",
                    }
                ],
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "conflict")
            self.assertEqual(result.data["expected_sha256"], stale_sha)
            self.assertEqual(result.data["current_sha256"], _sha("one\nchanged\n"))
            self.assertEqual(path.read_text(encoding="utf-8"), "one\nchanged\n")

    def test_edit_file_rejects_wrong_old_text_for_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = EditFileTool(workspace=workspace)
            original = "line1\nline2\n"
            path = workspace / "mismatch.txt"
            path.write_text(original, encoding="utf-8")

            result = tool.execute(
                context=context,
                path="mismatch.txt",
                expected_sha256=_sha(original),
                edits=[
                    {
                        "op": "replace",
                        "start_line": 2,
                        "end_line": 2,
                        "old_text": "wrong\n",
                        "new_text": "lineX\n",
                    }
                ],
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "invalid_args")
            self.assertIn("actual_text_preview", result.data)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_edit_file_rejects_overlapping_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            context = ToolExecutionContext(session_id="s_test")
            tool = EditFileTool(workspace=workspace)
            original = "a\nb\nc\n"
            path = workspace / "overlap.txt"
            path.write_text(original, encoding="utf-8")

            result = tool.execute(
                context=context,
                path="overlap.txt",
                expected_sha256=_sha(original),
                edits=[
                    {
                        "op": "replace",
                        "start_line": 2,
                        "end_line": 2,
                        "old_text": "b\n",
                        "new_text": "beta\n",
                    },
                    {
                        "op": "insert_after",
                        "line": 2,
                        "new_text": "after-beta\n",
                    },
                ],
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.code, "invalid_args")
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
