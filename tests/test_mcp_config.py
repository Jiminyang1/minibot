from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.mcp_host.config import load_mcp_config
from minibot.mcp_host.models import StdioTransportConfig, StreamableHTTPTransportConfig


class MCPConfigTests(unittest.TestCase):
    def test_loads_valid_servers_and_resolves_env_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "mcp.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "servers": [
                        {
                          "name": "figma",
                          "enabled": true,
                          "trusted": true,
                          "timeout_seconds": 45,
                          "transport": {
                            "type": "streamable_http",
                            "url": "https://figma.example.com/mcp",
                            "headers": {
                              "Authorization": "Bearer ${FIGMA_TOKEN}"
                            }
                          }
                        },
                        {
                          "name": "gmail",
                          "enabled": true,
                          "trusted": false,
                          "timeout_seconds": 30,
                          "transport": {
                            "type": "streamable_http",
                            "url": "https://gmail.example.com/mcp",
                            "headers": {}
                          }
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"FIGMA_TOKEN": "secret-token"}, clear=False):
                result = load_mcp_config(workspace)

            self.assertEqual(result.warnings, [])
            self.assertEqual([server.name for server in result.servers], ["figma", "gmail"])
            self.assertEqual(
                result.servers[0].transport.headers,
                {"Authorization": "Bearer secret-token"},
            )
            self.assertTrue(result.servers[0].trusted)
            self.assertEqual(result.servers[0].timeout_seconds, 45)
            self.assertIsInstance(result.servers[0].transport, StreamableHTTPTransportConfig)

    def test_missing_env_soft_fails_enabled_server_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "mcp.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "servers": [
                        {
                          "name": "figma",
                          "enabled": true,
                          "transport": {
                            "type": "streamable_http",
                            "url": "https://figma.example.com/mcp",
                            "headers": {
                              "Authorization": "Bearer ${FIGMA_TOKEN}"
                            }
                          }
                        },
                        {
                          "name": "gmail",
                          "enabled": true,
                          "transport": {
                            "type": "streamable_http",
                            "url": "https://gmail.example.com/mcp",
                            "headers": {}
                          }
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=False):
                result = load_mcp_config(workspace)

            self.assertEqual([server.name for server in result.servers], ["gmail"])
            self.assertEqual(len(result.warnings), 1)
            self.assertIn("FIGMA_TOKEN", result.warnings[0])

    def test_disabled_server_does_not_require_env_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "mcp.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "servers": [
                        {
                          "name": "figma",
                          "enabled": false,
                          "transport": {
                            "type": "streamable_http",
                            "url": "https://figma.example.com/mcp",
                            "headers": {
                              "Authorization": "Bearer ${FIGMA_TOKEN}"
                            }
                          }
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=False):
                result = load_mcp_config(workspace)

            self.assertEqual(result.warnings, [])
            self.assertEqual(len(result.servers), 1)
            self.assertFalse(result.servers[0].enabled)

    def test_loads_stdio_transport_with_env_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "mcp.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "servers": [
                        {
                          "name": "filesystem",
                          "enabled": true,
                          "transport": {
                            "type": "stdio",
                            "command": "python",
                            "args": ["server.py"],
                            "cwd": ".",
                            "env": {
                              "ROOT": "${WORKSPACE_ROOT}"
                            }
                          }
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"WORKSPACE_ROOT": "/tmp/work"}, clear=False):
                result = load_mcp_config(workspace)

            self.assertEqual(result.warnings, [])
            self.assertEqual(len(result.servers), 1)
            server = result.servers[0]
            self.assertIsInstance(server.transport, StdioTransportConfig)
            assert isinstance(server.transport, StdioTransportConfig)
            self.assertEqual(server.transport.command, "python")
            self.assertEqual(server.transport.args, ("server.py",))
            self.assertEqual(server.transport.cwd, str(workspace.resolve()))
            self.assertEqual(server.transport.env, {"ROOT": "/tmp/work"})


if __name__ == "__main__":
    unittest.main()
