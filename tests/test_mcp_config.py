from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minibot.bootstrap import _resolve_mcp_config
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

    def test_resolves_stdio_command_and_args_env_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "mcp.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "servers": [
                        {
                          "name": "bundled",
                          "enabled": true,
                          "transport": {
                            "type": "stdio",
                            "command": "${MINIBOT_PYTHON}",
                            "args": ["${SERVER_ROOT}/server.py", "--flag"],
                            "env": {}
                          }
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "MINIBOT_PYTHON": "/tmp/venv/bin/python",
                    "SERVER_ROOT": "/tmp/minibot",
                },
                clear=False,
            ):
                result = load_mcp_config(workspace)

            self.assertEqual(result.warnings, [])
            self.assertEqual(len(result.servers), 1)
            server = result.servers[0]
            self.assertIsInstance(server.transport, StdioTransportConfig)
            assert isinstance(server.transport, StdioTransportConfig)
            self.assertEqual(server.transport.command, "/tmp/venv/bin/python")
            self.assertEqual(
                server.transport.args,
                ("/tmp/minibot/server.py", "--flag"),
            )

    def test_resolve_global_mcp_config_prefers_env_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            env_dir = root / "env-config"
            home = root / "home"
            package_dir.mkdir()
            env_dir.mkdir()
            (package_dir / "mcp.json").write_text('{"servers":[]}', encoding="utf-8")
            (env_dir / "mcp.json").write_text('{"servers":[]}', encoding="utf-8")

            with patch.dict(
                os.environ,
                {"MINIBOT_MCP_CONFIG_PATH": str(env_dir / "mcp.json")},
                clear=False,
            ), patch.object(Path, "home", return_value=home):
                config_root, config_path, source = _resolve_mcp_config(package_dir)

            self.assertEqual(config_root, env_dir.resolve())
            self.assertEqual(config_path, (env_dir / "mcp.json").resolve())
            self.assertEqual(source, "env")

    def test_resolve_global_mcp_config_prefers_user_over_bundled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            home = root / "home"
            user_config_dir = home / ".minibot"
            package_dir.mkdir()
            user_config_dir.mkdir(parents=True)
            (package_dir / "mcp.json").write_text('{"servers":[]}', encoding="utf-8")
            (user_config_dir / "mcp.json").write_text('{"servers":[]}', encoding="utf-8")

            with patch.dict(
                os.environ,
                {"MINIBOT_MCP_CONFIG_PATH": ""},
                clear=False,
            ), patch.object(Path, "home", return_value=home):
                config_root, config_path, source = _resolve_mcp_config(package_dir)

            self.assertEqual(config_root, user_config_dir.resolve())
            self.assertEqual(config_path, (user_config_dir / "mcp.json").resolve())
            self.assertEqual(source, "user")

    def test_resolve_global_mcp_config_falls_back_to_bundled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_dir = root / "package"
            home = root / "home"
            package_dir.mkdir()
            (package_dir / "mcp.json").write_text('{"servers":[]}', encoding="utf-8")

            with patch.dict(
                os.environ,
                {"MINIBOT_MCP_CONFIG_PATH": ""},
                clear=False,
            ), patch.object(Path, "home", return_value=home):
                config_root, config_path, source = _resolve_mcp_config(package_dir)

            self.assertEqual(config_root, package_dir.resolve())
            self.assertEqual(config_path, (package_dir / "mcp.json").resolve())
            self.assertEqual(source, "bundled")


if __name__ == "__main__":
    unittest.main()
