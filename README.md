# MiniBot

本地命令行 AI agent，基于 OpenAI-compatible `chat.completions`，当前支持：

- tool calling
- 会话持久化与自动 compact
- 用户长期记忆
- skills 按需读取
- MCP tools（`stdio` + `streamable_http`）
- bundled local MCP servers（SQLite demo、macOS system）

## 快速开始

在 `minibot/.env` 写入基础配置：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=
MINIBOT_MODEL=gpt-5.4-mini
MINIBOT_MAX_PARALLEL_TOOLS=4
```

安装依赖并启动：

```bash
cd /Users/jiminyang/Desktop/ai-projects/agent
pip install -r minibot/requirements.txt
python -m minibot
```

如果你当前就在 `.../agent/minibot` 目录里，先 `cd ..` 再运行；这个项目要从包的上一级目录启动。

## MCP 配置

MiniBot 启动时会优先读取当前目录下的 `mcp.json`；如果没有，再回退到 `minibot/mcp.json`。

支持的 transport：

- `stdio`
- `streamable_http`

几个关键规则：

- `enabled: true` 的 server 会在启动时立刻连接并做 tool discovery
- 单个 server 失败只会告警并跳过，不会阻止整体启动
- 工具名统一注册成 `mcp__<server>__<tool>`
- `trusted: true` 的 server 免审批；否则走正常审批流
- `transport.headers` / `transport.env` 支持 `${ENV_VAR}` 占位符
- `transport.cwd` 的相对路径按 `mcp.json` 所在目录解析

默认 bundled servers：

- `sqlite`
  - 启动脚本：`mcp_servers/sqlite_server.py`
  - 默认数据源：`examples/mcp/demo.sqlite3`
  - 可用 `SQLITE_PATH` 覆盖
- `macos_system`
  - 启动脚本：`mcp_servers/macos_system_server.py`
  - 提供 Calendar / Reminders / Notes 能力

最小本地 `stdio` 示例：

```json
{
  "servers": [
    {
      "name": "sqlite",
      "enabled": true,
      "trusted": true,
      "timeout_seconds": 30,
      "transport": {
        "type": "stdio",
        "command": "python3",
        "args": ["mcp_servers/sqlite_server.py"],
        "cwd": ".",
        "env": {}
      }
    }
  ]
}
```

最小远端 `streamable_http` 示例：

```json
{
  "servers": [
    {
      "name": "figma",
      "enabled": true,
      "trusted": false,
      "timeout_seconds": 30,
      "transport": {
        "type": "streamable_http",
        "url": "https://example.com/mcp",
        "headers": {
          "Authorization": "Bearer ${FIGMA_MCP_TOKEN}"
        }
      }
    }
  ]
}
```

## 常用命令

- `/sessions` 查看会话列表
- `/new` 新建会话
- `/resume <id>` 恢复会话
- `/delete <id|current>` 删除会话
- `/compact` 手动压缩当前会话
- `/memory` 查看长期记忆
- `/memory clear` 清空长期记忆
- `/memory forget <id>` 删除单条长期记忆
- `/skills` 查看可用 skills
- `/help` 显示帮助

## 测试

在 `minibot/` 目录运行：

```bash
python -m unittest discover -s tests
```

只跑 MCP 相关：

```bash
python -m unittest tests.test_mcp_config tests.test_mcp_client tests.test_mcp_manager tests.test_mcp_tools
```

## 说明

- `read_skill` 读取到的正文不会进入 system prompt，只会作为当前会话里的 tool result
- 对模型来说，MCP tool 和本地 tool 没区别；它看到的是统一后的 tool schema
- SQLite 路径、数据库账户、AppleScript 细节这类底层配置属于各自 MCP server，不属于 agent 本身
