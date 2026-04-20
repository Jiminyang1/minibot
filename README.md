# MiniBot

本地命令行 AI agent，基于 OpenAI-compatible `chat.completions`。同步 turn loop + tool calling，把本地工具、MCP server、skills、长期记忆统一接在一起。

主要能力：

- tool calling（本地工具 + MCP 工具统一 schema）
- 会话持久化 + 超阈值自动 compact
- 跨会话的用户长期记忆
- Skills 按需加载（L1 元数据常驻 + L2 正文懒读）
- MCP 支持 `stdio` / `streamable_http`
- 随项目自带的本地 MCP servers：SQLite demo、macOS system

## 架构概览

数据流自上而下：REPL 把用户输入交给 `TurnEngine`；`ContextManager` 先把长期记忆 / skill L1 / 工具 schema 组装成 prompt；`AgentRunner` 再拿着 prompt 跟 LLM 一来一回，模型返回 `tool_call` 时去 `ToolRegistry` 查到对应的 `Tool` 执行；tool 的结果回灌到 history，直到这一轮没有新的 tool call，`TurnEngine` 把消息写回会话存储。

```
   ┌──────────────┐
   │  REPL / UI   │  user input · approvals · printing
   └──────┬───────┘
          │
   ┌──────▼───────┐                           ┌──────────────┐
   │  TurnEngine  │─── 追加消息 ───────────────►│ Session     │
   │ 单轮协调器     │                           │  Store       │
   └──────┬───────┘                           └──────────────┘
          │ 1. build context
          │
   ┌──────▼────────────┐        ┌──────────────────────────────┐
   │  ContextManager   │◄───────│  UserMemoryStore             │
   │ · system prompt   │◄───────│  SkillRegistry  (L1 元数据)   │
   │ · 历史 / compact   │◄───────│  ToolRegistry  (tool schema) │
   └──────┬────────────┘        └──────────────────────────────┘
          │ 2. prompt + tool schemas
          │
   ┌──────▼────────────┐    chat.completion     ┌──────────┐
   │   AgentRunner     │◄──────────────────────►│   LLM    │
   │  LLM ↔ tool loop  │                        └──────────┘
   └──────┬────────────┘
          │ 3. tool_call(name, args) → 查注册表并执行
          │
   ┌──────▼───────────┐
   │  ToolRegistry    │   对 LLM：本地 / MCP tool 统一接口
   └──────┬───────────┘
          │
    ┌─────┴──────────────────────────┐
    │                                │
┌───▼─────────┐              ┌───────▼──────────┐
│ Local Tools │              │  MCPToolProxy    │
│ fs/exec/... │              │  (mcp_host)      │
│ read_skill  │              └───────┬──────────┘
│ (→ L2 body) │                      │  stdio / streamable_http
└─────────────┘                      │
                           ┌─────────┴─────────┐
                           │                   │
                     ┌─────▼──────┐     ┌──────▼───────┐
                     │  bundled   │     │  remote MCP  │
                     │  servers   │     │  servers     │
                     │  (sqlite / │     │  (HTTP)      │
                     │   macOS)   │     │              │
                     └────────────┘     └──────────────┘
```

几个关键约束：

- 主 turn loop 是**同步**的；异步只存在于 MCP client 的后台线程，对上层透明。
- 对模型来说本地 tool 和 MCP tool 没差别，都是 `ToolRegistry` 里同一种 `Tool`；MCP 工具统一命名 `mcp__<server>__<tool>`。
- function call / tool / MCP 是三层：function call 是模型层调用格式，tool 是 MiniBot 暴露给模型的能力对象，MCP 是外部能力接入协议。

## 核心模块

### Core runtime (`runtime/`)

一次用户输入 → `TurnEngine.run_turn` 协调一次 turn：

- `ContextManager` 组装 system prompt（基础 prompt + 长期记忆 + skill L1 元数据 + 工具 schema），管理历史，超过 token 阈值时调用 summarizer 压缩。
- `AgentRunner` 跑 LLM ↔ tool 的循环；一次响应里的多个 tool call 并发执行（受 `max_parallel_tools` 限制），非 `trusted` 的 tool 走审批。
- `ToolOutputMaterializer` 把体积大的 tool 输出落到 `ArtifactStore`，返回给模型的只是引用，避免撑爆上下文。

### Tools (`tools/`)

工具按能力域分成 toolset 工厂函数（见 `tools/__init__.py`），在 `__main__.py` 里组合注入：

- `filesystem_toolset`：`read_file` / `write_file` / `edit_file` / `list_dir` / `search_files` / `read_artifact`
- `shell_toolset`：`exec`
- `network_toolset`：`web_search` / `fetch_url`
- `memory_toolset`：`remember` / `forget`
- `skill_toolset`：`read_skill`

所有工具都实现同一个 `Tool` 接口，统一注册进 `ToolRegistry`。MCP 工具通过 `MCPToolProxy` 伪装成同一种接口接入。

### MCP (`mcp_host/` + `mcp_servers/`)

MCP 在 MiniBot 里是统一的外部能力接入层，分两边：

- `mcp_host/`：客户端侧，包含配置解析、transport（`stdio` / `streamable_http`）、host 生命周期管理，以及把每个 MCP tool 包成本地 `Tool` 的 `MCPToolProxy`。
- `mcp_servers/`：本地自带的 MCP server 实现，每个 server 自包含一个目录/文件：
  - `sqlite_server.py`：只读 SQLite demo
  - `macos_system/`：Calendar / Reminders / Notes（AppleScript 桥）

运行模型：每个 MCP server 一个后台 asyncio loop thread，`Tool.execute()` 通过 `run_coroutine_threadsafe` 同步等结果。`stdio` server 由 MiniBot 起子进程；`streamable_http` 只是连接已有的远端 server，不由 MiniBot 启动。

### Session (`session/`)

每个会话对应 `<workspace>/.minibot/sessions/<session_id>/`：

- `meta.json`：标题、创建/更新时间
- `messages.jsonl`：逐条消息（含 tool call / tool result）append 写入

`SessionManager` 负责 create / list / resume / delete / rename；compact 结果落回同一份文件。

### Memory (`user_memory.py`)

跨会话的**全局**长期记忆（不是 session 级）：

- 存储：`~/.minibot/user_memory.json`
- 结构：一组 `{id, content, created_at}`
- 每轮由 `ContextManager` 塞进 system prompt 头部
- 模型通过 `remember` / `forget` 自行维护

适合稳定的用户事实（身份、偏好、常驻路径），不是临时 scratchpad。

### Skills (`skills/`)

参考 Anthropic 的 progressive disclosure：

- **L1 元数据**（`name` / `description` / `tools`）随 system prompt 注入，便宜、常驻。
- **L2 正文**只有模型调用 `read_skill` 时才作为 tool result 进入当前会话历史；history 被裁剪或 compact 后会消失或被摘要化。

所以 skill 更像"按需拉取的工作流说明"，不是永久内置规则。写法和已有的 `skills/*.md` 保持一致即可。

### 可观测性（`run_log.py` + `artifacts.py` + `event_handler`）

MiniBot 把"一次 turn 发生了什么"分成**运行期事件流**和**事后落盘记录**两类，职责分开。

#### 运行期事件流（in-process callback）

核心组件（`TurnEngine` / `AgentRunner` / `MCPHost` / MCP transport）都接受一个 `event_handler: Callable[[str], None]` 回调。在 CLI 场景下 `__main__.py` 里注入的是 `ui.tool_log`，把事件实时打到终端；也可以换成别的实现（结构化 logger、Web UI、OpenTelemetry 等），完全不碰业务代码。

目前会 emit 的事件大致包括：

- **Turn 级**：开始处理 / 每轮 LLM 请求与最终回答耗时 / 达到最大迭代 / compact 触发与结果
- **Tool 级**：每次 tool 调用的参数预览、返回摘要；MCP 调用会额外标注成 `mcp__<server>__<tool>`
- **MCP 生命周期**（`MCPHost`）：server 连接成功 / 发现的 tool 数量 / 某个 server 初始化失败跳过 / 工具名称冲突跳过
- **MCP transport**：子进程 stderr、连接异常、重试等底层事件

这一层是纯内存 callback，不持久化。它负责"现在正在发生什么、卡在哪"。

#### 事后落盘记录

- **原始消息历史** → `<workspace>/.minibot/sessions/<id>/messages.jsonl`
  - 每条 message（含 tool call / tool result）按出现顺序 append 写入，是真相来源，用于 resume。
- **结构化运行摘要** → `<workspace>/.minibot/runs.jsonl`
  - `RunLogStore` 在每轮结束时追加一条 `RunLogRecord`，字段包含：
    - `run_id` / `session_id` / `turn_index` / `timestamp` / `duration_ms`
    - `status`（`success` / `failed`）、`model`、`user_input_preview` / `final_reply_preview`
    - token 用量（`input_tokens` / `output_tokens` / `total_tokens`）
    - `llm_call_count` / `tool_call_count` / `tools_used`
    - MCP 维度：`mcp_tool_call_count` / `mcp_servers_used` / `mcp_transports_used` / `mcp_error_count`
    - compact 标记：`did_compact` / `compact_message`
    - 失败时的 `error_type` / `error_message_preview`
  - append-only JSONL，方便事后 `jq` / pandas 做用量统计与失败分析。
- **大体积 tool 输出** → `<workspace>/.minibot/artifacts/`
  - `ToolOutputMaterializer` 把超阈值的 tool 结果落到 `ArtifactStore`，只把引用喂给模型；模型需要全文时用 `read_artifact` 按需拉取，避免撑爆上下文。

两类放在一起看：**事件流**解决"运行中"的问题，**落盘记录**解决"运行后"的问题。

## 快速开始

项目本身是一个 Python 包，**从包的上一级目录**启动。

在 `minibot/.env` 写入基础配置：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=
MINIBOT_MODEL=gpt-5.4-mini
```

安装依赖并启动：

```bash
cd /Users/jiminyang/Desktop/ai-projects/agent
pip install -r minibot/requirements.txt
python -m minibot
```

## 配置

### Agent 侧（`.env` / 环境变量）

对应 `config.py:Config.from_env`：

| 变量 | 默认 | 作用 |
|---|---|---|
| `OPENAI_API_KEY` | — | 必填 |
| `OPENAI_BASE_URL` | 官方 | 指向 proxy / self-hosted endpoint |
| `MINIBOT_MODEL` | `gpt-5.4-mini` | 模型名 |
| `MINIBOT_MAX_ITERATIONS` | `20` | 单轮 turn 内 LLM ↔ tool 最大循环次数 |
| `MINIBOT_MAX_PARALLEL_TOOLS` | `4` | 同一响应多 tool call 并发上限 |
| `MINIBOT_AUTO_APPROVE` | `false` | `true` 时跳过审批 |
| `MINIBOT_MAX_HISTORY_TURNS` | `40` | compact 前允许的最大历史 turn |
| `MINIBOT_COMPACT_TOKEN_THRESHOLD` | `40000` | 触发自动 compact 的 token 阈值 |
| `MINIBOT_RESERVED_COMPLETION_TOKENS` | `4096` | 留给 completion 的 token 预算 |
| `MINIBOT_COMPACT_KEEP_RECENT` | `10` | compact 时保留的最近消息数 |

持久化路径（按约定生成，无需配置）：

- 会话消息：`<workspace>/.minibot/sessions/<session_id>/messages.jsonl`
- 运行摘要：`<workspace>/.minibot/runs.jsonl`
- Artifacts：`<workspace>/.minibot/artifacts/`
- 长期记忆：`~/.minibot/user_memory.json`

### MCP (`mcp.json`)

查找顺序：先找当前工作目录的 `mcp.json`，找不到回退到 `minibot/mcp.json`。

几个关键规则：

- `enabled: true` 的 server 启动时立刻连接并做 tool discovery
- 单个 server 失败只告警跳过，不阻止整体启动
- 工具名统一注册为 `mcp__<server>__<tool>`
- `trusted: true` 免审批；否则走正常审批流
- `transport.headers` / `transport.env` 支持 `${ENV_VAR}` 占位符
- `transport.cwd` 的相对路径按 `mcp.json` 所在目录解析

支持两种 transport：

- `stdio`：MiniBot 启动本地子进程，适合本机能力（SQLite、macOS 等）
- `streamable_http`：连接已有的远端 server，不由 MiniBot 启动

默认 bundled servers：

- `sqlite` → `mcp_servers/sqlite_server.py`（默认库 `examples/mcp/demo.sqlite3`，可用 `SQLITE_PATH` 覆盖）
- `macos_system` → `mcp_servers/macos_system/server.py`（Calendar / Reminders / Notes）

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
