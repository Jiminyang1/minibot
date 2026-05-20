# MiniBot

本地命令行 AI agent，基于 OpenAI-compatible `chat.completions`。同步 turn loop + tool calling，把本地工具、MCP server、skills、长期记忆统一接在一起。

主要能力：

- tool calling（本地工具 + MCP 工具统一 schema）
- 会话持久化 + 超阈值自动 compact
- 结构化 agent event stream + 本地 SSE/Web UI
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
  - `macos_system/`：Calendar / Reminders / Notes / Mail（AppleScript 桥）

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

### 可观测性（`RuntimeEvent` + SSE + `run_log.py`）

MiniBot 把"一次 turn 发生了什么"分成**运行期事件流**和**事后落盘记录**两类，职责分开。

#### 运行期事件流

核心 runtime 现在 emit 结构化 `RuntimeEvent`，而不是中文字符串日志：

```python
RuntimeEvent(
    id: str,
    run_id: str,
    session_id: str,
    seq: int,
    type: str,
    created_at: str,
    payload: dict,
)
```

`TurnEngine` / `AgentRunner` 通过 `RuntimeEventEmitter` 发事件；CLI 只负责把事件格式化成终端文本，Web server 则把同一批事件写入进程内 `RunEventStore`，再通过标准 SSE 的 `id/event/data` 给订阅者。第一版不做 token 级 streaming，只在最终回答完成后发 `message.completed`。

Web 侧把"创建 run"和"订阅事件"分开：`POST /runs` 立即返回 `run_id`，前端再用 `EventSource` 连接 `GET /runs/{run_id}/events`。SSE 断开只会取消订阅，不会取消后台 run；取消 run 必须显式调用 `POST /runs/{run_id}/cancel`。`/events` 支持浏览器自动发送的 `Last-Event-ID`，可以从已收到的 `seq` 之后 replay。

当前事件类型：

- **Turn 级**：开始处理 / 每轮 LLM 请求与最终回答耗时 / 达到最大迭代 / compact 触发与结果
- **Tool 级**：每次 tool 调用的参数预览、返回摘要；MCP 调用会额外标注成 `mcp__<server>__<tool>`
- **审批级**：`approval.required` 暂停等待，`approval.resolved` 后继续或失败
- **消息级**：`message.completed` 只代表最终 assistant answer，不代表 tool 调用记录

Web UI 里左侧 `events` 展示模型请求、tool 调用、审批和错误；右侧 `conversation` 只展示用户输入和最终 assistant answer。会话历史仍完整落盘，包含 tool call / tool result，但这些不再伪装成聊天气泡。

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

项目使用 `uv` 管理独立 Python 环境，并通过 `.python-version` 固定解释器版本。所有运行、测试、依赖安装命令都应从 `minibot/` 目录用 `uv` 执行，不直接使用系统 `python` / `pip`。

在 `minibot/.env` 写入基础配置：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=
MINIBOT_MODEL=gpt-5.4-mini
```

安装依赖并启动：

```bash
cd /Users/jimin/Desktop/minibot
uv sync
uv run minibot
```

启动本地 SSE server：

```bash
cd /Users/jimin/Desktop/minibot
uv run minibot-server --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765/` 可以使用本地 Web UI。布局是固定视口高度：左侧 events 独立滚动，右侧 conversation 独立滚动，底部输入区不会被历史消息撑走。

第一版 Web API 暴露 MiniBot 自己的 agent event stream，不伪装成 OpenAI API：

- `POST /runs`：body 为 `{"input":"...", "session_id":"current"}`，返回 `{"run_id":"...","status":"running"}`
- `GET /runs/{run_id}/events`：返回 `text/event-stream`，支持 `Last-Event-ID`
- `POST /runs/{run_id}/cancel`：显式取消后台 run
- `POST /runs/{run_id}/approvals/{approval_id}`：body 为 `{"approved": true}`，用于继续需要审批的工具调用
- `GET /sessions` / `GET /sessions/{session_id}/messages`：供 Web UI 读取会话列表和最终对话历史

兼容入口 `POST /runs/stream` 仍保留，但新 UI 不再使用它。

## 配置

### Agent 侧（`.env` / 环境变量）

对应 `config.py:Config.from_env`：

| 变量 | 默认 | 作用 |
|---|---|---|
| `OPENAI_API_KEY` | — | 必填 |
| `OPENAI_BASE_URL` | 官方 | 指向 proxy / self-hosted endpoint |
| `MINIBOT_MODEL` | `gpt-5.4-mini` | 模型名 |
| `MINIBOT_APPROVAL_MODE` | `ask` | 工具审批模式：`ask` 表示敏感工具需要确认，`always` 表示自动批准 |
| `MINIBOT_MAX_ITERATIONS` | `20` | 单轮 turn 内 LLM ↔ tool 最大循环次数 |
| `MINIBOT_MAX_PARALLEL_TOOLS` | `4` | 同一响应多 tool call 并发上限 |
| `MINIBOT_AUTO_APPROVE` | — | 旧变量；未设置 `MINIBOT_APPROVAL_MODE` 时，`true` 会映射成 `always` |
| `MINIBOT_MAX_HISTORY_TURNS` | `40` | compact 前允许的最大历史 turn |
| `MINIBOT_COMPACT_TOKEN_THRESHOLD` | `40000` | 触发自动 compact 的 token 阈值 |
| `MINIBOT_RESERVED_COMPLETION_TOKENS` | `4096` | 留给 completion 的 token 预算 |
| `MINIBOT_COMPACT_KEEP_RECENT` | `10` | compact 时保留的最近消息数 |
| `MINIBOT_INCLUDE_REASONING_CONTENT` | `auto` | `auto` 时 DeepSeek endpoint/model 会把 `reasoning_content` 回传给模型；OpenAI 默认剥离。可设 `true` / `false` 强制覆盖 |

`MINIBOT_APPROVAL_MODE` 只是启动默认值；CLI 启动后可用 `/permission ask` 或 `/permission always` 切换当前进程的审批模式，不会自动写回 `.env`。

持久化路径（按约定生成，无需配置）：

- 会话消息：`<workspace>/.minibot/sessions/<session_id>/messages.jsonl`
- 运行摘要：`<workspace>/.minibot/runs.jsonl`
- Artifacts：`<workspace>/.minibot/artifacts/`
- 长期记忆：`~/.minibot/user_memory.json`

### MCP (`mcp.json`)

MCP 配置是 MiniBot 全局配置，不随当前项目目录变化。查找顺序：

1. `MINIBOT_MCP_CONFIG_PATH` 指定的 `mcp.json` 文件或目录
2. `~/.minibot/mcp.json`
3. 包内默认配置 `minibot/mcp.json`

几个关键规则：

- `enabled: true` 的 server 启动时立刻连接并做 tool discovery
- 单个 server 失败只告警跳过，不阻止整体启动
- 工具名统一注册为 `mcp__<server>__<tool>`
- `trusted: true` 免审批；否则走正常审批流
- `transport.headers` / `transport.env` 支持 `${ENV_VAR}` 占位符
- `transport.cwd` 的相对路径按 `mcp.json` 所在目录解析
- 内置 Python MCP server 可使用 `${MINIBOT_PYTHON}` 和 `${MINIBOT_PACKAGE_DIR}`，分别指向当前 MiniBot 解释器和安装目录

支持两种 transport：

- `stdio`：MiniBot 启动本地子进程，适合本机能力（SQLite、macOS 等）
- `streamable_http`：连接已有的远端 server，不由 MiniBot 启动

默认配置包含这些 MCP servers：

- `sqlite` → `mcp_servers/sqlite_server.py`（默认库 `examples/mcp/demo.sqlite3`，可用 `SQLITE_PATH` 覆盖）
- `macos_system` → `mcp_servers/macos_system/server.py`（Calendar / Reminders / Notes / Mail）
- `drawio` → `npx -y @drawio/mcp`（官方 draw.io MCP tool server，暴露 `open_drawio_xml/csv/mermaid`；首次启动需要本机有 Node.js / npx，且可能联网拉取 npm 包）

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
- `/permission [ask|always]` 查看或切换当前进程的工具审批模式
- `/config` 查看当前运行配置和审批模式
- `/help` 显示帮助

## 测试

在 `minibot/` 目录运行：

```bash
uv run python -m unittest discover -s tests
```

只跑 MCP 相关：

```bash
uv run python -m unittest tests.test_mcp_config tests.test_mcp_client tests.test_mcp_manager tests.test_mcp_tools
```
