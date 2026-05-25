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

数据流自上而下：CLI / Web UI / 未来 SDK 把用户输入交给 `AgentSession`，由它管理一次 run 的生命周期、事件 emitter、取消信号和会话锁；`SessionManager` 负责启动 / 恢复 / 新建 / 删除和落盘；`ApprovalBroker` 处理 Web 审批请求的同步等待与解析（可被取消信号中断）。`AgentSession` 调用 `TurnEngine` 编排一次 turn——`TurnEngine` **不再一次性预拼上下文**，而是向 `AgentLoop` 注入两个回调：`prepare_next_turn()` 和 `on_message()`。`AgentLoop` 跑 LLM ↔ tool 循环，**每个 iteration** 先调 `prepare_next_turn()` → `ContextWindowManager` 从 session **投影**出模型可见消息、组装 system prompt（长期记忆 + skill L1 + 工具 schema），并在预测请求超预算时做 **turn-aware 就地压缩**（压缩只追加 `compaction` entry，非破坏；重复压缩用 previous summary 更新结构化 checkpoint；最近若是单个超预算的只读工具事务块则整块从投影中省略，非只读块直接失败）。模型返回 `tool_call` 时经 `RuntimeHookManager`（审批等内部策略）后通过 `ToolRegistry` 执行；user / assistant / tool 每条消息定稿即通过 `on_message()` 由 `TurnRecorder` 追加成 append-only entry，run log 从同一批 `MessageEvent` 统计。

**控制流 / 单轮循环**（核心：循环每轮回调 `prepare_next_turn` 重投影上下文 + `on_message` 逐条落盘）：

```
   ┌──────────────┐
   │ CLI / Web UI │  user input · approvals · rendering
   └──────┬───────┘
          │ prompt
   ┌──────▼────────────┐
   │  AgentSession     │  run 生命周期 · 会话锁 · cancel · run.* 事件
   └──────┬────────────┘
          │
   ┌──────▼────────────┐  向 AgentLoop 注入两个回调:
   │   TurnEngine      │    prepare_next_turn()  /  on_message()
   │   单轮协调器        │
   └──────┬────────────┘
          │ run(spec)
   ┌──────▼─────────────────────────────────────────┐        ┌────────┐
   │                  AgentLoop                      │◄──────►│  LLM   │
   │  LLM ↔ tool 循环 · 每个 iteration 依次:           │        └────────┘
   │   ① prepare_next_turn()  → 组装 / 压缩上下文      │
   │   ② chat.completion      → 是否有 tool_call      │
   │   ③ tool_call → hooks → ToolRegistry 执行 (见下)  │
   │   ④ on_message(每条 user/assistant/tool)         │
   └──┬───────────────┬──────────────────┬───────────┘
    ① │             ④ │                ③ │
   ┌──▼───────────────────┐ │       ┌─────▼─────────────┐
   │ ContextWindowManager │ │       │ RuntimeHookManager│
   │ · 从 session 投影消息  │ │       │ approval / 未来策略 │
   │ · 组 system prompt    │ │       └───────────────────┘
   │ · 超预算→turn-aware 压缩│ │
   │   (追加 compaction)   │ │   reads: UserMemoryStore /
   │ · drop 超大只读工具块   │ │          SkillRegistry(L1) /
   │ · after_context hook  │ │          ToolRegistry(schema)
   └──────────────────────┘ │
                            ▼
              ┌──────────────────┐      ┌─────────────────────────┐
              │   TurnRecorder   │─────►│      Session Store      │
              │ 逐条 append entry │      │ messages.jsonl (append) │
              │ + run log        │      │ meta.json / runs.jsonl  │
              └──────────────────┘      └─────────────────────────┘
```

**工具扇出**（`tool_call` → 注册表 → 本地 / MCP，对模型统一为同一种 `Tool`）：

```
   ┌───────────────┐
   │  ToolRegistry │   对 LLM：本地 / MCP tool 统一接口
   └──────┬────────┘
    ┌─────┴──────────────────────────┐
┌───▼─────────┐              ┌────────▼─────────┐
│ Local Tools │              │  MCPToolProxy    │
│ fs/exec/... │              │  (mcp_host)      │
│ read_skill  │              └────────┬─────────┘
│ (→ L2 body) │                       │  stdio / streamable_http
└─────────────┘                       │
                            ┌─────────┴─────────┐
                            │                   │
                      ┌─────▼──────┐     ┌───────▼──────┐
                      │  bundled   │     │  remote MCP  │
                      │  servers   │     │  servers     │
                      │  (sqlite / │     │  (HTTP)      │
                      │   macOS)   │     │              │
                      └────────────┘     └──────────────┘
```

几个关键约束：

- 主 turn loop 是**同步**的；异步只存在于 MCP client 的后台线程，对上层透明。
- session log 是 **append-only 的唯一真相源**；压缩只追加 `compaction` 标记、不删历史，模型可见消息每轮由 `SessionContextProjector` 投影得到——压缩天然生效、可重算、崩溃不丢已产出消息。
- 自动压缩优先从 user turn 边界保留最近上下文，只有单个 turn 本身超过保留预算时才 split turn；摘要输入会截断大型 tool result，并把上一轮 summary、当前被压缩消息、split turn 前缀合成结构化 checkpoint。重复压缩时 projector 生成的 summary message 不会再作为新 conversation 重复摘要。
- 上下文管理在**循环内每个 iteration** 进行（非一次性预拼）：预算判断优先用上一次 LLM 回传的真实 `input_tokens` + 新增消息估算，仅冷启动 / 压缩时才全量估算。
- 对模型来说本地 tool 和 MCP tool 没差别，都是 `ToolRegistry` 里同一种 `Tool`；MCP 工具统一命名 `mcp__<server>__<tool>`。
- hooks 是内部策略层，不是第三方插件 API。hook context 只暴露 `run_id` / `session_id` / `workspace` / `mode` / `emitter` / `cancel_event`，不暴露 `SessionManager`、`TurnEngine`、`AgentLoop`。
- function call / tool / MCP 是三层：function call 是模型层调用格式，tool 是 MiniBot 暴露给模型的能力对象，MCP 是外部能力接入协议。

## 核心模块

### Core runtime (`runtime/`)

一次用户输入 → `AgentSession.prompt` 进入统一 run 生命周期：

- `AgentSession` 是 CLI / Web / 未来 SDK 的统一运行入口，负责 `run.started` / `run.completed` / `run.failed`、会话并发锁和取消信号。
- `SessionManager` 负责 startup / resume / new / delete / list 和文件落盘，frontend 和 slash command 直接使用它。
- `ApprovalBroker` 是 Web 审批请求的同步 rendezvous；CLI 审批通过 `[y/N]` prompt 接到 `ApprovalPolicy.handler`。
- `TurnEngine` 编排一次 turn：向 `AgentLoop` 注入 `prepare_next_turn()`（→ `ContextWindowManager.build_context`，含 flush 上一轮 compaction）和 `on_message()`（→ `TurnRecorder` 逐条落盘 + 收集 run log），自身不再持有循环内的消息列表；手动 `compact_session()` 和 `list_available_skills()` 仍然保留在这里供 slash command 使用。
- `TurnRecorder` 负责把本轮 `MessageEvent` 逐条追加到 session，统计 MCP 用量并写 `RunLogRecord`。
- `ContextWindowManager` 组装 system prompt（基础 prompt + 长期记忆 + skill L1 元数据 + 工具 schema），判断预算并编排压缩；turn-aware 切点、summary request、summary file blocks、read/modified file metadata 提取在 `runtime/compaction.py`，token 估算在 `runtime/token_budget.py`。压缩 entry 追加到 session log；重复压缩显式合并 previous summary，并在 `details` 中累计 read/modified files；如果最新只读工具事务块本身超过预算，会整块从模型投影中省略，非只读工具块直接失败。
- `RuntimeHookManager` 承载横切运行时策略；内置 `ApprovalHook` 处理非 `trusted` tool 的审批，未来 plan mode 可以作为 hook bundle 接入。当前 `mode` 默认为 `"default"`，只透传给 hook context。
- `AgentLoop` 只跑 LLM ↔ tool 的循环；一次响应里的多个 tool call 并发执行（受 `max_parallel_tools` 限制），并在模型请求/响应和工具 prepare/execute 前后驱动 hook pipeline。
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
- `messages.jsonl`：append-only session entries，包含 `message` 和 `compaction`

`SessionManager` 负责 startup / create / list / resume / delete / rename；compact 只追加 `compaction` entry，读取时由 `SessionContextProjector` 投影出模型可见消息。`compaction` entry 保存 `summary`、`first_kept_entry_id`、`tokens_before`，以及可选 `details`；当前 details 用于累计 `{read_files, modified_files}`，也会以 `<read-files>` / `<modified-files>` block 附在摘要末尾。

### Context compaction

压缩保持 MiniBot 原有 append-only 模型，不引入 session tree 或 branch summary：

- `SessionContextProjector` 只看最新 `compaction` entry，投影出一条 synthetic assistant summary message，再接上 `first_kept_entry_id` 之后的原始 message；旧 message 仍留在 `messages.jsonl`。
- `runtime/compaction.py` 是纯规则层：选择 turn-aware cut point、构造 `SummaryRequest`、跳过 projector synthetic summary、提取 compaction details、把 `<read-files>` / `<modified-files>` 附回 summary。
- `runtime/token_budget.py` 只负责 token 估算，`ContextWindowManager` 只负责预算判断、调用 summarizer、写入新的 compaction entry。
- 文件 metadata 不解析普通 tool result 文本：`read_file` / `write_file` / `edit_file` 从 tool call 参数 `path` 提取；`read_artifact` 只从匹配的 tool result JSON 提取 `data.kind == "file"` 且 `data.name` 非空的文件名。重复压缩会合并上一条 compaction details，去重排序，并让 modified 覆盖 read。
- 如果最新工具事务块本身超预算且所有工具只读，投影中会整块省略，同时 summary 保留 `[Omitted oversized read-only tool transaction]` note；包含非只读工具时直接失败，避免丢失写操作上下文。

### Memory (`user_memory.py`)

跨会话的**全局**长期记忆（不是 session 级）：

- 存储：`~/.minibot/user_memory.json`
- 结构：一组 `{id, content, created_at}`
- 每轮由 `ContextWindowManager` 塞进 system prompt 头部
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

`TurnEngine` / `AgentLoop` 通过 `RuntimeEventEmitter` 发事件；CLI 侧默认只渲染工具、审批、错误和 compact，`--verbose` 才展示模型轮次和 context usage；Web server 则把同一批事件写入进程内 `RunEventStore`，再通过标准 SSE 的 `id/event/data` 给订阅者。第一版不做 token 级 streaming，只在最终回答完成后发 `message.completed`。

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

`minibot` 只支持交互模式；不要把 prompt 作为位置参数传入。可选参数：

```bash
uv run minibot --verbose   # 显示模型轮次、context usage 和完整工具参数摘要
uv run minibot --no-color  # 关闭 ANSI 样式；也尊重 NO_COLOR
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
| `MINIBOT_COMPACT_TOKEN_THRESHOLD` | `40000` | 触发自动 compact 的 token 阈值 |
| `MINIBOT_RESERVED_COMPLETION_TOKENS` | `4096` | 留给 completion 的 token 预算 |
| `MINIBOT_COMPACT_KEEP_RECENT_TOKENS` | `16000` | compact 后保留的最近上下文 token 目标 |
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
