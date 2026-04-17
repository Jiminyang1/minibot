# MiniBot

一个本地命令行 AI agent。核心能力是：

- tool calling
- 会话持久化
- 长输出 artifact 化
- 自动 compaction
- 全局用户长期记忆
- 基于 skill 的 workflow guidance
- macOS Calendar / Reminders / Notes 联动

## 快速开始

在仓库根目录创建 `.env`：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=
MINIBOT_MODEL=gpt-5.4-mini
```

运行：

```bash
python -m minibot
```

## CLI

常用命令：

- `/sessions`：查看会话列表
- `/new`：新建会话
- `/resume <id>`：恢复会话
- `/delete <id|current>`：删除会话
- `/compact`：手动压缩当前会话
- `/memory`：查看长期记忆
- `/memory clear`：清空长期记忆
- `/memory forget <id>`：删除单条长期记忆
- `/skills`：查看当前可用 skills
- `/help`：显示帮助

## 当前目录结构

```text
minibot/
├── __main__.py
├── cli.py
├── ui.py
├── config.py
├── llm.py
├── prompts.py
├── artifacts.py
├── user_memory.py
├── runtime/
│   ├── __init__.py
│   ├── agent_runner.py
│   ├── context_manager.py
│   └── turn_engine.py
├── tools/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── result.py
│   ├── exec_cmd.py
│   ├── read_file.py
│   ├── read_artifact.py
│   ├── write_file.py
│   ├── edit_file.py
│   ├── list_dir.py
│   ├── search_files.py
│   ├── web_search.py
│   ├── fetch_url.py
│   ├── memory_tools.py
│   └── macos_apps.py
├── macos/
│   ├── __init__.py
│   └── bridge.py
├── session/
│   ├── __init__.py
│   ├── models.py
│   └── store.py
├── skills/
│   ├── __init__.py
│   ├── registry.py
│   ├── calendar.md
│   ├── reminders.md
│   └── notes.md
└── tests/
```

## 架构

### 分层

```text
入口
- __main__.py
- cli.py / ui.py

运行时编排
- runtime/turn_engine.py
- runtime/context_manager.py
- runtime/agent_runner.py

能力层
- tools/
- macos/
- llm.py

状态层
- session/
- user_memory.py
- skills/
- prompts.py
```

### 运行流

```mermaid
flowchart TB
    User([User]) --> CLI["cli.run_repl"]
    CLI --> Engine["TurnEngine"]
    Engine --> Context["ContextManager"]
    Engine --> Runner["AgentRunner"]
    Engine --> Session["SessionManager"]
    Context --> Memory["UserMemoryStore"]
    Context --> Skills["SkillRegistry"]
    Context --> Tools["ToolRegistry"]
    Runner --> Tools
    Runner --> LLM["LLMClient"]
    Tools --> Session
    Tools --> Mac["AppleScriptBridge"]
```

### 一轮对话

```mermaid
sequenceDiagram
    actor User as 用户
    participant CLI as cli
    participant Engine as TurnEngine
    participant Context as ContextManager
    participant Runner as AgentRunner
    participant LLM as LLMClient
    participant Tools as ToolRegistry
    participant Session as SessionManager

    User->>CLI: 输入消息
    CLI->>Engine: handle_turn(session, input)
    Engine->>Context: prepare_for_turn(...)
    Context-->>Engine: PreparedContext
    Engine->>Session: append user message
    Engine->>Runner: run(RunSpec)

    loop tool-calling loop
        Runner->>LLM: chat(messages, tool_definitions)
        LLM-->>Runner: assistant / tool_calls
        opt 有 tool_calls
            Runner->>Tools: execute(name, args)
            Tools-->>Runner: ToolResult
        end
    end

    Runner-->>Engine: final reply + events
    Engine->>Session: append assistant/tool events
    Engine-->>CLI: TurnResult
```

## Prompt 组装

`ContextManager` 每轮都会重新拼一次请求。当前顺序是：

1. base system prompt
2. memory instructions
3. user memory block
4. available skills metadata
5. matched skills guidance
6. visible history
7. current user input
8. tool definitions

这里要分清：

- `tools` 是 function calling 能力，走 `tool_definitions`
- `skills` 是 prompt guidance，不可调用

当前 skill 策略：

- 所有当前可用 skill 的 metadata 都会常驻注入
- 当前输入命中的 skill 最多额外展开 2 个
- top skill 可能注入 full body
- secondary skill 只注入 summary

## Tool 体系

当前工具按 toolset 装配：

- `filesystem_toolset`
  - `read_file`
  - `read_artifact`
  - `write_file`
  - `edit_file`
  - `list_dir`
  - `search_files`
- `shell_toolset`
  - `exec`
- `network_toolset`
  - `web_search`
  - `fetch_url`
- `memory_toolset`
  - `remember`
  - `forget`
- `macos_toolset`
  - `calendar_list_events`
  - `calendar_create_event`
  - `reminders_list`
  - `reminders_create`
  - `reminders_complete`
  - `notes_search`
  - `notes_create`
  - `notes_append`

### ToolResult

所有工具都统一返回 `ToolResult`：

```json
{
  "ok": true,
  "code": "success",
  "summary": "已读取 foo.py（8241 字符，已截断预览）。",
  "data": {
    "path": "foo.py",
    "preview": "...",
    "total_chars": 8241
  },
  "artifact": {
    "id": "a_123abc456def",
    "kind": "file",
    "name": "foo.py"
  },
  "truncated": true
}
```

约束：

- `data` 只放小而稳定的结构化内容
- 大内容走 artifact
- 模型只能看到 `ok/code/summary/data/artifact/truncated`
- `meta` 只用于本地调试，不进 prompt

## 持久化

### Session

每个会话落在：

```text
.minibot/sessions/<session_id>/
├── meta.json
├── messages.jsonl
└── artifacts/
    └── <artifact_id>.json
```

说明：

- `meta.json`：标题、时间、message_count
- `messages.jsonl`：user / assistant / tool 事件日志
- `artifacts/`：长文件、长输出、长网页内容

当前会话指针在：

```text
.minibot/current_session
```

### Long-term Memory

全局长期记忆在：

```text
~/.minibot/user_memory.json
```

这里只存稳定事实，不存临时任务进度。

## macOS 集成

`macos_toolset()` 只会在下面条件满足时注册：

- `sys.platform == "darwin"`
- 系统存在 `osascript`

底层统一走 [macos/bridge.py](/Users/jiminyang/Desktop/ai-projects/agent/minibot/macos/bridge.py) 里的 `AppleScriptBridge`，不暴露通用 `run_applescript` tool。

当前范围只做：

- Calendar
- Reminders
- Notes

暂不做：

- Clock / Alarm
- 通用 app 控制
- UI automation fallback

所有本地写操作都要求 approval。

## 配置

环境变量：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=
MINIBOT_MODEL=gpt-5.4-mini
MINIBOT_MAX_ITERATIONS=20
MINIBOT_MAX_HISTORY_TURNS=40
MINIBOT_COMPACT_TOKEN_THRESHOLD=40000
MINIBOT_RESERVED_COMPLETION_TOKENS=4096
MINIBOT_COMPACT_KEEP_RECENT=10
MINIBOT_AUTO_APPROVE=false
```

默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `gpt-5.4-mini` | 默认模型 |
| `max_iterations` | `20` | 单轮 tool-calling 最大迭代次数 |
| `max_history_turns` | `40` | 带入模型的最大历史轮次 |
| `compact_token_threshold` | `40000` | 请求 token 预算上限 |
| `reserved_completion_tokens` | `4096` | 给模型输出预留的预算 |
| `compact_keep_recent` | `10` | 压缩时保留的最近轮次 |
| `auto_approve` | `false` | 是否自动批准敏感工具 |

## 测试

跑全部测试：

```bash
python -m unittest discover -s tests
```

跑 macOS integration：

```bash
MINIBOT_RUN_MACOS_INTEGRATION=1 python -m unittest tests.test_macos_integration
```

默认会跳过这些集成测试，避免直接改你的日历、提醒和笔记。

## 已知边界

- `fetch_url` 更适合静态网页、普通文章页和 SSR 页面；纯前端重渲染页面不保证能拿到完整正文
- 当前 skill matching 仍然是轻量规则式，不是 embedding retrieval
- skill metadata 会常驻注入，但详细 workflow 只会给命中的 skill
- artifact 是 session-scoped，不是全局 blob store
