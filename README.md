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
│   ├── read_skill.py
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

### 分层组件图

```mermaid
flowchart TB
    subgraph Entry["Composition Root"]
        Main["__main__.py"]
        CLI["cli.py (REPL / 斜杠命令)"]
    end

    subgraph Runtime["Runtime 层 (单轮编排)"]
        TE["TurnEngine<br/>事件广播 / 消息持久化"]
        CM["ContextManager<br/>组装 system prompt<br/>token 预算 / 压缩触发"]
        AR["AgentRunner<br/>LLM tool-calling 循环<br/>审批 / 迭代上限"]
        SUM["Summarizer<br/>压缩旧轮次"]
    end

    subgraph Capability["Capability 层 (Tools)"]
        TR["ToolRegistry"]
        FS["filesystem_toolset<br/>read/write/edit/list/search"]
        SH["shell_toolset<br/>exec"]
        NET["network_toolset<br/>fetch_url / web_search"]
        MEM["memory_toolset<br/>remember / forget"]
        MAC["macos_toolset<br/>calendar_* / reminders_* / notes_*"]
        SK["skill_toolset<br/>read_skill"]
    end

    subgraph State["State 层 (持久化)"]
        SM["SessionManager<br/>.minibot/sessions/&lt;id&gt;/"]
        UM["UserMemoryStore<br/>~/.minibot/user_memory.json"]
        SR["SkillRegistry<br/>只加载 + 按名查"]
        AS["ArtifactStore<br/>大对象落盘 + ArtifactRef"]
    end

    subgraph External["External"]
        LLM["OpenAIClient"]
        APPLE["AppleScriptBridge"]
    end

    Main --> CLI
    CLI --> TE
    TE --> CM
    TE --> AR
    CM --> UM
    CM --> SR
    CM --> TR
    CM --> SUM
    SUM --> LLM
    AR --> LLM
    AR --> TR
    TR --> FS
    TR --> SH
    TR --> NET
    TR --> MEM
    TR --> MAC
    TR --> SK
    SK -.按 name 查.-> SR
    MAC --> APPLE
    FS --> AS
    NET --> AS
    TE --> SM
    MEM --> UM
```

核心分工：

- **Runtime** 只做编排，不拥有业务逻辑
- **Capability** 所有能力都挂在 `ToolRegistry` 下，按 toolset 装配
- **State** 三种存储各司其职：会话 / 长期记忆 / skill 目录
- `SkillRegistry` 在热路径上只出现在 prompt 装配（拿 L1 目录）和 `read_skill` 工具（按 name 查）两处

### 一轮对话（含 skill pull 链路）

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant TE as TurnEngine
    participant CM as ContextManager
    participant SR as SkillRegistry
    participant AR as AgentRunner
    participant LLM as OpenAIClient
    participant TR as ToolRegistry
    participant RS as ReadSkillTool
    participant Tgt as Target Tool<br/>(例 calendar_list_events)

    U->>TE: handle_turn(user_input)
    TE->>CM: prepare_for_turn(session, input)
    CM->>SR: list() 可用 skills
    CM->>TR: get_definitions() 工具 schema
    Note over CM: 组装 system prompt:<br/>base + memory + L1 skill 目录<br/>(只有 name/desc/tools)
    CM-->>TE: PreparedContext

    TE->>AR: run(spec)
    loop LLM tool-calling loop
        AR->>LLM: chat(messages, tools)
        LLM-->>AR: tool_calls

        alt 模型决定先 pull skill (可选)
            AR->>TR: dispatch("read_skill", {name})
            TR->>RS: execute
            RS->>SR: get_by_name(name)
            SR-->>RS: Skill(body, ...)
            RS-->>AR: ToolResult(data.body = L2 正文)
            Note over AR,LLM: body 进入下一轮<br/>tool message
        end

        AR->>TR: dispatch(target_tool, args)
        TR->>Tgt: execute
        Tgt-->>AR: ToolResult
    end
    AR-->>TE: 最终回答
    TE->>TE: 持久化 user + assistant 消息
    TE-->>U: 回答
```

## Prompt 组装

`ContextManager` 每轮重新拼一次请求。结构如下：

```mermaid
flowchart LR
    A["base SYSTEM_PROMPT"] --> P
    B["MEMORY_INSTRUCTIONS"] --> P
    C["## User Memory Data<br/>(若非空)"] --> P
    D["## Available Skills<br/>L1 目录 + 调用 read_skill 指令"] --> P
    P["System message"] --> M
    H["history (按 max_history_turns 截)"] --> M
    U["当前 user input"] --> M
    T["tool_definitions[]<br/>(含 read_skill + 业务工具)"]

    M["messages[]"] --> REQ["LLM request"]
    T --> REQ
```

分工：

- `tools` 是 function calling 能力，走 `tool_definitions`
- `skills` 是 workflow guidance，走 `tool_definitions` 里的 `read_skill` **按需拉取**

当前 skill 策略（progressive disclosure / model-pulled）：

- **L1 metadata**（name / description / tools）每轮常驻在系统提示的 `## Available Skills` 块
- **L2 body** 不由框架匹配注入，只在模型调用 `read_skill` 时进 tool message
- 是否加载、加载哪个、加载几个，完全由模型自行判断
- 框架不再做 trigger 匹配、不再有 top-2 / full / summary 三档模式
- L2 内容走 tool message 通道进上下文，模型清楚那是"刚加载的参考资料"而不是"新系统规则"

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
- `skill_toolset`
  - `read_skill`

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
- `messages.jsonl`：transcript，只存 user / assistant / tool 历史，供后续拼上下文
- `artifacts/`：长文件、长输出、长网页内容

当前会话指针在：

```text
.minibot/current_session
```

### Run Log

每轮用户输入还会额外追加一条运行摘要日志：

```text
.minibot/runs.jsonl
```

说明：

- `runs.jsonl`：turn 级观测摘要日志，记录耗时、工具数、工具名、compact 情况、错误摘要等
- run log 不进入模型上下文，也不替代 session transcript

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
- skill 采用 pull 模式：L1 目录常驻、L2 正文按需 `read_skill`；框架不再做 trigger 匹配，模型不读就不加载
- 若 skill 正文没有 L1 目录无法传达的关键信息（默认值、跨工具编排、错误恢复等），模型多半不会去 pull——这是预期行为，不是 bug
- artifact 是 session-scoped，不是全局 blob store
