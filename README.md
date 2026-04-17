# MiniBot

一个从零构建的命令行 AI Agent，基于 OpenAI 兼容 API，具备工具调用、会话持久化和自动压缩能力。

## 快速开始

```bash
# 配置
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 运行
python -m minibot
```

## 架构

### 分层视图

自上而下：外层负责"何时做"，内层负责"怎么做"。

```
┌───────────────────  入口  ───────────────────┐
│  __main__.py                  装配与启动     │
│  cli.py  +  ui.py             REPL 分派 / 样式│
├───────────────────  编排  ───────────────────┤
│  loop.py      (TurnEngine)    一轮消息编排   │
│  compaction.py                token 预算+摘要│
├──────────────────  Agent 核心  ──────────────┤
│  agent.py     (Spec + Runner) tool-calling   │
│  context.py   build_messages  prompt 组装    │
├───────────────────  能力  ───────────────────┤
│  tools/                       工具抽象 + 实现│
│  llm.py                       LLM 抽象 + 实现│
├───────────────────  状态  ───────────────────┤
│  session/                     短期记忆       │
│  memory.py                    长期记忆       │
│  config.py  prompts.py        配置 / 提示词  │
└──────────────────────────────────────────────┘
```

### 组件依赖

实线 = 调用，虚线 = 构造期装配。

```mermaid
flowchart TB
    User([用户]) --> CliMod

    subgraph entry [入口]
        Main["__main__<br/>装配"]
        CliMod["cli<br/>REPL 分派"]
        UiMod["ui<br/>样式 / 打印"]
    end

    subgraph orch [编排]
        Engine["TurnEngine<br/>loop.py"]
        Compact["compaction<br/>摘要 + 预算"]
    end

    subgraph core [Agent 核心]
        Spec["AgentSpec<br/>静态定义"]
        Runner["AgentRunner<br/>tool 循环"]
        Ctx["build_messages<br/>context"]
    end

    subgraph caps [能力]
        Tools["ToolRegistry<br/>tools/"]
        Llm["LLMClient<br/>llm.py"]
    end

    subgraph state [状态]
        SM[("SessionManager<br/>session/")]
        MS[("MemoryStore<br/>memory")]
        Cfg["Config<br/>config"]
    end

    Main -.装配.-> Engine
    Main -.装配.-> Runner
    Main -.装配.-> CliMod

    CliMod --> Engine
    CliMod --> UiMod
    CliMod --> SM
    CliMod --> MS

    Engine --> Runner
    Engine --> Compact
    Engine --> Ctx
    Engine --> SM
    Engine --> MS

    Runner --> Spec
    Runner --> Tools
    Runner --> Llm
    Tools --> MS

    Compact --> Llm
    Compact --> SM
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `__main__.py` | 入口：加载配置 → 组装依赖 → 启动 REPL |
| `agent.py` | `AgentSpec` + `AgentRunner` — 静态定义、工具注册表和执行循环 |
| `context.py` | `build_messages()` — 组装 `[system, ...history, user]` |
| `llm.py` | LLM 抽象层 — `LLMClient` 接口 + `OpenAIClient` 实现 |
| `loop.py` | `TurnEngine` — 管理 session 状态、历史窗口、压缩触发、日志和 runner 调用 |
| `cli.py` | REPL 交互 — 斜杠命令分派（`/sessions`, `/resume`, `/new`, `/delete`, `/compact`, `/memory`），只管编排，不管视觉 |
| `ui.py` | 终端样式与打印 helper — ANSI 颜色、banner/status/help 面板、`tool_log`、`prompt_approval`，全项目唯一的输出出口 |
| `compaction.py` | 基于 token 的会话压缩 — tiktoken 估算，超阈值时 LLM 摘要 |
| `config.py` | 配置中心 — `.env` 解析 + `Config` dataclass |
| `prompts.py` | 系统提示词（对话 + 长期记忆指令 + 摘要） |
| `memory.py` | `MemoryStore` — 跨 session 的长期记忆（用户事实/项目状态），每轮注入 system prompt |
| `session/models.py` | `MessageEvent` + `Session` 领域模型 |
| `session/store.py` | `SessionManager` — JSONL 会话持久化 + 当前会话指针 |
| `tools/` | `Tool` 抽象 + `ToolRegistry` + 工具实现（`exec`、`read_file`、`write_file`、`list_dir`、`search_files`、`remember`、`forget`） |

## 一轮对话的时序

```mermaid
sequenceDiagram
    actor User as 用户
    participant CLI as cli.run_repl
    participant Engine as TurnEngine
    participant Compact as compaction
    participant Runner as AgentRunner
    participant LLM as LLMClient
    participant Tools as ToolRegistry
    participant SM as SessionManager
    participant MS as MemoryStore

    User->>CLI: 输入一行
    CLI->>Engine: handle_turn(session, input)

    Engine->>MS: render_for_prompt
    MS-->>Engine: 长期记忆 block

    Engine->>Compact: maybe_compact(session, input)
    alt 预计 token 超预算
        Compact->>LLM: summarize(旧历史)
        LLM-->>Compact: 摘要
        Compact->>SM: save(压缩后 session)
    end

    Engine->>SM: save(user message)
    Engine->>Runner: run(messages)

    loop tool-calling loop
        Runner->>LLM: chat(messages, tools)
        LLM-->>Runner: assistant + tool_calls?
        opt 有 tool_calls
            Runner->>Tools: execute(name, args)
            Tools-->>Runner: 结果 (含 remember / forget)
        end
    end

    Runner-->>Engine: reply + events
    Engine->>SM: save(events)
    Engine-->>CLI: TurnResult
    CLI->>User: 打印回复
```

要点：

- **effective_system_prompt** = 基础 prompt + 长期记忆指令 + 当前记忆，每轮现拼
- **compaction** 在 user message 入库**前**做判断，基于预估的完整请求 payload
- **tool 循环**最多 `max_iterations` 轮；无 `tool_calls` 时直接返回最终回答
- **持久化**发生两次：user 输入落盘一次、runner 结束后把 assistant/tool 事件落盘一次

## 设计决策

**TurnEngine 是运行时中心** — 一轮消息从头到尾的 orchestration 都由 `TurnEngine` 管理：context usage log、compact、history、runner 调用、session save/load。

**AgentSpec / AgentRunner 分离** — `AgentSpec` 只存静态定义（prompt、tool registry、iterations），`AgentRunner` 只负责 tool-calling execution loop。

**ContextBuilder 独立** — prompt 组装不是 runner 的职责，统一由 `ContextBuilder` 处理，避免执行器同时关心 history 和消息格式。

**Tool 抽象 + ToolRegistry** — 工具不是零散函数，而是统一的能力对象；`ToolRegistry` 负责暴露 function schema 和按名称调度执行。

**LLM 可替换** — `LLMClient` 是抽象接口，实现 `chat()` 方法即可接入任何 provider（OpenAI、DeepSeek、本地模型等）。

**Request-time token 压缩** — 压缩判断基于真实请求 payload：`system prompt + visible history + current user input + tools schema`，再扣除预留输出 token。

**工具安全** — `exec` 工具内置危险命令正则拦截（rm -rf、dd、fork bomb 等）。

**会话持久化** — JSONL 格式存储在 `.minibot/sessions/`，当前会话指针存储在 `.minibot/current_session`；若指针悬空会自动清理并修正。

**长期记忆（跨 session）** — `MemoryStore` 持久化在 `.minibot/memory.json`，每轮拼接到 system prompt 之后全量注入。Agent 通过 `remember` / `forget` 两个工具主动管理；REPL 可用 `/memory`、`/memory clear`、`/memory forget <id>` 查看和编辑。只记稳定事实（姓名、环境、偏好、项目状态），daily 细节靠 session 历史本身，不进入长期记忆。长期记忆会计入 compaction 的 token 预算。

## 配置

通过 `.env` 或环境变量：

```bash
OPENAI_API_KEY=sk-xxx          # 必填
OPENAI_BASE_URL=               # 可选，自定义 API 端点
MINIBOT_MODEL=gpt-5.4-mini     # 可选，默认模型
```

`Config` 内部参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_history_turns` | 40 | 送给模型的最大历史轮次 |
| `compact_token_threshold` | 40000 | 触发压缩的 token 阈值 |
| `reserved_completion_tokens` | 4096 | 为模型输出预留的 token 预算 |
| `compact_keep_recent` | 10 | 压缩时保留的最近轮次数 |
