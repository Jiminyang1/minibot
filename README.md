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

```
┌─────────────────────────────────────────┐
│              CLI / REPL                 │  用户交互、斜杠命令
│              (cli.py)                   │
├─────────────────────────────────────────┤
│             TurnEngine                  │  一轮消息编排：日志、压缩、持久化
│             (loop.py)                   │
├─────────────────────────────────────────┤
│            AgentSpec                    │  静态定义：prompt / tools / limits
│            AgentRunner                  │  执行器：tool-calling loop
│             (agent.py)                  │
│                                         │
│  AgentSpec    ─ 我是谁 / 我能做什么      │
│  AgentRunner  ─ 如何跑模型和工具循环     │
│  ToolRegistry ─ 如何统一管理工具能力     │
├─────────────────────────────────────────┤
│         build_messages()                │  prompt 组装
│            (context.py)                 │
│                                         │
│  system_prompt + history + user_input   │
├─────────────────────────────────────────┤
│            LLM Client                   │  Provider 无关的抽象接口
│             (llm.py)                    │
│                                         │
│  LLMClient (abstract)                   │
│    └── OpenAIClient                     │
├─────────────────────────────────────────┤
│     Session / Store / Compaction        │  持久化 + Token 级压缩
│  (session/  compaction.py  config.py)   │
└─────────────────────────────────────────┘
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `__main__.py` | 入口：加载配置 → 组装依赖 → 启动 REPL |
| `agent.py` | `AgentSpec` + `AgentRunner` — 静态定义、工具注册表和执行循环 |
| `context.py` | `build_messages()` — 组装 `[system, ...history, user]` |
| `llm.py` | LLM 抽象层 — `LLMClient` 接口 + `OpenAIClient` 实现 |
| `loop.py` | `TurnEngine` — 管理 session 状态、历史窗口、压缩触发、日志和 runner 调用 |
| `cli.py` | REPL 交互 — 启动新会话，斜杠命令（`/sessions`, `/resume`, `/new`, `/delete`, `/compact`） |
| `compaction.py` | 基于 token 的会话压缩 — tiktoken 估算，超阈值时 LLM 摘要 |
| `config.py` | 配置中心 — `.env` 解析 + `Config` dataclass |
| `prompts.py` | 系统提示词（对话 + 摘要） |
| `session/models.py` | `MessageEvent` + `Session` 领域模型 |
| `session/store.py` | `SessionManager` — JSONL 会话持久化 + 当前会话指针 |
| `tools/` | `Tool` 抽象 + `ToolRegistry` + 工具实现（`exec`、`read_file`、`write_file`、`list_dir`、`search_files`） |

## 数据流

```
用户输入
  → cli.run_repl()
    → turn_engine.handle_turn(session, input)
      1. loop 日志 — 打印当前可见上下文 token 占用
      2. compaction — request-time budget 超阈值? → LLM 摘要旧历史
      3. session.history_for_model() — 取最近 N 轮
      4. build_messages(system_prompt, history, input)
         └── 构造 [system_prompt, ...history, user_input]
      5. runner.run(messages)
         └── tool-calling loop (最多 max_iterations 轮)
              ├── llm.chat(messages, tools)
              ├── tool_registry.execute(name, args)
              └── 无 tool_calls → 返回最终回答
      6. session 持久化
    → TurnResult(reply, did_compact)
  → 打印回复
```

## 设计决策

**TurnEngine 是运行时中心** — 一轮消息从头到尾的 orchestration 都由 `TurnEngine` 管理：context usage log、compact、history、runner 调用、session save/load。

**AgentSpec / AgentRunner 分离** — `AgentSpec` 只存静态定义（prompt、tool registry、iterations），`AgentRunner` 只负责 tool-calling execution loop。

**ContextBuilder 独立** — prompt 组装不是 runner 的职责，统一由 `ContextBuilder` 处理，避免执行器同时关心 history 和消息格式。

**Tool 抽象 + ToolRegistry** — 工具不是零散函数，而是统一的能力对象；`ToolRegistry` 负责暴露 function schema 和按名称调度执行。

**LLM 可替换** — `LLMClient` 是抽象接口，实现 `chat()` 方法即可接入任何 provider（OpenAI、DeepSeek、本地模型等）。

**Request-time token 压缩** — 压缩判断基于真实请求 payload：`system prompt + visible history + current user input + tools schema`，再扣除预留输出 token。

**工具安全** — `exec` 工具内置危险命令正则拦截（rm -rf、dd、fork bomb 等）。

**会话持久化** — JSONL 格式存储在 `.minibot/sessions/`，当前会话指针存储在 `.minibot/current_session`；若指针悬空会自动清理并修正。

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
