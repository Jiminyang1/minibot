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
│             AgentLoop                   │  对话编排：历史窗口、压缩、持久化
│             (loop.py)                   │
├─────────────────────────────────────────┤
│           ★ Agent Core ★               │  核心：身份 + 工具 + 执行循环
│             (agent.py)                  │
│                                         │
│  system_prompt  ─ 我是谁                │
│  tools          ─ 我能做什么            │
│  tool_executor  ─ 怎么执行工具          │
│  max_iterations ─ 安全上限              │
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
| `agent.py` | **Agent Core** — 拥有 system prompt、工具集、tool-calling 循环 |
| `llm.py` | LLM 抽象层 — `LLMClient` 接口 + `OpenAIClient` 实现 |
| `loop.py` | 对话编排 — 管理 session 状态、历史窗口、压缩触发 |
| `cli.py` | REPL 交互 — 斜杠命令（`/sessions`, `/resume`, `/compact`） |
| `compaction.py` | 基于 token 的会话压缩 — tiktoken 估算，超阈值时 LLM 摘要 |
| `config.py` | 配置中心 — `.env` 解析 + `Config` dataclass |
| `prompts.py` | 系统提示词（对话 + 摘要） |
| `session/models.py` | `MessageEvent` + `Session` 领域模型 |
| `session/store.py` | `SessionManager` — JSONL 文件持久化 |
| `tools/` | 工具注册表 + 实现（`exec` 命令执行、`read_file` 文件读取） |

## 数据流

```
用户输入
  → cli.run_repl()
    → loop.handle_turn(session, input)
      1. compaction — token 超阈值? → LLM 摘要旧历史
      2. session.history_for_model() — 取最近 N 轮
      3. agent.run(history, input)
         ├── 拼装 [system_prompt, ...history, user_input]
         └── tool-calling loop (最多 max_iterations 轮)
              ├── llm.chat(messages, tools)
              ├── tool_executor(name, args)
              └── 无 tool_calls → 返回最终回答
      4. session 持久化
    → TurnResult(reply, did_compact)
  → 打印回复
```

## 设计决策

**Agent 是唯一核心** — 不是 LLM wrapper，而是拥有身份（prompt）、能力（tools）和执行策略（loop + max_iterations）的完整 agent。

**LLM 可替换** — `LLMClient` 是抽象接口，实现 `chat()` 方法即可接入任何 provider（OpenAI、DeepSeek、本地模型等）。

**Token 级压缩** — 用 tiktoken 估算实际 token 量（fallback 到字符启发式），比按消息条数判断更准确。默认 40k tokens 触发压缩。

**工具安全** — `exec` 工具内置危险命令正则拦截（rm -rf、dd、fork bomb 等）。

**会话持久化** — JSONL 格式存储在 `.minibot/sessions/`，支持多会话切换和恢复。

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
| `compact_keep_recent` | 10 | 压缩时保留的最近轮次数 |
