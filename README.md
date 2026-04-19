# MiniBot

本地命令行 AI agent，基于 OpenAI-compatible `chat.completions`，支持：

- tool calling
- 会话持久化
- 自动 compact
- 用户长期记忆
- skills 按需读取
- macOS Calendar / Reminders / Notes 工具

## 快速开始

在仓库根目录创建 `.env`：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=
MINIBOT_MODEL=gpt-5.4-mini
MINIBOT_MAX_PARALLEL_TOOLS=4
```

安装依赖并启动：

```bash
pip install -r requirements.txt
python -m minibot
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

## 关键目录

```text
minibot/
├── __main__.py            # 组装依赖，启动 REPL
├── cli.py                 # REPL 与斜杠命令
├── config.py              # 集中配置
├── llm.py                 # LLM 适配器 (OpenAI-compatible)
├── prompts.py             # 系统提示词
├── ui.py                  # 终端样式与交互原语
├── artifacts.py           # ArtifactStore / ArtifactRef / ArtifactPage
├── user_memory.py         # 用户长期记忆
├── run_log.py             # 运行日志
├── runtime/               # TurnEngine / AgentRunner / ContextManager / Materializer
├── session/               # SessionManager + 消息模型
├── tools/                 # Tool 基类、Registry、各工具实现
├── skills/                # Markdown 技能文档
├── macos/                 # AppleScript 桥接
└── tests/
```

## 运行方式

- `TurnEngine` 负责单轮编排和持久化
- `ContextManager` 负责 system prompt、历史、memory 和 compact
- `AgentRunner` 负责 tool-calling 循环
- `AgentRunner` 支持对单轮内连续的安全只读工具做批并发执行
- `ToolOutputMaterializer` 决定工具产出内联还是落盘为 artifact
- `ToolRegistry` 统一管理工具定义和执行，支持按 `kernel`/`extension` 分层
- 会话消息保存在 `.minibot/sessions/<session_id>/messages.jsonl`
- Artifact 保存在 `.minibot/sessions/<session_id>/artifacts/`
- run log 保存在 `.minibot/runs.jsonl`

## 架构图

### 分层视图

```text
┌─────────────────────────────────────────────────────────────────────┐
│  入口 & UI 层                                                        │
│  ┌──────────────┐     ┌────────────┐                                │
│  │ __main__.py  │────▶│   cli.py   │◀────┐  ui.py (终端样式)        │
│  │ (组装依赖)    │     │  (REPL)    │     │  prompts.py              │
│  └──────────────┘     └─────┬──────┘     │                          │
└─────────────────────────────┼────────────┼──────────────────────────┘
                              │            │ event_handler
                              ▼            │ approval_handler
┌─────────────────────────────────────────────────────────────────────┐
│  运行时编排层 (runtime/)                                             │
│                                                                      │
│   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐    │
│   │ TurnEngine   │──▶│ ContextManager  │──▶│  AgentRunner     │    │
│   │ (单轮协调)    │   │ (上下文/Compact) │   │  (Tool 循环)      │    │
│   └──────────────┘   └─────────────────┘   └────────┬─────────┘    │
│                                                      │               │
│                                                      ▼  materialize()│
│                     ┌───────────────────────┐                        │
│                     │ ToolOutputMaterializer│  阈值决策中枢          │
│                     │  content ≤ 3K → 内联  │                        │
│                     │  content > 3K → 落盘  │                        │
│                     └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌──────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
│  LLM 适配层       │  │  能力层 (tools/)    │  │  持久化层             │
│  ┌────────────┐  │  │  ┌──────────────┐  │  │  ┌───────────────┐   │
│  │ LLMClient  │  │  │  │ ToolRegistry │  │  │  │SessionManager │   │
│  │(抽象接口)   │  │  │  │ (layer 过滤) │  │  │  │(消息 JSONL)   │   │
│  └─────┬──────┘  │  │  └──────┬───────┘  │  │  └───────────────┘   │
│        ▼          │  │         ▼          │  │  ┌───────────────┐   │
│  ┌────────────┐  │  │  ┌──────────────┐  │  │  │ ArtifactStore │◀──┤
│  │OpenAIClient│  │  │  │    Tool      │  │  │  │(独立存储)      │   │
│  └────────────┘  │  │  │ kernel/ext   │  │  │  └───────────────┘   │
└──────────────────┘  │  └──────┬───────┘  │  │  ┌───────────────┐   │
                      │         ▼          │  │  │UserMemoryStore│   │
                      │   返回 ToolOutput  │  │  │ (全局 JSON)    │   │
                      │                    │  │  └───────────────┘   │
                      └────────────────────┘  │  ┌───────────────┐   │
                                              │  │  RunLogStore  │   │
                                              │  └───────────────┘   │
                                              └──────────────────────┘
```

### 工具两阶段数据流

```text
        ┌────────┐                                ┌──────────┐
        │  Tool  │                                │   LLM    │
        └───┬────┘                                └────▲─────┘
            │                                          │
   execute()│ 返回                         tool_msg    │
            ▼                                          │
     ┌────────────┐                            ┌───────────────┐
     │ ToolOutput │                            │  ToolResult   │
     │ ─ content  │  ─────materialize()────▶   │ ─ artifact    │
     │ ─ data     │                            │ ─ data        │
     │ ─ summary  │                            │ ─ summary     │
     └────────────┘                            └───────────────┘
        工具语义                                  模型可见形态
       (无存储知识)                               (统一决策过)
                       │
                       ▼
               ArtifactStore.put_text()
               (大内容落盘，返回 ArtifactRef)
```

工具只负责产出 `ToolOutput`（含原始 `content`），不关心存储策略；`ToolOutputMaterializer` 统一决定内联还是落盘，模型拿到的是规范化的 `ToolResult`。

### 工具集合

```text
filesystem_toolset(workspace, artifact_store)
   ├─ ReadFileTool      [kernel]
   ├─ WriteFileTool     [kernel]
   ├─ EditFileTool      [kernel]
   ├─ ListDirTool       [kernel]
   ├─ SearchFilesTool   [kernel]
   └─ ReadArtifactTool  [kernel]   持有 ArtifactStore

shell_toolset(workspace)
   └─ ExecTool          [kernel]

network_toolset()
   ├─ WebSearchTool     [extension]
   └─ FetchUrlTool      [extension]

macos_toolset()     → Calendar / Reminders / Notes  [extension]
memory_toolset()    → RememberTool / ForgetTool     [kernel]
skill_toolset()     → ReadSkillTool (按需加载 skills/*.md)  [kernel]
```

## 配置

可用环境变量：

```bash
MINIBOT_MODEL=gpt-5.4-mini
MINIBOT_MAX_ITERATIONS=20
MINIBOT_MAX_PARALLEL_TOOLS=4
MINIBOT_MAX_HISTORY_TURNS=40
MINIBOT_COMPACT_TOKEN_THRESHOLD=40000
MINIBOT_RESERVED_COMPLETION_TOKENS=4096
MINIBOT_COMPACT_KEEP_RECENT=10
MINIBOT_AUTO_APPROVE=false
```

## 测试

运行全部测试：

```bash
python -m unittest discover -s tests
```

运行 macOS integration：

```bash
MINIBOT_RUN_MACOS_INTEGRATION=1 python -m unittest tests.test_macos_integration
```

## 说明

- `fetch_url` 更适合公开网页和普通文章页，不保证拿到纯前端站点的完整正文
- skill 采用 pull 模式：目录常驻，正文由模型通过 `read_skill` 按需读取
- 工具产出统一为 `ToolOutput`；大内容由 `ToolOutputMaterializer` 落盘为 artifact，而不是让工具自己管存储
