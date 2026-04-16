# MiniBot Python 学习指南

本文档基于 minibot 项目中实际出现的每一个 Python 语法和概念，按主题分类讲解。
每个知识点都标注了对应的源文件和行号，方便你对照阅读。

---

## 目录

1. [变量与基本类型](#1-变量与基本类型)
2. [字符串操作](#2-字符串操作)
3. [数据结构：list、dict、tuple、set](#3-数据结构listdicttupleset)
4. [控制流：if / for / while](#4-控制流if--for--while)
5. [函数](#5-函数)
6. [类与面向对象](#6-类与面向对象)
7. [异常处理：try / except / raise](#7-异常处理try--except--raise)
8. [模块与包（import 系统）](#8-模块与包import-系统)
9. [类型注解（Type Hints）](#9-类型注解type-hints)
10. [装饰器（Decorators）](#10-装饰器decorators)
11. [标准库速查](#11-标准库速查)
12. [设计模式与惯用法](#12-设计模式与惯用法)

---

## 1. 变量与基本类型

Python 不需要声明类型，直接赋值即可：

```python
iteration = 0              # int（整数）
model = "gpt-5.4-mini"     # str（字符串）
api_key = None              # NoneType（空值）
did_compact = True          # bool（布尔值）
elapsed_ms = 3.14           # float（浮点数）
```

### 在项目中的出现


| 位置                     | 代码                                                    | 类型         |
| ---------------------- | ----------------------------------------------------- | ---------- |
| `agent.py:73`          | `iteration = 0`                                       | int        |
| `agent.py:40`          | `self.model = model`                                  | str        |
| `session/models.py:29` | `id: str | None = None`                               | str 或 None |
| `compaction.py:24`     | `projected = session.turn_count() + max(0, incoming)` | int        |


### None 的用法

`None` 是 Python 的"空值"，相当于其他语言的 `null`：

```python
# config.py:10 — 参数可以是 Path 或 None
def load_env(package_dir: Path | None = None) -> None:
    env_path = (package_dir or Path(__file__).resolve().parent) / ".env"
```

`package_dir or Path(...)` 的意思是：如果 `package_dir` 是 `None`（falsy），就用后面的默认值。

### Truthy / Falsy

Python 中以下值视为 `False`（falsy），其余都是 `True`（truthy）：

```python
None, False, 0, 0.0, "", [], {}, set()
```

项目中大量使用这个特性：

```python
# agent.py:89 — tool_calls 为空列表或 None 时进入分支
if not msg.tool_calls:

# cli.py:113 — 空字符串视为 falsy
if not user_msg:
    continue

# agent.py:91 — msg.content 为 None 时用空字符串
return msg.content or "", events
```

---

## 2. 字符串操作

### f-string（格式化字符串）

在字符串前加 `f`，用 `{表达式}` 嵌入变量：

```python
# cli.py:29
f"当前会话: {session.session_id} | {session.title} | "
f"{session.turn_count()} 轮对话 / {len(session.messages)} 条消息"
```

`{}` 里可以放任何表达式，包括函数调用。

### 常用字符串方法

```python
# 在项目中出现的每一个字符串方法：

text.strip()           # 去掉首尾空白     — cli.py:105, config.py:19
text.split()           # 按空白拆成列表   — session/models.py:15
text.split("=", 1)     # 按 "=" 拆，最多拆 1 次 — config.py:22
text.startswith("#")   # 是否以 "#" 开头  — config.py:20
text.lower()           # 转小写           — cli.py:110
text.upper()           # 转大写           — agent.py:188
text.replace(a, b)     # 替换子串         — session/models.py:11
"\n".join(lines)       # 用换行符拼接列表  — agent.py:201, session/store.py:73
line.splitlines()      # 按行拆分         — config.py:18
```

### 字符串拼接（相邻字面量自动合并）

```python
# agent.py:17-19 — 相邻的字符串字面量会自动拼成一个
SYSTEM_PROMPT = (
    "你是一个强大的 AI 助手，可以使用工具来完成任务。"
    "如果需要执行命令或读文件，请务必调用工具。"
)
# 等价于: "你是一个强大的 AI 助手，...请务必调用工具。"
```

### 字符串切片

```python
# cli.py:77 — 从第 7 个字符之后截取
target = raw[len("/resume"):].strip()

# agent.py:101 — 取前 100 个字符
result[:100]

# session/models.py:16 — 取前 limit 个字符
compact[:limit] + "..."

# session/models.py:35 — hex 字符串取前 12 个
uuid.uuid4().hex[:12]
```

切片语法：`text[start:end]`，省略 start 从头开始，省略 end 到末尾。

---

## 3. 数据结构：list、dict、tuple、set

### list（列表）— 有序、可变

```python
# agent.py:65-69 — 创建列表，用 * 展开另一个列表
messages: list[dict[str, Any]] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *(history or []),        # * 展开：把 history 的每个元素铺进来
    {"role": "user", "content": user_input},
]

# 常用操作
messages.append(item)        # 末尾追加      — agent.py:86
sessions.append(session)     # 末尾追加      — session/store.py:82
len(self.messages)           # 获取长度      — cli.py:29
turns[-max_turns:]           # 取最后 N 个   — session/models.py:134
```

### 列表推导式（List Comprehension）

用一行代码从旧列表生成新列表：

```python
# session/models.py:136 — 对每个 message 调用 to_model_message()
return [message.to_model_message() for message in messages]

# session/models.py:196 — 嵌套推导：把二维列表"拍平"
return [message for turn in turns for message in turn]
# 等价于:
# result = []
# for turn in turns:
#     for message in turn:
#         result.append(message)

# agent.py:161-171 — 推导式里也可以嵌套复杂表达式
payload["tool_calls"] = [
    {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    for tc in message.tool_calls
]

# session/store.py:68-70 — 带 json.dumps 的推导
message_lines = [
    json.dumps({"type": "message", **message.to_dict()}, ensure_ascii=False)
    for message in session.messages
]
```

### dict（字典）— 键值对

```python
# agent.py:49 — 创建字典
client_kwargs: dict[str, Any] = {"api_key": api_key}

# 常用操作
d["key"]                # 取值（key 不存在会报错）  — agent.py:176
d.get("key")            # 取值（不存在返回 None）   — agent.py:179
d.get("key", "默认值")  # 取值（不存在返回默认值）  — session/store.py:50

# ** 展开字典
{"type": "message", **message.to_dict()}   # session/store.py:69
# 等价于把 to_dict() 返回的所有键值对铺进新字典

OpenAI(**client_kwargs)                     # agent.py:53
# 等价于 OpenAI(api_key="xxx", base_url="yyy")
```

### tuple（元组）— 有序、不可变

```python
# compaction.py:19 — 函数返回元组
def maybe_compact(...) -> tuple[bool, str]:
    return True, "已压缩当前会话: 50 -> 12 条消息"

# cli.py:56 — 解构赋值：一次接收两个返回值
did_compact, message = maybe_compact(...)

# session/models.py:170-171 — 返回元组
return preamble, turns

# exec_cmd.py:9 — 元组列表
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+..."), "rm -rf (递归强制删除)"),
    # 每个元素是一个 (Pattern, str) 的元组
]
```

### set（集合）— 无序、不重复、快速查找

```python
# cli.py:110 — 用集合做"多值匹配"
if user_msg.lower() in {"exit", "quit"}:
# 比写 user_msg == "exit" or user_msg == "quit" 更简洁
```

---

## 4. 控制流：if / for / while

### if / elif / else

```python
# session/models.py:133-134
if max_turns > 0:
    turns = turns[-max_turns:]

# cli.py:65 — 三元表达式（一行 if-else）
prefix = "手动 compact" if manual else "自动 compact"
```

### for 循环

```python
# config.py:18 — 遍历文件的每一行
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue         # 跳过当前循环，进入下一次
    key, value = line.split("=", 1)

# exec_cmd.py:30 — 遍历时同时解构
for pattern, label in _DANGEROUS_PATTERNS:
    if pattern.search(command):
        return label
```

### while 循环

```python
# agent.py:74 — 无限循环直到 return 跳出
while True:
    iteration += 1
    resp = self.client.chat.completions.create(...)
    if not msg.tool_calls:
        return msg.content or "", events    # 跳出循环

# session/store.py:26 — 带条件的 while
while self._path(resolved_id).exists():
    resolved_id = f"s_{...}_{suffix}"
    suffix += 1
```

### break 与 continue

```python
# cli.py:108 — break 跳出整个 while 循环
except (EOFError, KeyboardInterrupt):
    print("\n已退出。")
    break

# cli.py:114 — continue 跳过本次循环
if not user_msg:
    continue          # 空输入，回到 while 开头等下一次
```

### 海象运算符 `:=`（Python 3.8+）

在表达式中赋值：

```python
# agent.py:192 — 赋值的同时做 if 判断
if tool_calls := message.get("tool_calls"):
    for call in tool_calls:
        ...
# 等价于:
# tool_calls = message.get("tool_calls")
# if tool_calls:
#     ...
```

---

## 5. 函数

### 基本定义

```python
# session/models.py:10
def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
```

`-> str` 是返回值类型注解（不强制，只是提示）。

### 参数类型

```python
def maybe_compact(
    session: Session,                                  # 普通参数（位置参数）
    manager: SessionManager,                           # 普通参数
    summarizer: Callable[[list[dict[str, Any]]], str], # 普通参数（类型是函数）
    *,                                                 # * 之后的全部是「仅关键字参数」
    threshold: int,                                    # 必须写 threshold=30 调用
    keep_recent: int,                                  # 必须写 keep_recent=10 调用
    incoming: int = 1,                                 # 有默认值，可以不传
) -> tuple[bool, str]:
```

`*` 的含义：`*` 之后的参数必须用 `key=value` 的形式传入，不能按位置传。
这样做是为了可读性，防止调用时参数顺序搞混。

项目中大量使用 `*`：

- `agent.py:37` — `__init__(self, *, model, event_handler)`
- `session/models.py:24-33` — `__init__(self, *, role, content, ...)`
- `compaction.py:15` — `maybe_compact(..., *, threshold, keep_recent, incoming)`
- `session/store.py:22` — `create_session(..., *, title)`

### 默认参数

```python
# session/models.py:14
def preview(text: str, limit: int = 30) -> str:
#                       ^^^^^^^^^^^^^ 默认值是 30，调用时可以不传
```

### **kwargs（关键字参数展开）

```python
# agent.py:53
client_kwargs = {"api_key": "xxx", "base_url": "yyy"}
self.client = OpenAI(**client_kwargs)
# 等价于: OpenAI(api_key="xxx", base_url="yyy")
```

### lambda（匿名函数）

```python
# __main__.py:24
event_handler=lambda msg: print(f"  🔧 {msg}")
# 等价于:
# def event_handler(msg):
#     print(f"  🔧 {msg}")

# session/store.py:83 — 作为排序的 key
sorted(sessions, key=lambda session: session.updated_at, reverse=True)
```

### 函数作为参数（一等公民）

Python 中函数可以像变量一样传递：

```python
# compaction.py:14 — summarizer 参数的类型是一个函数
summarizer: Callable[[list[dict[str, Any]]], str]

# cli.py:59 — 把 agent 的方法当作函数传进去
maybe_compact(session, manager, agent.summarize_history, ...)
```

---

## 6. 类与面向对象

### 基本类定义

```python
# session/models.py:19-41
class MessageEvent:
    """One message in a conversation."""               # 类的文档字符串

    __slots__ = ("id", "role", "content", ...)         # 后面会解释

    def __init__(self, *, role: str, content: str, ...) -> None:
        self.id = id or ("m_" + uuid.uuid4().hex[:12]) # self.xxx = 设置实例属性
        self.role = role
        self.content = content
```

- `class 类名:` 定义一个类
- `__init__` 是构造函数，创建实例时自动调用
- `self` 是实例自身的引用（类似 JS 的 `this`）
- `self.xxx = ...` 给实例绑定属性

### 创建和使用实例

```python
# session/models.py:161-163
summary_message = MessageEvent(
    role="assistant",
    content="[Summary of earlier conversation]\n" + summary_text.strip(),
)
summary_message.role       # 访问属性 -> "assistant"
summary_message.to_dict()  # 调用方法
```

### 实例方法、类方法、静态方法

```python
class Session:
    # 实例方法：第一个参数是 self，操作实例的数据
    def add_message(self, message: MessageEvent) -> None:
        self.messages.append(message)

    # 类方法：第一个参数是 cls（类本身），通常用来做"工厂方法"
    @classmethod
    def from_dict(cls, data: dict) -> "MessageEvent":
        return cls(role=data["role"], content=data["content"])
    # cls(...) 等价于 MessageEvent(...)，但子类继承时 cls 会自动变成子类

    # 静态方法：不接收 self 也不接收 cls，就是一个放在类里的普通函数
    @staticmethod
    def _flatten_turns(turns: list[list[MessageEvent]]) -> list[MessageEvent]:
        return [msg for turn in turns for msg in turn]
```

项目中的类方法使用场景：


| 位置                        | 用途                                      |
| ------------------------- | --------------------------------------- |
| `session/models.py:43-59` | `MessageEvent.create(...)` — 创建新消息的工厂方法 |
| `session/models.py:76-86` | `MessageEvent.from_dict(...)` — 从字典恢复对象 |
| `config.py:33-37`         | `Config.from_env()` — 从环境变量构建配置         |


### `__slots__`

```python
# session/models.py:22
__slots__ = ("id", "role", "content", "created_at", "tool_calls", "tool_call_id", "name")
```

默认情况下 Python 用 `__dict__` 字典存储实例属性，比较浪费内存。
`__slots__` 告诉 Python：这个类只有这些属性，不需要 `__dict__`。

好处：省内存、防止拼错属性名（`msg.roel = "user"` 会直接报错）。

### `self` 的理解

```python
class Session:
    def turn_count(self) -> int:
        _, turns = self._split_preamble_and_turns()
        return len(turns)

# 当你调用:
current_session.turn_count()
# Python 实际执行的是:
Session.turn_count(current_session)
# self 就是 current_session
```

---

## 7. 异常处理：try / except / raise

### 基本用法

```python
# __main__.py:21-28
try:
    agent = MiniBotAgent(model=config.model, ...)
except RuntimeError as exc:        # 只捕获 RuntimeError 类型
    print(f"配置错误: {exc}")       # exc 是异常对象
    return
```

### 捕获多种异常

```python
# cli.py:106 — 捕获两种异常
except (EOFError, KeyboardInterrupt):
    print("\n已退出。")
    break
```

### 捕获所有异常

```python
# cli.py:147 — Exception 是几乎所有异常的基类
except Exception as exc:
    print(f"\n❌ 运行失败: {exc}")
    continue
```

### 主动抛出异常

```python
# agent.py:45-47
if not api_key:
    raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 或环境变量里设置。")

# agent.py:115
raise ValueError("没有可供摘要的历史消息。")
```

`raise` = 抛出异常，程序会中断并寻找最近的 `except` 来捕获。

### try 嵌套在项目中的完整流程

```python
# cli.py:132-149 — 用户发送消息时的完整异常保护
try:
    _handle_compact(...)                     # 可能失败（LLM 调用）
    history = current_session.history_for_model(...)
    current_session.add_message(...)
    manager.save(current_session)
    reply, turn_events = agent.run(...)      # 可能失败（LLM / 工具）
    for event in turn_events:
        current_session.add_message(event)
    manager.save(current_session)
except Exception as exc:
    print(f"\n❌ 运行失败: {exc}")
    continue                                 # 不崩溃，继续等下一次输入
```

---

## 8. 模块与包（import 系统）

这是理解项目结构最重要的部分。

### 基本 import

```python
import os                     # 导入整个模块，用 os.environ 访问
import json                   # 导入整个模块，用 json.loads() 访问
from pathlib import Path      # 从模块中导入某个名字
from openai import OpenAI     # 从第三方包中导入
```

### 相对导入（`.` 开头）

项目内部模块之间用 `.` 表示"当前包"：

```python
# agent.py:13-14 — 从同级模块导入
from .session import MessageEvent        # . = minibot 包
from .tools import TOOL_DEFINITIONS      # . = minibot 包

# session/store.py:9 — 从同一个子包内导入
from .models import MessageEvent, Session  # . = minibot.session 包
```


| 写法                             | 含义                    |
| ------------------------------ | --------------------- |
| `from .config import Config`   | 当前包的 `config.py`      |
| `from .session import Session` | 当前包的 `session/` 子包    |
| `from . import exec_cmd`       | 当前包的 `exec_cmd.py` 模块 |


### `__init__.py` — 把目录变成包

一个目录里有 `__init__.py`，Python 就把它当作一个"包"。

```python
# session/__init__.py — 决定 from .session import 能导入什么
from .models import MessageEvent, Session
from .store import SessionManager

__all__ = ["MessageEvent", "Session", "SessionManager"]
```

这样外部可以直接写 `from .session import Session`，而不需要
`from .session.models import Session`。

`__all__` 定义了 `from .session import *` 时会导出哪些名字。

### `__main__.py` — 包的入口点

当你运行 `python -m minibot` 时，Python 会执行 `minibot/__main__.py`。

### `tools/__init__.py` — 注册表模式

```python
# tools/__init__.py:8
from . import exec_cmd, read_file
# 导入同包内的两个模块，相当于:
# from minibot.tools import exec_cmd
# from minibot.tools import read_file
```

### `TYPE_CHECKING` — 只在类型检查时导入

```python
# cli.py:5,10-12
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import MiniBotAgent
    from .config import Config
```

`TYPE_CHECKING` 在运行时是 `False`，只有 IDE / mypy 做类型检查时才是 `True`。
这样做是为了避免循环导入（cli 导入 agent，agent 又导入 cli 的情况）。

### 项目的完整导入关系

```
__main__.py
├── from .config import Config, load_env
├── from .agent import MiniBotAgent
├── from .cli import run_repl
└── from .session import SessionManager

cli.py
├── from .compaction import maybe_compact
├── from .session import MessageEvent, Session, SessionManager
└── (TYPE_CHECKING) from .agent, .config

agent.py
├── from .session import MessageEvent
└── from .tools import TOOL_DEFINITIONS, execute_tool

compaction.py
└── from .session import Session, SessionManager

tools/__init__.py
├── from . import exec_cmd
└── from . import read_file

session/__init__.py
├── from .models import MessageEvent, Session
└── from .store import SessionManager
```

---

## 9. 类型注解（Type Hints）

Python 的类型注解不强制执行，但能让 IDE 提供自动补全和错误提示。

### 基本类型

```python
model: str = "gpt-5.4-mini"
max_turns: int = 40
manual: bool = False
```

### 联合类型（`|`）

```python
# session/models.py:29 — 可以是 str 或 None
id: str | None = None

# cli.py:76 — 返回 Session 或 None
def _handle_resume(...) -> Session | None:
```

`str | None` 等价于老写法 `Optional[str]`。需要 `from __future__ import annotations` 才能在低版本 Python 使用。

### 容器类型

```python
list[str]                        # 字符串列表
dict[str, Any]                   # 键是 str，值是任意类型的字典
tuple[bool, str]                 # 两个元素的元组：(bool, str)
tuple[int, int]                  # 两个元素的元组：(int, int)
list[dict[str, Any]]             # 字典列表
list[list[MessageEvent]]         # 二维列表
list[tuple[re.Pattern[str], str]]  # 元组列表
```

### `Any` — 任意类型

```python
from typing import Any

# 当你不想/没法精确描述类型时
args: dict[str, Any]     # 值可以是任何类型
```

### `Callable` — 函数类型

```python
from collections.abc import Callable

# agent.py:38 — 接收一个 str 参数、无返回值的函数
event_handler: Callable[[str], None] | None = None

# compaction.py:14 — 接收 list[dict[str, Any]] 参数、返回 str 的函数
summarizer: Callable[[list[dict[str, Any]]], str]
```

格式：`Callable[[参数类型1, 参数类型2, ...], 返回类型]`

### `from __future__ import annotations`

```python
# 出现在几乎每个文件的第一行
from __future__ import annotations
```

作用：让所有类型注解变成字符串（延迟求值），这样你可以：

- 在类型注解中使用 `str | None` 而不需要 Python 3.10+
- 在类型注解中引用还没定义的类（前向引用）

---

## 10. 装饰器（Decorators）

装饰器是放在函数/类定义上面的 `@xxx`，用来修改它的行为。

### `@dataclass`

```python
# config.py:26-31
@dataclass(frozen=True)
class Config:
    model: str = "gpt-5.4-mini"
    max_history_turns: int = 40
    compact_threshold: int = 30
    compact_keep_recent: int = 10
```

`@dataclass` 自动帮你生成 `__init__`、`__repr__`、`__eq__` 等方法。
等价于手写：

```python
class Config:
    def __init__(self, model="gpt-5.4-mini", max_history_turns=40, ...):
        self.model = model
        self.max_history_turns = max_history_turns
        ...
```

`frozen=True` 表示创建后不能修改属性（不可变对象）。

### `@classmethod`

```python
# config.py:33-37
@classmethod
def from_env(cls) -> Config:
    return cls(
        model=os.environ.get("MINIBOT_MODEL", cls.model),
    )

# 调用方式:
config = Config.from_env()    # cls 就是 Config 类本身
```

### `@staticmethod`

```python
# agent.py:136-141
@staticmethod
def _preview(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
```

`@staticmethod` 不接收 `self` 也不接收 `cls`，纯粹是个工具函数，
放在类里只是因为逻辑上属于这个类。

---

## 11. 标准库速查

### `os` — 操作系统接口

```python
import os

os.environ.get("OPENAI_API_KEY")           # 读取环境变量（不存在返回 None）
os.environ.get("KEY", "default")           # 读取环境变量（不存在返回默认值）
os.environ.setdefault("KEY", "value")      # 设置环境变量（已存在则不覆盖）
```

### `pathlib.Path` — 文件路径操作

```python
from pathlib import Path

Path(__file__)                  # 当前文件的路径
Path(__file__).resolve()        # 绝对路径
Path(__file__).resolve().parent # 所在目录
Path.cwd()                     # 当前工作目录

path / "subdir" / "file.txt"   # 用 / 拼接路径（替代 os.path.join）
path.exists()                  # 文件是否存在
path.read_text(encoding="utf-8")  # 读取文件内容为字符串
path.write_text(content, encoding="utf-8")  # 写入文件
path.mkdir(parents=True, exist_ok=True)     # 创建目录（含父目录）
path.glob("*.jsonl")           # 匹配文件名模式
path.stem                      # 文件名不含扩展名：foo.jsonl -> foo
```

### `json` — JSON 处理

```python
import json

json.loads(line)                              # 字符串 -> Python 对象
json.dumps(data, ensure_ascii=False)          # Python 对象 -> 字符串
# ensure_ascii=False 让中文直接输出，不转成 \uXXXX
```

### `subprocess` — 执行外部命令

```python
import subprocess

result = subprocess.run(
    command,
    shell=True,           # 通过 shell 解释命令（支持管道等）
    capture_output=True,  # 捕获 stdout 和 stderr
    text=True,            # 输出为字符串（而非 bytes）
)
result.stdout             # 标准输出
result.stderr             # 标准错误
```

### `re` — 正则表达式

```python
import re

pattern = re.compile(r"\brm\s+-rf\b")  # 编译正则（提高多次匹配性能）
pattern.search(command)                 # 在 command 中搜索匹配（返回 Match 或 None）
```

`r"..."` 是原始字符串，`\b` 不会被当成转义。

### `uuid` — 生成唯一 ID

```python
import uuid

uuid.uuid4().hex[:12]    # 生成随机 UUID，取前 12 个十六进制字符
# 例如: "a1b2c3d4e5f6"
```

### `datetime` — 日期时间

```python
from datetime import UTC, datetime

datetime.now(UTC)                         # 当前 UTC 时间
datetime.now(UTC).isoformat()             # ISO 格式字符串
datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化：20260416_143052
```

### `time` — 性能计时

```python
import time

started_at = time.perf_counter()       # 高精度计时器（秒）
# ... 执行操作 ...
elapsed = time.perf_counter() - started_at
elapsed_ms = int(elapsed * 1000)       # 转毫秒
```

### `dataclasses` — 数据类

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    model: str = "gpt-5.4-mini"
    # 自动生成 __init__, __repr__, __eq__
    # frozen=True → 属性创建后不可修改
```

---

## 12. 设计模式与惯用法

### `_` 下划线命名 — 私有约定

```python
_REGISTRY           # 模块级私有变量（tools/__init__.py:10）
_DANGEROUS_PATTERNS # 模块级私有变量（exec_cmd.py:9）
_check_dangerous    # 模块级私有函数（exec_cmd.py:28）
_print_help         # 模块级私有函数（cli.py:15）
self._emit          # 实例私有方法  （agent.py:132）
```

Python 没有真正的 private，`_` 前缀只是约定："这是内部实现，外部不应该直接使用"。

### 注册表模式（Registry）

```python
# tools/__init__.py — 用字典映射名字到函数
_REGISTRY: dict[str, Callable] = {
    "exec": exec_cmd.execute,
    "read_file": read_file.execute,
}

def execute_tool(name: str, args: dict) -> str:
    fn = _REGISTRY.get(name)    # 按名字查找函数
    if fn is None:
        return f"未知工具: {name}"
    return fn(args)             # 调用找到的函数
```

好处：添加新工具只需加一个文件 + 在注册表加一行。

### 回调模式（Callback）

```python
# agent.py:38 — 构造时传入一个回调函数
event_handler: Callable[[str], None] | None = None

# agent.py:132-134 — 有事件时调用回调
def _emit(self, message: str) -> None:
    if self.event_handler:
        self.event_handler(message)

# __main__.py:24 — 调用方决定"怎么处理事件"
event_handler=lambda msg: print(f"  🔧 {msg}")
```

Agent 不关心事件怎么展示，它只负责"通知"。
展示逻辑由调用方通过回调定义。

### 依赖注入

```python
# __main__.py — 在入口处创建所有依赖，传给需要它们的模块
config = Config.from_env()
manager = SessionManager(Path.cwd())
agent = MiniBotAgent(model=config.model, ...)
run_repl(agent, manager, config)         # 把依赖注入 REPL
```

`cli.py` 不自己创建 agent 或 manager，而是由外部传入。
好处：容易测试、容易替换实现。

### 工厂方法

```python
# session/models.py:43-59
@classmethod
def create(cls, *, role, content, ...) -> "MessageEvent":
    return cls(role=role, content=content, ...)

# config.py:33-37
@classmethod
def from_env(cls) -> Config:
    return cls(model=os.environ.get("MINIBOT_MODEL", cls.model))
```

不直接用 `__init__`，而是用命名清晰的类方法来创建对象。
`create` 和 `from_env` 比 `__init__` 更能表达意图。

### Docstring（文档字符串）

```python
# 三引号字符串放在函数/类/模块的第一行，就是文档
"""Run MiniBot: python -m minibot"""                    # 模块文档（__main__.py:1）

class MiniBotAgent:
    """Small agent: one tool-calling loop."""            # 类文档（agent.py:32）

    def run(self, user_input: str, ...) -> ...:
        """Run one user request until the model returns a final answer."""  # 方法文档
```

可以用 `help(MiniBotAgent)` 或 IDE 悬浮查看这些文档。

---

## 附：项目数据流完整走读

当用户输入一条消息，完整经过以下步骤：

```
1. cli.py: run_repl()
   │
   ├─ 读取用户输入: input("\nYou: ")
   │
   ├─ 检查斜杠命令 → 如果是 /help, /sessions 等 → 处理后 continue
   │
   ├─ 自动压缩检查: _handle_compact()
   │   └─ compaction.py: maybe_compact()
   │       ├─ session.turn_count() 计算轮次
   │       ├─ 如果超阈值 → 调用 summarizer() 生成摘要
   │       └─ session.compact_with_summary() 替换旧消息
   │
   ├─ 构建历史: session.history_for_model()
   │   └─ session/models.py: 按轮次切分 → 取最近 N 轮 → 转成 dict 列表
   │
   ├─ 保存用户消息: session.add_message() → manager.save()
   │
   ├─ 调用 Agent: agent.run(user_msg, history=history)
   │   └─ agent.py: MiniBotAgent.run()
   │       ├─ 组装 messages = [system_prompt, ...history, user_msg]
   │       └─ while True:
   │           ├─ 调用 LLM: client.chat.completions.create()
   │           ├─ 如果没有 tool_calls → return 最终回答
   │           └─ 如果有 tool_calls:
   │               ├─ tools/__init__.py: execute_tool(name, args)
   │               │   └─ 查注册表 → 调用对应工具函数
   │               └─ 把工具结果加入 messages → 继续循环
   │
   ├─ 保存 Agent 回复: session.add_message() → manager.save()
   │
   └─ 打印回复: print(f"\nAgent: {reply}")
```

---

> 学习建议：从 `__main__.py` 开始读，跟着函数调用跳到对应模块，
> 对照本文档查不懂的语法。每个概念都能在项目中找到真实使用场景。

