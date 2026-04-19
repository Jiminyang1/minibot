# MiniBot Tool Call 设计教学

这份文档讲 4 件事：

1. 怎么设计一个可维护的 tool call 契约
2. 怎么统一工具输入/输出格式
3. 怎么在不把全项目 async 化的前提下做单轮多工具并发
4. 这套并发模型当前不解决什么问题

本文对应的关键实现：

- `minibot/llm.py`
- `minibot/tools/base.py`
- `minibot/tools/registry.py`
- `minibot/tools/result.py`
- `minibot/runtime/tool_output_materializer.py`
- `minibot/runtime/agent_runner.py`

## 1. 先把边界分清

MiniBot 的 tool calling 分成 4 层：

1. `LLMClient`
  - 负责向模型请求结果
  - 返回标准化后的 `LLMResponse`
2. `Tool`
  - 每个工具只负责自己的业务逻辑
  - 不负责 artifact、并发调度、消息写回
3. `ToolRegistry`
  - 负责工具查找、参数校验、执行封装
  - 统一把异常转换成 `ToolOutput.failure(...)`
4. `AgentRunner`
  - 负责整个 tool-calling 循环
  - 包括审批、批次划分、并发执行、结果顺序回写

这个分层很重要。不要把调度策略写进工具本身，也不要把工具输出格式分散到各处。

## 2. Tool call 的统一输入格式

模型侧统一看见的是：

```python
ToolCall(
    id="call_1",
    name="read_file",
    arguments='{"path":"README.md"}',
)
```

这里有两个关键点：

- `name` 是稳定标识，必须和 `Tool.name` 一致
- `arguments` 保留原始 JSON 字符串，由 runner 统一 `json.loads`

这样做的好处：

- provider 差异被隔离在 `LLMClient`
- tool 层永远只接收 Python 命名参数
- runner 可以统一处理坏 JSON、空参数、非法类型

结论：不要让每个工具自己 parse JSON；模型输出先归一化，再进入本地执行层。

## 3. Tool 的统一定义

每个工具都实现同一个抽象：

```python
class Tool(ABC):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    @property
    def requires_approval(self) -> bool:
        return False

    @property
    def read_only(self) -> bool:
        return False

    @property
    def exclusive(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return self.read_only and not self.exclusive

    def execute(self, *, context: ToolExecutionContext, **kwargs: Any) -> ToolOutput:
        ...
```

> Python 备忘
>
> - `ABC` + `@abstractmethod`：标记抽象基类。子类没实现抽象方法就 `TypeError: Can't instantiate abstract class`，是"接口契约"的运行时强制。
> - `@property`：把方法当属性访问（`tool.name` 而不是 `tool.name()`）。这里全用 property 的好处是：子类只要返回常量，不必重复写 `__init__`。
> - `def execute(self, *, context, **kwargs)` 里那个裸 `*`：表示后面的参数都必须用关键字传。这样模型给的 JSON `{"path": "..."}` 反序列化成 `**kwargs` 时不会和位置参数撞车，签名也能稳定向前兼容。
> - `from __future__ import annotations`：让所有类型注解按字符串延迟求值，避免运行时解析顺序问题，也能在 `Type1 | Type2` 这种 PEP 604 写法在更老 Python 上工作。

这里把"怎么执行"与"能不能并发"拆开了：

- `requires_approval`
  - 是否需要用户批准
- `read_only`
  - 是否是只读、无副作用工具
- `exclusive`
  - 是否必须独占执行
- `concurrency_safe`
  - 当前策略下是否允许和别的工具同批并发

当前 MiniBot v1 的经验规则：


| 类型           | `read_only`   | `exclusive` | 能否并发 |
| ------------ | ------------- | ----------- | ---- |
| 本地只读文件工具     | `True`        | `False`     | 可以   |
| 网络只读工具       | `True`        | `True`      | 先不要  |
| 写文件/执行命令/写记忆 | `False`       | `True`      | 不可以  |
| macOS 工具     | `False` 或保守处理 | `True`      | 不可以  |


注意 `read_only` 和 `concurrency_safe` 不是同义词。`fetch_url` 是只读的，但仍然标 `exclusive=True`，因为它走外网，多并发抓同一个站很容易触发限流。也就是：

> 能不能并发，不看"像不像只读"，看"是否足够稳定到值得并发"。

## 4. Tool 输出为什么要分两层

MiniBot 现在是两段式输出：

1. 工具返回 `ToolOutput`
2. runtime 把它物料化成 `ToolResult`

### 4.1 ToolOutput：工具语义层

工具只返回语义结果：

```python
ToolOutput.success(
    "已读取 README.md。",
    data={"path": "README.md", "file_sha256": "..."},
    content=full_text,
    content_kind="file",
    content_name="README.md",
)
```

> Python 备忘
>
> - `success(...)` / `failure(...)` 是 `@classmethod` 工厂方法（见 `tools/result.py`）。它们替代了多套构造器重载——Python 没有方法重载，工厂方法是惯用替代。
> - `ToolOutput` 本身是 `@dataclass(frozen=True)`：实例不可变，赋值会抛 `FrozenInstanceError`。这点对并发非常重要，下文 §10.7 会再讲。

`ToolOutput` 的职责：

- `summary`
  - 给模型看的简短摘要
- `data`
  - 结构化、小体积、稳定字段
- `content`
  - 大正文，可选
- `code`
  - 标准状态码，如 `success`、`invalid_args`、`not_found`、`conflict`

工具层只描述"我做了什么、结果是什么"，不要关心"要不要落盘 artifact"。

### 4.2 ToolResult：模型可见层

`ToolOutputMaterializer` 决定：

- 小内容直接内联进 `data["content"]`
- 大内容落盘为 artifact，并给模型返回 `artifact` 引用 + preview

所以模型最终看到的永远是同一种包裹结构：

```json
{
  "ok": true,
  "code": "success",
  "summary": "已读取 README.md。",
  "data": {...},
  "artifact": null,
  "truncated": false
}
```

这个统一格式的价值非常高：

- 模型不用记不同工具的私有返回格式
- runner 不用为每个工具写特判
- 大结果处理逻辑可以集中治理

## 5. 参数校验为什么放在 Registry

`ToolRegistry` 现在拆成两步：

1. `prepare(...)`
2. `invoke(...)`

### 5.1 prepare

`prepare(...)` 做这些事情：

- 校验 `args` 必须是 dict
- 查找工具
- 用 `inspect.signature(...).bind(...)` 校验参数
- 成功时返回 `PreparedToolCall`
- 失败时返回 `ToolOutput.failure(...)`

> Python 备忘
>
> - `inspect.signature(fn)` 返回函数的形参签名；`.bind(**args)` 模拟一次调用，但**不实际执行**。如果缺必填参数、有多余参数、类型不匹配关键字规则，会抛 `TypeError`。这是"在不调用工具的前提下提前校验参数"的标准技巧。
> - 返回类型 `PreparedToolCall | ToolOutput` 是 PEP 604 的联合类型语法（等价于 `typing.Union[PreparedToolCall, ToolOutput]`）。"用返回值表达成功/失败"在 Python 里没有 Rust 那么强的语言支持，靠 `isinstance` 来 narrow，配合静态类型检查器够用。

### 5.2 invoke

`invoke(...)` 只做一件事：

- 真正调用 `tool.execute(...)`
- 把异常兜底转成统一的 `ToolOutput.failure(...)`

这样拆的原因是并发：

- 主线程可以先 `prepare`
- worker 线程只跑 `invoke`
- 串行路径和并发路径共用同一套校验与错误封装

结论：凡是"执行前就能知道的错误"，都尽量在 `prepare` 解决，不要进线程池后再炸。

## 6. 并发执行的正确位置

并发应该放在 `AgentRunner`，不是放在工具里。

原因：

- runner 才知道这一轮有哪些 `tool_calls`
- runner 才知道 approval 状态
- runner 才知道哪些工具允许并发
- runner 才负责保证消息顺序稳定

### 6.1 一轮内的执行流水线

```
模型返回 tool_calls
        │
        ▼
_plan_tool_calls(...)        ← 主线程：parse args、查找 Tool、记日志
        │
        ▼
_partition_tool_batches(...) ← 主线程：按 concurrency_safe 切批
        │
        ▼
对每个 batch：
  - 单个调用 → _execute_planned_tool_call(...)
  - 多个调用 → _execute_parallel_batch(...)
        │
        ▼
按原始 tool_call 顺序 materialize、append tool message
```

要点：

- **prepare 与审批永远在主线程**：被拒绝的调用直接返回 `ToolOutput.failure("denied", ...)`，根本不会进线程池。
- **worker 线程只跑 `invoke`**：纯执行，不碰 `messages`、不碰 artifact 落盘。
- **结果按原索引回填**：哪怕 `fast` 比 `slow` 先返回，写回 `messages` 时也按模型原始顺序。

> Python 备忘 — `concurrent.futures` 三件套
>
> - `ThreadPoolExecutor(max_workers=N)`：构造一个最多 N 个线程的池，线程是 `submit` 时按需懒创建的。
> - `executor.submit(fn, *args, **kwargs)` → `Future`：把任务丢进去，立即返回一个 `Future` 句柄（非阻塞）。
> - `future.result(timeout=None)`：阻塞等结果。如果 worker 抛异常，这里会**重新抛出同样的异常**——这是我们能在主线程统一兜异常的关键。
>
> 我们在代码里用 `with ThreadPoolExecutor(...) as executor:` 这种上下文管理器写法，等价于退出时调用 `executor.shutdown(wait=True)`，会把已 submit 的任务全部跑完才返回。

### 6.2 错误隔离

`_execute_parallel_batch` 在 `future.result()` 外包了一层 try：

```python
try:
    outputs[index] = future.result()
except Exception as exc:
    outputs[index] = ToolOutput.failure(
        "error",
        f"工具 {planned.tool_call.name} 执行失败: {exc}",
        ...
    )
```

也就是说：**一个 worker 抛异常，只会污染它对应那一格输出，不会拖垮整批**。这是 runner 必须自己兜的事，工具内部当然也应该尽量不抛。

## 7. 为什么 MiniBot v1 用线程池，不先上 asyncio

因为当前 MiniBot 的主链路是同步的：

- `LLMClient.chat(...)` 是同步
- `Tool.execute(...)` 是同步
- `TurnEngine.handle_turn(...)` 是同步

而且大多数工具本来就是阻塞型 I/O：

- 本地文件读写
- `subprocess.run`
- `urllib` 抓网页

这类场景下，线程池是最小改动方案：

- 不需要重写整个调用栈
- 不需要把所有工具改成 `async def`
- 可以先拿到 80% 的收益

所以 v1 的策略是：

- 架构继续保持同步
- 单轮内局部使用 `ThreadPoolExecutor`
- 等以后工具层和 LLM 层都 async 化了，再考虑 `asyncio.gather`

> Python 备忘 — GIL（Global Interpreter Lock）
>
> CPython 解释器有一把全局锁，**同一时刻只允许一个线程执行 Python 字节码**。但当线程进入"阻塞型系统调用"时（文件 IO、socket 收发、`subprocess.wait`、`time.sleep`），它会**主动释放 GIL**，让其他线程跑。
>
> - 阻塞 IO 工具：线程池有效，因为大多数时间 GIL 是释放的。
> - 纯 CPU 工具（JSON 解析大文件、压缩、矩阵运算除非走 numpy 释放 GIL 的路径）：线程池没收益，必须走 `multiprocessing` 或 C 扩展。
>
> Python 3.13 引入了实验性的 free-threaded 构建（PEP 703）可以无 GIL，但生态还不普及。MiniBot 当前不依赖这点。

## 8. 为什么每个并发批次新建一个池

实现上是 `with ThreadPoolExecutor(max_workers=...) as executor: ...`，也就是"开池 → 跑任务 → 关池"，每个 batch 一轮。这看起来像浪费，但在 MiniBot 这个量级是最优解：

### 8.1 真实成本

- 构造 `ThreadPoolExecutor` 本身几乎零开销
- 线程是 `submit` 时按需懒创建的，每个约 50–200μs
- `__exit__` 触发的 `shutdown(wait=True)` 也是亚毫秒级

而单个工具的耗时是 1ms ~ 几秒。**信噪比 50–1000 倍**，开关池在 profiler 里看不见。

> Python 备忘 — `with` 语句和上下文管理器
>
> `with ThreadPoolExecutor(...) as executor:` 这一行其实是：
>
> ```python
> executor = ThreadPoolExecutor(...)
> try:
>     ...
> finally:
>     executor.__exit__(exc_type, exc, tb)  # = executor.shutdown(wait=True)
> ```
>
> 任何对象只要实现了 `__enter__` 和 `__exit__` 就能放进 `with`。`ThreadPoolExecutor.__exit__` 的实现是 `shutdown(wait=True)`，这就是"出 `with` 块时一定会等所有任务跑完"的来源。所以这个写法的语义不是"开池关池"，而是"保证不会漏掉清理"。

### 8.2 这种写法换来的好处

- `**max_workers` 按批裁剪**：`min(self.max_parallel_tools, len(submitted))`，一批只有 2 个工具就只起 2 个线程，不会闲置 4 个。常驻池没法这么干。
- **零状态泄漏**：批次之间线程不复用，`threading.local`、HTTP 连接缓存、SSL session 之类的隐式共享都不会跨批漂移。
- **干净的故障边界**：`with` 退出时强制 join，异常抛出也保证不会拖着孤儿线程进入下一轮 LLM 调用。
- **代码可读**：没有 executor 字段、没有 `__del__`、没有 "是否已 shutdown" 的状态位。

### 8.3 什么时候才该改成长驻池

满足下面任一条再考虑：

1. profiler 显示池构造/销毁占 turn 总耗时 >5%
2. 引入跨 turn / 跨 session 的后台任务，需要 future 跨批共享
3. 改用进程池跑 CPU-bound 工具（构造成本量级是 10ms 起，per-batch 就开始亏）

目前都不满足。**保持 per-batch，简单且正确。**

## 9. 并发批次怎么划分

当前规则非常简单：

```python
if tool.concurrency_safe:
    current_batch.append(tool_call)
else:
    flush(current_batch)
    batches.append([tool_call])
```

也就是说：

- 连续的安全工具可以同批并发
- 中间一旦出现独占工具，就先把前面的批冲刷掉
- 整轮内的批次顺序保持和模型给出的顺序一致

例子：

```text
[read_file, read_skill, exec, search_files]
```

会被切成：

```text
[[read_file, read_skill], [exec], [search_files]]
```

注意这里的设计取舍：

- 我们不跨独占工具重排
- 我们优先保持模型原始意图和可重放性
- 这比"全局最优调度"更稳

### 9.1 串行回退

`max_parallel_tools <= 1` 时直接退化为"每个调用单独成批"，整条流水线和原来的纯串行实现等价。这条路径有用：

- 调试并发相关 bug 时可以一键关掉
- 在不信任新工具并发安全性的早期可以临时收紧

### 9.2 没有显式的批大小上限

如果模型一口气返回 30 个 `read_file`，它们会被塞进同一批，再用 `max_parallel_tools` 个线程逐个消化。功能上没问题，但你看到的 emit 会一次性涌出 30 行。如果想让 CLI 体感更线性，可以再加一道按 `max_parallel_tools` 切片的逻辑。当前没做，留给未来优化。

## 10. 并发时最容易犯的错

### 10.1 在 worker 线程里做审批

不要这样做。approval 必须在主线程先判定。

否则会出现：

- 多线程同时弹审批
- 审批顺序混乱
- 被拒绝的任务已经开始执行

正确做法：

- 主线程先 `prepare`
- 主线程先 `approve`
- 只有允许的调用才提交到线程池

### 10.2 在 worker 线程里直接写消息

不要让线程自己拼 `tool` message 或直接修改 `messages/events`。

否则会出现：

- 顺序不稳定
- artifact 引用和消息对不上
- 同一轮的结果难以复现

正确做法：

- worker 只返回 `ToolOutput`
- 主线程统一 `materialize`
- 主线程按原始 `tool_call` 顺序 append `tool` message

### 10.3 把可变副作用工具错误标成 `concurrency_safe`

一个工具如果会：

- 改文件
- 改全局记忆
- 改外部系统状态
- 隐式依赖严格顺序

就不要标成 `concurrency_safe`。宁可保守串行，也不要在并发下偶现一致性问题。

### 10.4 在 `data` 里塞大正文

大正文应该放 `content`，不是 `data`。

原因：

- `ToolOutput.data` 有大小约束
- `content` 才会被统一内联/落盘
- 并发场景下，统一 materialize 才能保证所有大结果走同一条路径

### 10.5 期待同批内 worker 之间有依赖

同一批的所有调用必须**互相独立**。不要假设：

- "B 可以读 A 写出来的文件"
- "B 可以拿到 A 的返回值"
- "A 一定先完成"

如果存在依赖，就让模型分两轮调用，或者把那个工具标成 `exclusive=True`。

### 10.6 把超时寄希望于线程池

Python 的线程**无法被强制取消**。`ThreadPoolExecutor` 没有"杀线程"接口，`future.cancel()` 只能取消还没开始的任务，已经在跑的工具一旦卡住，整批的 `shutdown(wait=True)` 就会一起卡。

所以：

- 工具内部必须自己设超时（`subprocess.run(timeout=...)`、`urlopen(timeout=...)`）
- 不要假设 runner 会替你兜超时
- 文件 IO 类工具默认不设 IO 超时是个有意识的取舍——前提是本地磁盘是健康的

> Python 备忘 — 线程为什么不能取消
>
> 历史上 Python 没有提供 `Thread.kill()` / `Thread.cancel()`，因为强制中断一个正在跑的线程无法保证它持有的锁、文件句柄、临时状态被正确清理（参考 Java 当年废弃 `Thread.stop()` 的教训）。
>
> 替代手段：
>
> - 让线程自己周期性地检查一个 `threading.Event`，主动退出（适合循环体）
> - 用 `daemon=True` 让线程在主进程退出时跟着死（注意：这是"退出时"，不是"任意时刻"）
> - 把工作转到子进程，需要时 `process.terminate()`
>
> `ThreadPoolExecutor` 的 worker 默认就是 daemon 线程（CPython 实现细节），所以主程序整个崩溃时不会被卡死，但**进程内**没法选择性杀掉某个 worker。

### 10.7 在 `ToolExecutionContext` 里塞可变状态

`ToolExecutionContext` 现在是 `@dataclass(frozen=True)`，且只有 `session_id`。这是**故意**的：它会被同批多个 worker 共享。

如果以后往里加字段，请遵守：

- 全部不可变（基本类型 / frozen dataclass）
- 不放可变集合（`list`、`dict`、`set`）
- 不放有内部状态的客户端对象（除非那个对象自己保证线程安全）

否则就会从"看起来工作"逐渐演变成"偶现脏数据"。

> Python 备忘 — `@dataclass(frozen=True)` 与线程安全
>
> - `frozen=True` 会让 dataclass 的所有字段在 `__init__` 后变成只读，赋值会抛 `FrozenInstanceError`。
> - 但"frozen"只管字段引用本身——如果字段指向一个 `list`，你仍然可以 `ctx.items.append(...)` 改动那个 list。所以"不可变上下文"的含义是**字段指向的对象本身也必须是不可变值**（`int`、`str`、`tuple`、frozen dataclass 等）。
> - `threading.local()` 是一种"每个线程一份独立副本"的存储；如果某个第三方库（比如 `requests.Session`）内部用了 `threading.local`，你就不能把它当成"跨线程共享的客户端"——每个 worker 会看到不同的内部状态，缓存可能失效但行为不会出错。真正会出错的是把不带任何同步的可变对象（比如普通 `dict`）共享给多个 worker 同时写。

## 11. 当前不解决什么问题

诚实声明 v1 的边界，免得被误用：

- **没有 per-host 限流**：`fetch_url` 用 `exclusive=True` 兜底，所有外网请求强制串行。这是粗粒度的解法，未来可以改成 per-domain 信号量。
- **没有整批 wall-clock 超时**：worker 一旦在某个工具里 hang 住，主线程会卡在 `with` 退出处。依赖每个工具自带超时。
- **没有线程取消**：上文已说，无法中途打断。
- **没有跨 turn / 跨 session 的并发**：每次 LLM 调用之间仍是严格同步。要做"模型在思考下一轮时后台预取上一轮 artifact"之类的优化，需要更大的架构改动。
- **没有 async LLM**：`LLMClient.chat` 同步阻塞，多轮 turn 之间没法做 pipelining。

这些不是 bug，是当前阶段的取舍。把边界写清楚，比偷偷做半套要好。

## 12. 新工具应该怎么设计

推荐按下面顺序做：

### 第一步：先定并发属性

问自己 3 个问题：

1. 它是不是只读？
2. 它是不是稳定到值得并发？
3. 它会不会和别的工具抢同一份状态？

如果答案是：

- 只读 + 稳定 + 无共享副作用：`read_only=True`
- 但仍需保守串行：再加 `exclusive=True`
- 有副作用：保持默认即可

### 第二步：把参数 schema 写完整

要求：

- 名字稳定
- schema 自解释
- `required` 明确
- `additionalProperties=False`

不要靠提示词口头约束参数格式，schema 才是机器契约。

### 第三步：返回 `ToolOutput`

规则：

- 简短摘要放 `summary`
- 短结构化字段放 `data`
- 大正文放 `content`
- 失败时用统一 `code`

### 第四步：自带超时和资源上限

- 网络请求 / `subprocess`：必须设 timeout
- 文件读写：要设大小上限（参考 `read_file._MAX_SIZE`）
- 不要假设 runner 会兜底

### 第五步：不要自己做这些事

工具里不要自己做：

- artifact 落盘
- JSON 字符串封装
- 并发控制
- 消息写回
- 参数签名校验

这些都应该由 runtime 或 registry 统一处理。

## 13. 一个合格的新工具例子

```python
class ReadFooTool(Tool):
    @property
    def name(self) -> str:
        return "read_foo"

    @property
    def description(self) -> str:
        return "读取 foo 配置"

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def execute(self, *, context: ToolExecutionContext, path: str, **kwargs: Any) -> ToolOutput:
        del context, kwargs
        content = ...
        return ToolOutput.success(
            "已读取 foo 配置。",
            data={"path": path},
            content=content,
            content_kind="file",
            content_name=path,
        )
```

这个工具做对了几件事：

- 输入 schema 明确
- 并发属性明确（read_only，不带 exclusive，可与其他只读工具同批）
- 输出走统一 `ToolOutput`
- 不碰 runtime 关心的东西

## 14. 一套实用的设计原则

最后给一套足够实战的规则：

1. 模型输出先标准化，再进工具层。
2. 工具只做业务，不做调度。
3. 输出统一成 `ToolOutput -> ToolResult` 两段式。
4. 参数校验和异常封装集中在 `ToolRegistry`。
5. 并发调度集中在 `AgentRunner`。
6. 并发只给"安全只读"工具，不给"理论上可并发"的工具。
7. 线程只负责执行，主线程负责审批、顺序、物料化和消息落地。
8. 共享给 worker 的上下文必须不可变。
9. 超时和资源上限是工具自己的责任，不是 runner 的。
10. 先做保守正确，再做激进并发。

如果你后面继续演进 MiniBot，这一套结构还能继续扩展到：

- 多会话并发
- 后台任务
- 子 agent
- async provider
- 更细粒度的调度策略（per-host 限流、批大小上限、整批超时）

但前提都是一样的：先把 tool call 契约和统一格式设计好。

## 附录 A：本项目用到的 Python 特性速查

下面把全文出现过的关键语言/库特性集中列一遍，方便回查。

### A.1 类型注解相关


| 写法                                   | 含义                             | 出现位置                                                           |
| ------------------------------------ | ------------------------------ | -------------------------------------------------------------- |
| `from __future__ import annotations` | 所有注解延迟求值（变成字符串），避免循环引用、加快导入    | 几乎所有模块顶部                                                       |
| `X                                   | Y`                             | PEP 604 联合类型，等价于 `Union[X, Y]`                                 |
| `TypeAlias` / `Literal["a", "b"]`    | 给类型起别名 / 字面量类型                 | `tools/base.py` 的 `ToolLayer = Literal["kernel", "extension"]` |
| `dict[str, Any]`                     | 内置泛型（PEP 585，3.9+），不再需要 `Dict` | tool 的 `parameters` 返回值                                        |
| `Callable[[str], None]`              | 可调用对象类型                        | `event_handler` 形参                                             |
| `TYPE_CHECKING`                      | 只在类型检查阶段导入，运行时跳过，破循环依赖         | `tools/base.py`                                                |


### A.2 类与抽象


| 写法                                     | 含义                                               |
| -------------------------------------- | ------------------------------------------------ |
| `class Tool(ABC):` + `@abstractmethod` | 抽象基类。子类不实现就无法实例化（运行时强制）                          |
| `@property`                            | 把方法暴露成属性访问                                       |
| `@dataclass(frozen=True)`              | 自动生成 `__init__/__repr__/__eq__/__hash`__，且实例不可变  |
| `@classmethod`                         | 工厂方法常用形式（`ToolOutput.success`、`Config.from_env`） |
| `def fn(self, *, kw_only_arg)`         | `*` 之后的参数必须以关键字传，签名更稳                            |
| `**kwargs: Any`                        | 接收剩余关键字参数，配合 `inspect.signature.bind` 校验         |


### A.3 反射与运行时校验


| 写法                       | 含义                             |
| ------------------------ | ------------------------------ |
| `inspect.signature(fn)`  | 拿到函数签名对象                       |
| `.bind(*args, **kwargs)` | 模拟一次绑定，但不调用；参数不匹配抛 `TypeError` |
| `isinstance(x, Cls)`     | 运行时类型判断；同时让静态检查器 narrow 类型     |


### A.4 并发（重点）


| 写法 / 概念                                                | 含义                                                 |
| ------------------------------------------------------ | -------------------------------------------------- |
| `concurrent.futures.ThreadPoolExecutor(max_workers=N)` | 高层线程池；`submit/map/shutdown` 三件套                    |
| `executor.submit(fn, *args)` → `Future`                | 非阻塞提交，立即拿到 future 句柄                               |
| `future.result(timeout=None)`                          | 阻塞等结果；worker 内异常会在这里**重新抛出**                       |
| `future.cancel()`                                      | 仅对**还没开始执行**的任务有效；正在跑的杀不掉                          |
| `with ThreadPoolExecutor(...) as ex:`                  | 退出时自动 `shutdown(wait=True)`，等所有任务跑完                |
| GIL（全局解释器锁）                                            | 同一时刻只有一个线程跑 Python 字节码；阻塞 IO 会主动释放                 |
| `threading.Lock` / `threading.Event`                   | 显式同步原语；MiniBot 测试里 `_TrackingState.lock` 就是 `Lock` |
| `threading.local()`                                    | 每线程独立副本；某些库内部隐式用它做缓存                               |
| daemon 线程                                              | 主进程退出时跟着死；`ThreadPoolExecutor` 的 worker 默认是 daemon |
| `asyncio` / `async def`                                | 协程模型，**MiniBot v1 没用**；适合纯 IO 且全栈都能 await 的场景      |


并发模型对比口诀：

- **CPU 密集** → `multiprocessing` 或 C 扩展（绕开 GIL）
- **IO 密集 + 同步代码** → `ThreadPoolExecutor`（MiniBot 当前选择）
- **IO 密集 + 全栈 async** → `asyncio` + `asyncio.gather`

### A.5 上下文管理与异常


| 写法                   | 含义                                                   |
| -------------------- | ---------------------------------------------------- |
| `with obj:`          | 调用 `obj.__enter_`_ / `__exit__`；保证清理一定执行             |
| `try/except/finally` | `finally` 在异常路径上也会跑；可保证状态恢复                          |
| `raise X from Y`     | 在新异常里保留原异常作为 `__cause__`，便于 traceback 阅读             |
| `del unused_var`     | 显式丢弃，主要用来骗过 lint "未使用参数" 警告（如 `del context, kwargs`） |


### A.6 其它细节


| 写法                                                 | 含义                                                |
| -------------------------------------------------- | ------------------------------------------------- |
| `Path(...).resolve()`                              | 转绝对路径 + 解符号链接，常用于安全检查                             |
| `Path.is_relative_to(base)`                        | 判断路径是否落在某目录下（沙盒检查的关键）                             |
| `subprocess.run(..., timeout=N)`                   | 同步执行子进程；超时抛 `TimeoutExpired`                      |
| `urllib.request.urlopen(req, timeout=N)`           | 标准库 HTTP；带 `timeout` 是它能在线程池里安全使用的前提              |
| `json.loads / json.dumps(..., ensure_ascii=False)` | 反序列化 / 序列化；`ensure_ascii=False` 让中文不被转义成 `\uXXXX` |
| `hashlib.sha256(text.encode()).hexdigest()`        | 文件指纹；用于 `expected_sha256` 乐观锁                     |


如果上面有任何一条你不熟悉，强烈建议在动并发相关代码之前先去读一下对应的 Python 官方文档。并发问题难调，多读几页文档能省好几个晚上的 debug。