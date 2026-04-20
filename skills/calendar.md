---
name: calendar
description: 使用 macOS 日历工具查看和创建日程事件
tools:
  - exec
  - mcp__macos_system__calendar_list_events
  - mcp__macos_system__calendar_create_event
---
当用户要查询或创建日程事件时：

1. 列出事件时，优先确认时间范围（今天 / 本周 / 指定日期），不要默认抓取一整年数据。
2. 创建事件前，必须确认标题、开始/结束时间；地点和备注可选但建议补齐。
3. 时间参数优先使用本地 ISO 8601 日期时间字符串（例如 `2026-04-20T10:00:00`）。不要把模糊自然语言直接塞进参数。
4. 如果用户给的是带时区的时间，可以先换算成当前机器的本地时间再调用；当前 macOS MCP service 也会做一次本地化兜底。
5. 不要假设 Calendar 已经打开。先直接调用；如果报 `Application isn’t running` 或 `-600`，agent 应先用 `exec` 执行 `open -a Calendar`，然后重试一次。
6. 创建事件时，结束时间必须晚于开始时间；如果用户只给了开始时间但没给结束时间，先补问，不要擅自猜很长时间范围。
7. 如果指定了 `calendar_name`，失败时要优先区分是“日历不存在”“日历只读”还是“权限不足”，不要把不同错误混成一句泛化提示。
8. 工具返回后，在给用户的回复里总结关键信息：标题、时间段、所在日历、是否写入成功。
9. 常见错误处理：
   - `permission_denied`：提示检查系统设置里的日历权限或 Apple Events 权限
   - `invalid_args`：重点检查时间格式、开始/结束顺序
   - `not_found`：重点检查 `calendar_name`
   - `error`：如果包含 `Application isn’t running` 或 `-600`，先执行 `open -a Calendar` 再重试一次；如果重试后仍失败，再向用户解释错误
