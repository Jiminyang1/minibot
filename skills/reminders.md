---
name: reminders
description: 使用 macOS 提醒事项工具查看、创建和完成提醒
tools:
  - exec
  - mcp__macos_system__reminders_list
  - mcp__macos_system__reminders_create
  - mcp__macos_system__reminders_complete
---
当用户要创建、查看或完成提醒事项时：

1. 查看列表时，优先明确是 `open`、`completed` 还是 `all`。
2. 创建提醒时，至少要有清晰标题；如果用户给了明确时间，再填写 `due_at`。
3. 时间参数优先使用本地 ISO 8601 日期时间字符串（例如 `2026-04-20T18:00:00`）；不要把含糊自然语言时间直接塞进参数。
4. 如果用户给的是带时区的时间，可以先转本地时间；当前 service 也会做一次本地化兜底。
5. 完成提醒前，先确认拿到的是正确的 `reminder_id`，避免误完成；不要根据标题模糊匹配直接完成。
6. 不要假设 Reminders 已经打开。先直接调用；如果报 `Application isn’t running` 或 `-600`，agent 应先用 `exec` 执行 `open -a Reminders`，然后重试一次。
7. 如果用户没指定 `list_name`，可以让系统默认列表接管；但回复里要说明实际落到哪个列表。
8. 返回结果时说明提醒标题、所属列表、是否已完成、截止时间是否存在。
9. 常见错误处理：
   - `permission_denied`：提示检查提醒事项权限或自动化权限
   - `invalid_args`：重点检查标题为空、时间格式错误
   - `not_found`：重点检查 `reminder_id` 或指定列表
   - `error`：如果包含 `Application isn’t running` 或 `-600`，先执行 `open -a Reminders` 再重试一次；如果重试后仍失败，再向用户解释错误
