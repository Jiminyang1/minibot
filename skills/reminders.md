---
name: reminders
description: 使用 macOS 提醒事项工具查看、创建和完成提醒
triggers:
  - reminders
  - reminder
  - 提醒
  - 提醒事项
  - todo
  - 待办
tools:
  - reminders_list
  - reminders_create
  - reminders_complete
summary: 处理提醒事项时，优先把任务拆成明确标题和可选截止时间；完成操作前确认目标条目。
---
当用户要创建、查看或完成提醒事项时：

1. 查看列表时，优先明确是 `open`、`completed` 还是 `all`。
2. 创建提醒时，至少要有清晰标题；如果用户给了明确时间，再填写 `due_at`。
3. 不要把含糊的自然语言时间直接塞进工具参数；需要时先追问，再生成标准时间字符串。
4. 完成提醒前，先确认拿到的是正确的 `reminder_id`，避免误完成。
5. 返回结果时说明提醒标题、所属列表和截止时间是否存在。
