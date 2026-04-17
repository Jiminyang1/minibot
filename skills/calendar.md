---
name: calendar
description: 使用 macOS 日历工具查看日程或创建事件
triggers:
  - calendar
  - 日历
  - 日程
  - 会议
  - event
  - schedule
tools:
  - calendar_list_events
  - calendar_create_event
summary: 处理日历任务时，先确认时间窗口和必填字段；写入前不要猜缺失的结束时间。
---
当用户要查看或安排日历事件时：

1. 如果是查询类请求，优先使用 `calendar_list_events`，并把时间窗口限定清楚, 如果用户没有明确制定时间窗口，则使用当前时间作为开始时间，并且时间窗口为7天。
2. 如果是写入类请求，至少确认 `title`、`start_at`、`end_at`。
3. 如果用户没有给出结束时间、日期或时区相关细节，不要猜，先追问。
4. 如果用户想安排会议，必要时可以先查同一时间窗口是否已有事件，再创建新事件。
5. 返回结果时说明事件标题、时间范围和目标日历。
