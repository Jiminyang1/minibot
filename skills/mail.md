---
name: mail
description: 使用 macOS Mail/Mail.app 查看邮箱、搜索邮件、读取正文、创建草稿和发送邮件
tools:
  - exec
  - mcp__macos_system__mail_list_mailboxes
  - mcp__macos_system__mail_list_messages
  - mcp__macos_system__mail_search_messages
  - mcp__macos_system__mail_get_message
  - mcp__macos_system__mail_create_draft
  - mcp__macos_system__mail_send_message
---
当用户要处理 macOS Mail 邮件时：

1. 用户问“最近邮件”“收件箱里有什么”时，优先调用 `mcp__macos_system__mail_list_messages`，默认使用 `days_back=7`、`limit=10`、`unread_only=false`。不要把 `recent` 或 `latest` 当成搜索关键词。
2. 用户问“未读邮件”时，优先调用 `mcp__macos_system__mail_list_messages`，默认使用 `days_back=30`、`limit=20`、`unread_only=true`。如果用户明确说“全部未读”，可以传 `days_back=null`，但仍要控制 `limit`。
3. 用户说“最近 N 封”且没有时间窗口时，传 `limit=N`、`days_back=null`；用户说“最近 X 天”时，传 `days_back=X`，并用 `limit` 控制最多返回多少封。
4. 如果用户没有指定邮箱，可以先用 `mcp__macos_system__mail_list_mailboxes` 查看可用 mailbox；常见收件箱名字是 `INBOX`。
5. 搜索具体人名、主题或关键词时用 `mcp__macos_system__mail_search_messages`。默认只搜主题和发件人；只有用户明确要求搜正文时才传 `include_body=true`。
6. 工具返回的是摘要和 `message_id`；需要完整正文时再用 `mcp__macos_system__mail_get_message`。
7. 读取正文时，优先带上搜索/列表结果里的 `account_name` 和 `mailbox_name`，这样可以减少遍历范围，也避免大量邮件导致超时。
8. 创建草稿用 `mcp__macos_system__mail_create_draft`。如果用户只是让你“帮我写一封邮件”，优先创建草稿或先展示拟写内容，不要直接发送。
9. 发送邮件前必须确认收件人、主题、正文，必要时也确认 cc/bcc/sender；确认后调用发送工具时必须显式传 `confirm_send=true`。不要根据模糊姓名猜邮箱地址。
10. `sender` 是 Mail 里的发件人身份字符串，可选；除非用户明确指定，否则让 Mail 使用默认发件账号。
11. 邮件内容可能敏感。回复里只摘要必要信息，不要把完整邮件正文、完整地址列表或敏感内容重复给无关上下文。
12. 不要假设 Mail 已经打开。先直接调用；如果报 `Application isn’t running` 或 `-600`，agent 应先用 `exec` 执行 `open -a Mail`，然后重试一次。
13. 不要用 `exec` 直接读取、搜索或发送邮件；`exec` 只用于必要时打开 Mail.app，邮件操作走 MCP tools。
14. 常见错误处理：
   - `permission_denied`：提示检查系统设置里的 Mail 自动化/Apple Events 权限
   - `invalid_args`：重点检查空主题、空正文、收件人为空或 sender 不可用
   - `not_found`：重点检查 `account_name`、`mailbox_name` 或 `message_id`
   - `timeout`：缩小邮箱范围、减少 `limit`，或先列 mailbox 后在指定 mailbox 搜索
