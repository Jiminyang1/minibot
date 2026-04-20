---
name: notes
description: 使用 macOS 备忘录/Notes 工具搜索、创建和追加笔记
tools:
  - exec
  - mcp__macos_system__notes_search
  - mcp__macos_system__notes_create
  - mcp__macos_system__notes_append
---
当用户要查找或记录笔记时：

1. 如果用户想找旧内容，先用 `mcp__macos_system__notes_search` 定位，再决定是否追加。
2. 新建笔记时，至少确认 `title` 和 `content`。
3. 追加内容时，必须先拿到明确的 `note_id`，不要根据模糊标题直接猜目标。
4. 如果用户没有指定文件夹，可以使用默认文件夹；但回答里要说明实际写入到了哪个文件夹。
5. 不要假设 Notes 已经打开。先直接调用；如果报 `Application isn’t running` 或 `-600`，agent 应先用 `exec` 执行 `open -a Notes`，然后重试一次。
6. 搜索命中多个结果时，不要直接选择其中一个追加；先把候选标题 / 文件夹 / 预览列给用户确认。
7. 新建和追加都要避免把空字符串或只有空白的内容直接提交给工具。
8. 返回结果时给出标题、文件夹和简短预览，方便后续继续操作。
9. 常见错误处理：
   - `permission_denied`：提示检查 Notes 权限或自动化权限
   - `invalid_args`：重点检查空标题、空内容
   - `not_found`：重点检查 `note_id` 或文件夹名
   - `error`：如果包含 `Application isn’t running` 或 `-600`，先执行 `open -a Notes` 再重试一次；如果重试后仍失败，再向用户解释错误
