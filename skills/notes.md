---
name: notes
description: 使用 macOS 备忘录/Notes 工具搜索、创建和追加笔记
triggers:
  - notes
  - note
  - 笔记
  - 备忘录
  - 记事
  - 记录
tools:
  - notes_search
  - notes_create
  - notes_append
summary: 处理笔记任务时，先区分“搜索现有内容”还是“新建/追加”，写入前确认目标笔记或文件夹。
---
当用户要查找或记录笔记时：

1. 如果用户想找旧内容，先用 `notes_search` 定位，再决定是否追加。
2. 新建笔记时，至少确认 `title` 和 `content`。
3. 追加内容时，必须先拿到明确的 `note_id`，不要根据模糊标题直接猜目标。
4. 如果用户没有指定文件夹，可以使用默认文件夹；但回答里要说明实际写入到了哪个文件夹。
5. 返回结果时给出标题、文件夹和简短预览，方便后续继续操作。
