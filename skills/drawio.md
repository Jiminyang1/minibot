---
name: drawio
description: 使用 draw.io 生成流程图、架构图、ER 图、时序图、类图、网络图或 `.drawio` 文件。优先使用官方 draw.io MCP 打开编辑器；如果 MCP 不可用，则直接生成 `.drawio` XML 并按需导出 PNG/SVG/PDF。
tools:
  - fetch_url
  - write_file
  - exec
---
当用户要“画图 / 生成图 / draw.io / .drawio / 导出 PNG/SVG/PDF”时，使用这个 skill。

1. 优先检查当前可用工具里是否有官方 draw.io MCP：
   - `mcp__drawio__open_drawio_mermaid`
   - `mcp__drawio__open_drawio_xml`
   - `mcp__drawio__open_drawio_csv`
2. 选工具时不要乱用：
   - 简单流程图、时序图、状态图、快速草图，优先 `open_drawio_mermaid`
   - 架构图、ER 图、类图、泳道图、需要 container / layer / style / 自定义 shape 的图，优先 `open_drawio_xml`
   - 只有明确是表格驱动的图，才用 `open_drawio_csv`
3. 如果走 `open_drawio_xml`，复杂图先用 `fetch_url` 读取官方 XML 参考：
   - `https://raw.githubusercontent.com/jgraph/drawio-mcp/main/shared/xml-reference.md`
4. 如果 draw.io MCP 不可用，就直接生成 `.drawio` 文件：
   - 生成原生 `mxGraphModel` XML，不要生成 Mermaid 再转存
   - 用 `write_file` 写到当前工作目录
   - 文件名用小写 kebab-case，例如 `login-flow.drawio`
5. 如果用户要求导出 `png` / `svg` / `pdf`，先检查 draw.io CLI：
   - 先试 `which drawio`
   - macOS 再试 `/Applications/draw.io.app/Contents/MacOS/draw.io`
   - 导出命令：`drawio -x -f <format> -e -b 10 -o <output> <input.drawio>`
   - 导出成功后，优先保留 `<name>.drawio.<format>` 这种双后缀文件名
6. 如果本机能打开文件，最后用 `open <file>`；打开失败时，在回复里给出绝对路径。

关键约束：

- 不要输出 XML comments。
- 根结构必须包含 `<mxCell id="0"/>` 和 `<mxCell id="1" parent="0"/>`。
- 每个 edge 都必须带子节点 `<mxGeometry relative="1" as="geometry" />`，不要写成自闭合 edge。
- `value` 里出现 HTML 时必须做 XML escape，style 默认带 `html=1`。
- 所有 `mxCell id` 必须唯一。
- 优先生成未压缩 XML，不要自己产出 base64/deflate 压缩内容。
