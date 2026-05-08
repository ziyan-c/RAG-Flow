# table content 里面存在图片而非小图标 
这些图片被放在table content的文本内容里而非单独的图片层。
目前的处理只是把它当作小图标而没有解释性vlm说明，只是patching，而非captioning

# MinerU 可能漏掉文本附近的小图标

MinerU/OCR 对按钮、工具栏、状态图标等小视觉元素不稳定。文本中出现明显空位、孤立标点或 “click .” 这类模式时，通常需要 patching 阶段用 VLM 补回 `[Icon: name]`。

# 表格脚注裁剪依赖 bbox 和相邻 block

表格脚注有时不在 table bbox 内，而是在表格下方相邻区域。当前逻辑会尽量基于 table bbox、续表关系和后续 block 推断脚注 crop，但它仍然依赖 MinerU 的 bbox 质量。

# indexing 的 source metadata 必须一致

text indexing 和 visual indexing 必须使用相同的 `source` 值。文本点 ID 使用 chunk metadata，视觉页点 ID 使用独立 visual-page namespace，但两者仍然依赖同一个 `source` 值来让检索、过滤和上下文回填对齐。

# View PDF 涂色层可能低估实际处理范围

`PATCHING_VIEW.pdf`、`CAPTIONING_VIEW.pdf` 和 `CHUNKING_VIEW.pdf` 是人工审阅用的可视化辅助文件，不是处理覆盖率的唯一真相。它们依赖 MinerU 输出的 `bbox`、图片 block、表格 block 和后续阶段保存的 overlay metadata。

因此可能出现这种情况：PDF 上某些区域看起来没有涂色，但实际上对应文本、图片或表格内容已经被 patching、captioning 或 chunking 处理过。

反过来也要注意：未涂色并不自动等于“已经处理过”。它只说明当前 view PDF 没有画出对应 overlay。这个区域到底有没有进入某个阶段，仍然要回到 JSON、metadata 和运行统计确认。

常见原因：

- MinerU 的 bbox 本身可能偏小、偏移、合并多个视觉区域，或完全缺失。
- 某些处理逻辑基于 JSON 字段、图片文件、表格续表关系或相邻文本上下文，而不是每一步都生成一个新的可视化 bbox。
- 重叠区域、透明涂层和相邻 chunk 颜色覆盖可能让部分区域看起来不明显。
- table continuation 会把主表和续表关系写入 metadata；view PDF 画的是保存下来的表格/块级 bbox，不一定能精确覆盖每个单元格或每个视觉细节。
- 如果某个 block 没有合法 bbox，view PDF 可能无法画出对应 overlay，但该 block 的文本字段仍可能进入后续 JSON 处理。
- 如果 view PDF 使用的 `input-pdf` 不是 MinerU 解析时对应的 origin/source PDF，或者 PDF 存在 rotation/cropbox 差异，涂色层可能整体偏移。
- view PDF 也可能高估实际处理范围：它画的是候选区域或 metadata 区域，不一定等于“模型实际改写过这个区域”。

分阶段看：

- `PATCHING_VIEW.pdf` 画的是 patching 候选字段、表格续表 crop 区域和 linked inline icon bbox。`header`、`footer`、`page_number`、`aside_text`、`page_footnote`、`equation`、`seal`、`chart` 等 ignored block 不会被 patching，也通常不会出现在 patching view 里。`table_caption` 和 `image_caption` 默认也不是 patching 目标。
- `CAPTIONING_VIEW.pdf` 画的是 caption 目标 image block，以及被选择为上下文的 nearby text block。captioning 真正发送给 VLM 的图片来自 MinerU 导出的 `img_path` 文件，而不是 PDF overlay 本身；如果图片 block bbox 不准，view 可能看起来不准。被放进 prompt 的上下文 block 如果没有合法 bbox，也不会被涂色。
- `CHUNKING_VIEW.pdf` 只画 chunk metadata 里的 `bboxes_by_page`。如果某个 chunk 的 `chunk_content` 来自没有合法 bbox 的文本、图片说明或表格字段，它仍然可能进入 chunk，但 view PDF 没有区域可画。overlap chunk 也可能让同一个 bbox 被多个颜色覆盖。
- 如果 chunking 使用 `mode=page`，当前 page-level chunk metadata 不保存 `bboxes_by_page`，因此 `CHUNKING_VIEW.pdf` 可能没有涂色；这不代表 page chunks 没有生成。默认 `RAG_FLOW_CHUNK_MODE=auto` 会优先走 section/token chunking，不属于这个特殊路径。

判断是否真的被处理，应该优先对照这些产物：

- `*_PATCHED.json` 中相关字段是否出现 patching 结果或 `vlm-small-icon-*` metadata
- `*_PATCHED_CAPTIONED.json` 中图片 block 是否出现 `image_description_vlm`
- `*_PATCHED_CAPTIONED_CHUNKED.json` 中 chunk 是否包含相关正文
- chunk metadata 里的 `block_indices`、`page_indices`、`bboxes_by_page`
- 阶段运行日志中的 processed/patched/captioned/chunked 统计

结论：view PDF 中“未涂色”只能说明当前可视化层没有对应可画 bbox，不能直接说明该区域没有被处理。
