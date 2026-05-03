#set document(title: "RAG-Flow Patching 参数选择实验报告")
#set page(margin: 0.70in)
#set text(size: 9.35pt)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[RAG-Flow Patching 参数选择实验报告]
  #linebreak()
  #text(size: 10pt)[Small-Icon Patching with SGLang VLM]
]

= 摘要

本文目标不是单独比较 `DPI`，而是为 RAG-Flow patching 阶段选择一套可直接用于生产的
参数。实验分两类：第一类是 `100` 页吞吐实验，用来选择 `concurrency` 和 `batch_size`；
第二类是小图标密集页质量实验，用来选择 `DPI`。其余参数如 `max_new_tokens`、
`checkpoint_interval`、`page_window_size` 和 `SGLang mem_fraction_static` 采用工程约束
和观测结果共同决定。

最终推荐配置如下：

#table(
  columns: 4,
  inset: 4.2pt,
  [参数], [推荐值], [主要目标], [选择依据],
  [`RAG_FLOW_PATCH_DPI`], [`250`], [小图标补全质量], [缺图标字段 `6/8 correct`, `2/8 partial`, `0/8 wrong`],
  [`RAG_FLOW_PATCH_CONCURRENCY`], [`10`], [请求并发], [`c=10-15` 进入吞吐平台区；`10` 更稳],
  [`RAG_FLOW_PATCH_BATCH_SIZE`], [`140`], [减少请求轮次], [`b=140` 吞吐最高，`b=200` 已下降],
  [`RAG_FLOW_SGLANG_MEM_FRACTION_STATIC`], [`0.75`], [显存余量], [SGLang 稳定预分配，给驱动、图像输入和运行波动保留空间],
  [`RAG_FLOW_PATCH_MAX_NEW_TOKENS`], [`8000`], [避免长表格/长列表输出截断], [足够覆盖跨页表格和多图标列表，又低于服务端 `100k` context],
  [`RAG_FLOW_PATCH_CHECKPOINT_INTERVAL`], [`30`], [失败恢复], [降低长任务重跑成本，IO 开销相对 VLM 推理可忽略],
  [`RAG_FLOW_PATCH_PAGE_WINDOW_SIZE`], [`200`], [PDF 渲染边界], [避免整本文档一次 rasterize，同时保留跨页表格/续表上下文],
)

= 实验设置

吞吐实验使用目标技术手册连续 `100` 页样本：`page_idx=50-149`。该样本包含
`1394` 个 content blocks，其中 `987` 个字段会提交给 VLM。这个样本足够大，
可以观察并发和 batch 的平台区与下降点。

质量实验使用同一手册中小图标密集的三个参考页：`313`、`320`、`435`。这三页共
形成 `25` 个 VLM 提交字段，其中 `8` 个字段人工确认原 OCR 确实缺失 UI 图标。
质量判分不是数 `[Icon: ...]` 的数量，而是人工对照 PDF crop、MinerU OCR 和模型输出：
`correct` 表示语义和位置基本正确，`partial` 表示主要信息补回但标签或文本保持有瑕疵，
`wrong` 表示关键图标错漏或普通文本被幻觉插入图标。

#table(
  columns: 2,
  inset: 5pt,
  [GPU], [`NVIDIA GeForce RTX 5090`, 32607 MiB VRAM],
  [模型服务], [`SGLang` OpenAI-compatible API],
  [VLM], [`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`],
  [Context length], [`100000` tokens],
  [100 页吞吐样本], [`987` VLM requests],
  [三页质量样本], [`25` submitted fields；`8` icon-required fields],
)

#pagebreak()

= 指标和评判标准

为了让参数选择可复现，所有实验都固定同一份 MinerU `content_list`、同一套 patching
候选生成逻辑、同一个 SGLang 服务和同一个 VLM。每轮只改变目标参数，其他参数保持不变。

#block(breakable: false)[
  #table(
    columns: (0.95fr, 1.75fr, 1.25fr, 1.25fr),
    inset: 3.8pt,
    [实验], [扫描值], [固定条件], [判定目标],
    [`concurrency`], [`1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16`], [`DPI=250`, `batch_size=15`], [找吞吐平台区和失败点],
    [`batch_size`], [`3, 6, 9, 12, 15, 18, 24, 30, 36, 48, 60, 80, 100, 140, 200`], [`DPI=250`, `concurrency=10`], [找请求轮次收益的峰值和下降点],
    [`DPI` 吞吐], [`200, 250, 300`], [`concurrency=10`, `batch_size=140`], [估算分辨率带来的速度和资源成本],
    [`DPI` 质量], [`200, 250, 300`], [参考页 `313, 320, 435`，共 `25` 个字段], [人工判读小图标召回和误报],
  )
]

吞吐指标按 `requests_per_sec = requests_submitted / elapsed_sec` 计算。`requests_submitted`
是实际送入 VLM 的 patching 字段数量，不是 PDF 页数，也不是图标数量。`llm_batches`
表示调度轮次；`GPU util avg/max`、显存和功耗来自运行时采样。`status=0` 表示该轮完整运行，
失败轮次仍保留在数据中，但不作为推荐点。

质量指标的单位也是“字段”。`all_submitted_fields` 用来观察普通文本是否被误插入图标；
`icon_required_fields` 只统计人工确认原 OCR 缺图标的字段。`correct` 表示图标语义和插入
位置基本正确，且原文没有关键损坏；`partial` 表示主要缺失被补回，但图标命名偏泛、
表格/文本保持有轻微问题；`wrong` 表示关键动作图标错、缺图标未补回，或普通文本被模型
幻觉加入图标。这个评判标准故意不只数 `[Icon: ...]`，因为 patching 的目标是把正确的
视觉信息放回正确的文本位置。

= 并发参数：`concurrency=10`

并发扫描固定 `DPI=250`、`batch_size=15`，只改变同时发往 SGLang 的请求数。
结果显示 `concurrency=1` 到 `10` 吞吐明显上升；`10-15` 进入平台区；`14`
出现回落；`16` 在该轮实验中没有形成有效运行。因此选择 `10`，而不是继续追求更高并发。
这个选择保留了平台区吞吐，同时减少排队波动和服务端压力。

#figure(
  image("assets/chart-concurrency-throughput.svg", width: 92%),
  caption: [`concurrency` 扫描。`10` 已进入高吞吐平台区，继续加大并发收益很小且稳定性下降。],
)

#table(
  columns: 5,
  inset: 4pt,
  [设置], [吞吐], [相对 `c=1`], [GPU util avg/max], [结论],
  [`c=1`], [`2.76 req/s`], [`1.00x`], [`62.3% / 100%`], [并发不足],
  [`c=10`], [`4.57 req/s`], [`1.65x`], [`69.2% / 100%`], [推荐点],
  [`c=12`], [`4.57 req/s`], [`1.65x`], [`67.9% / 100%`], [几乎无增益],
  [`c=14`], [`4.11 req/s`], [`1.49x`], [`68.7% / 100%`], [回落],
)

= Batch 参数：`batch_size=140`

`batch_size` 控制一次调度中打包多少个 patching 字段。它不等于 VLM 的 context window，
而是影响请求轮次和调度开销。固定 `DPI=250`、`concurrency=10` 后，batch 太小会
造成大量请求轮次，例如 `b=3` 需要 `329` 个 LLM batches；batch 增大后吞吐持续改善。
`b=140` 达到本轮最高 `5.35 req/s`，而 `b=200` 已下降到 `5.00 req/s`，说明最大值已经越过。

#figure(
  image("assets/chart-batch-throughput.svg", width: 92%),
  caption: [`batch_size` 扫描。`140` 是本轮最高点，`200` 已低于 `140`，因此默认不继续放大。],
)

#block(breakable: false)[
  #table(
    columns: 5,
    inset: 4pt,
    [设置], [LLM batches], [吞吐], [耗时], [结论],
    [`b=3`], [`329`], [`3.17 req/s`], [`311.33 s`], [轮次过多],
    [`b=48`], [`21`], [`4.95 req/s`], [`199.41 s`], [接近平台区],
    [`b=140`], [`8`], [`5.35 req/s`], [`184.52 s`], [推荐点],
    [`b=200`], [`5`], [`5.00 req/s`], [`197.44 s`], [过大后下降],
  )
]

= DPI 参数：`DPI=250`

吞吐上 `200 DPI` 最快，但 patching 的核心目标不是最快，而是把 MinerU 丢掉的小 UI 图标
补回文本中。因此 DPI 需要单独做质量实验。三页质量实验的结论是：`250 DPI` 在真正
缺图标字段上没有 hard wrong；`200 DPI` 更容易错读关键图标；`300 DPI` 在普通文本和表格保持上
更稳，但仍会把两个关键动作图标识别错。

#figure(
  image("assets/chart-dpi-recognition-ratio.svg", width: 84%),
  caption: [三页参考集人工判读。`250 DPI` 在 `8` 个真正缺图标字段上为 `6 correct / 2 partial / 0 wrong`。],
)

#table(
  columns: 7,
  inset: 3.5pt,
  [DPI], [全字段 correct/partial/wrong], [缺图标字段 correct/partial/wrong], [吞吐], [耗时], [峰值显存], [解释],
  [`200`], [`15 / 1 / 9`], [`5 / 1 / 2`], [`2.82 req/s`], [`8.87 s`], [`25.7 GiB`], [最快，但错图标和误报更多],
  [`250`], [`17 / 2 / 6`], [`6 / 2 / 0`], [`2.76 req/s`], [`9.04 s`], [`26.2 GiB`], [默认最稳：缺图标字段没有 hard wrong],
  [`300`], [`18 / 0 / 7`], [`6 / 0 / 2`], [`2.50 req/s`], [`10.00 s`], [`26.8 GiB`], [表格/普通文本保持更好，但两个关键图标错],
)

#text(size: 7.9pt)[
  #table(
    columns: (0.75fr, 1.25fr, 1.25fr, 0.95fr, 0.95fr, 0.95fr, 1.35fr),
    inset: 2.5pt,
    [字段], [原始缺口], [人工参考], [`200 DPI`], [`250 DPI`], [`300 DPI`], [结论],
    [`320/4434`], [`Step 8 Click , ... Attachment`], [`Icon: attachment/link`], [refresh #linebreak() #text(fill: red)[wrong]], [attachment #linebreak() #text(fill: green)[correct]], [refresh #linebreak() #text(fill: red)[wrong]], [`250` 正确区分附件图标。],
    [`313/4319`], [`Click at the upper right...`], [`Icon: upload/resume`], [menu #linebreak() #text(fill: red)[wrong]], [upload #linebreak() #text(fill: green)[correct]], [menu #linebreak() #text(fill: red)[wrong]], [`250` 避免把上传动作误读成菜单。],
    [`313/4316`], [列表中 5 个操作图标缺失。], [face group, playback, filter, vehicle group, delete], [5/5 语义正确 #linebreak() #text(fill: green)[correct]], [2 个 group 标签偏泛 #linebreak() #text(fill: orange)[partial]], [5/5 语义正确 #linebreak() #text(fill: green)[correct]], [`250` 不是每例最强，但没有 hard wrong。],
    [`435/6210`], [表格图标退化为 `×`、`○`。], [close-all, split-screen, snapshot, stop/pause, frame-by-frame], [补回但改写表格 #linebreak() #text(fill: orange)[partial]], [补回但标签偏泛 #linebreak() #text(fill: orange)[partial]], [表格保持最好 #linebreak() #text(fill: green)[correct]], [`300` 更适合表格复查。],
  )
]

#pagebreak()

失败案例显示，DPI 不是唯一误差来源。典型错误分三类：

#block(breakable: false)[
  #table(
    columns: (1.1fr, 1.8fr, 1.8fr),
    inset: 3.5pt,
    [类型], [例子], [含义],
    [普通符号被当成图标], [`313/4312` bullet list、`313/4320` diamond list], [所有 DPI 都会把列表符号当成 `[Icon: bullet]` 或 `[Icon: diamond]`，这更像候选过滤/提示词问题，不是分辨率问题。],
    [OCR 乱码诱发幻觉], [`313/4318` broken `offline`、`320/4435` corrupted `.flv`], [模型会把损坏字符解释成图标。该类错误不能靠单纯提高 DPI 解决，需要后续用文本类型或字符异常规则抑制。],
    [细粒度动作图标混淆], [`313/4319` upload 被识别成 menu、`320/4434` attachment 被识别成 refresh], [这是 DPI 选择的关键证据：`250 DPI` 在这两个动作图标上正确，而 `200/300` 出现 hard wrong。],
    [表格结构保持差异], [`435/6210` operation table], [`300 DPI` 最能保持表格文本，但 `250 DPI` 仍能补回主要图标且没有关键动作图标 hard wrong，因此默认选择 `250`，表格复查可临时使用 `300`。],
  )
]

#pagebreak()

= 工程参数

下面这些参数没有像 `concurrency`、`batch_size`、`DPI` 那样做完整曲线扫描，因为它们主要
受可靠性、输出完整性和系统余量约束。但它们同样是最终配置的一部分。

#block(breakable: false)[
  #table(
    columns: 4,
    inset: 4pt,
    [参数], [推荐值], [风险如果过小], [风险如果过大 / 选择理由],
    [`SGLang mem_fraction_static`], [`0.75`], [KV cache 余量偏小，长上下文/并发更容易失败], [过大会挤压驱动、图像处理和系统余量；`0.75` 在 5090 上稳定],
    [`max_new_tokens`], [`8000`], [长表格、长列表或多图标字段可能被截断], [过大会增加异常长输出风险；`8000` 对 patching 已足够宽松],
    [`checkpoint_interval`], [`30`], [任务失败后重跑成本高], [过小会增加 IO；`30` 相对 VLM 推理开销很低],
    [`page_window_size`], [`200`], [太小可能切断跨页表格/续表证据], [太大接近整本 rasterize，CPU/内存和临时文件成本上升],
    [`request_timeout`], [`120 s`], [复杂表格请求容易超时], [过大只会延迟失败暴露；`120 s` 覆盖本实验正常请求],
    [`invalid_retry_limit`], [`0`], [不会重复追问模型修正格式], [当前策略直接采纳或 fallback，减少慢速重试和重复幻觉],
  )
]

= 环境记录

本轮实验的机器和服务身份如下。精确 Python package versions 在当时的 `machine.txt` 中未归档，
因此本文只记录已捕获的系统、模型和服务启动参数；两个 benchmark 脚本已补充 package version
采集，后续重跑会自动记录 `python`、`sglang`、`torch`、`nvidia-cudnn-cu12` 等版本。

#text(size: 8.2pt)[
  #table(
    columns: 2,
    inset: 3.5pt,
    [吞吐实验时间], [`2026-05-03T22:07:10+08:00`],
    [DPI 质量实验时间], [`2026-05-04T01:35:35+08:00`],
    [容器主机], [`autodl-container-jb1uyxxe0t-85d2ca45`],
    [系统], [`Ubuntu 22.04.5 LTS`, kernel `5.15.0-78-generic`],
    [CPU / 内存], [`Intel Xeon Platinum 8470Q`, `90G` RAM],
    [GPU / Driver], [`NVIDIA GeForce RTX 5090`, driver `580.105.08`, `32607 MiB` VRAM],
    [模型快照], [`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, snapshot `d1fef185160f938fca00c3c664f21250dd544d63`],
    [SGLang 启动参数], [`tp=1`, `mem_fraction_static=0.75`, `context=100000`, `quantization=moe_wna16`, `attention=triton`, `kv_cache=fp8_e5m2`],
  )
]

= 结论

本实验推荐的 patching 默认配置是：

#block(breakable: false)[
```env
RAG_FLOW_SGLANG_MEM_FRACTION_STATIC=0.75
RAG_FLOW_PATCH_DPI=250
RAG_FLOW_PATCH_CONCURRENCY=10
RAG_FLOW_PATCH_BATCH_SIZE=140
RAG_FLOW_PATCH_CHECKPOINT_INTERVAL=30
RAG_FLOW_PATCH_MAX_NEW_TOKENS=8000
RAG_FLOW_PATCH_PAGE_WINDOW_SIZE=200
RAG_FLOW_PATCH_LLM_TIMEOUT=120
RAG_FLOW_PATCH_INVALID_RETRY_LIMIT=0
```
]

这套配置的逻辑是：`concurrency=10` 取吞吐平台区的稳点，`batch_size=140` 取吞吐峰值前后
最可靠的位置，`DPI=250` 取小图标补全质量的稳点，其余参数保证长输出、长任务和服务器显存
有足够余量。换句话说，它不是单个指标的最高点，而是当前文档、当前 VLM 和当前服务器下
最适合作为默认值的一组参数。

#pagebreak()

= 可复现产物

#table(
  columns: 2,
  inset: 4pt,
  [100 页吞吐数据], [`experiments/patching-parameter-benchmark/clean-results-100p.csv`],
  [三页 DPI 重跑目录], [`experiments/patching-parameter-benchmark/data-icon-pages-redo/`],
  [DPI 质量比例], [`data-icon-pages-redo/recognition-ratio.csv`],
  [完整字段对照], [`data-icon-pages-redo/recognition-comparison-full.csv`],
  [100 页运行脚本], [`experiments/patching-parameter-benchmark/run_remote_benchmark_100p.sh`],
  [三页 DPI 脚本], [`experiments/patching-parameter-benchmark/run_remote_dpi_icon_pages.sh`],
)
