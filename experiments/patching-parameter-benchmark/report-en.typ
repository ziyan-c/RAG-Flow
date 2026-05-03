#set document(title: "RAG-Flow Patching Parameter Selection Study")
#set page(margin: 0.70in)
#set text(size: 9.35pt)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[RAG-Flow Patching Parameter Selection Study]
  #linebreak()
  #text(size: 10pt)[Small-Icon Patching with an SGLang VLM Backend]
]

= Abstract

This report selects production defaults for the RAG-Flow patching stage. It is
not only a DPI comparison. The study uses a 100-page throughput benchmark to
choose `concurrency` and `batch_size`, and a three-page icon-heavy quality
benchmark to choose `DPI`. The remaining parameters, including
`max_new_tokens`, `checkpoint_interval`, `page_window_size`, and SGLang
`mem_fraction_static`, are selected from reliability constraints and observed
resource behavior.

The recommended configuration is:

#table(
  columns: 4,
  inset: 4.2pt,
  [Parameter], [Recommended value], [Main objective], [Basis],
  [`RAG_FLOW_PATCH_DPI`], [`250`], [small-icon recovery quality], [`6/8 correct`, `2/8 partial`, `0/8 wrong` on icon-required fields],
  [`RAG_FLOW_PATCH_CONCURRENCY`], [`10`], [request concurrency], [`c=10-15` reaches the throughput plateau; `10` is the stable point],
  [`RAG_FLOW_PATCH_BATCH_SIZE`], [`140`], [reduce request rounds], [`b=140` is the observed peak; `b=200` already drops],
  [`RAG_FLOW_SGLANG_MEM_FRACTION_STATIC`], [`0.75`], [VRAM headroom], [stable SGLang pre-allocation with room for driver and image-input overhead],
  [`RAG_FLOW_PATCH_MAX_NEW_TOKENS`], [`8000`], [avoid output truncation], [large enough for long tables/lists while staying far below the server context],
  [`RAG_FLOW_PATCH_CHECKPOINT_INTERVAL`], [`30`], [failure recovery], [reduces rerun cost; IO overhead is small relative to VLM inference],
  [`RAG_FLOW_PATCH_PAGE_WINDOW_SIZE`], [`200`], [PDF rendering bound], [avoids whole-document rasterization while preserving cross-page evidence],
)

= Experimental Setup

The throughput benchmark uses a continuous 100-page subset of the target
technical manual: `page_idx=50-149`. It contains `1394` content blocks and
`987` fields submitted to the VLM. This sample is large enough to reveal
concurrency and batch-size plateaus and decline points.

The quality benchmark uses three icon-heavy reference pages from the same
manual: `313`, `320`, and `435`. They produce `25` submitted VLM fields, of
which `8` are manually confirmed to contain missing UI icons. Quality is not
scored by counting `[Icon: ...]` markers. Each output is reviewed against the
PDF crop, MinerU OCR text, and model output. `correct` means that icon meaning
and insertion location are basically right, or a no-icon field is preserved.
`partial` means the main information is recovered but labels or text/table
preservation are weaker. `wrong` means key icons are missed or mislabeled, or
ordinary text receives hallucinated icons.

#table(
  columns: 2,
  inset: 5pt,
  [GPU], [`NVIDIA GeForce RTX 5090`, 32607 MiB VRAM],
  [Model server], [`SGLang` OpenAI-compatible API],
  [VLM], [`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`],
  [Context length], [`100000` tokens],
  [100-page throughput sample], [`987` VLM requests],
  [Three-page quality sample], [`25` submitted fields; `8` icon-required fields],
)

#pagebreak()

= Metrics and Review Criteria

For reproducibility, all runs use the same MinerU `content_list`, the same
patching candidate generation logic, the same SGLang server, and the same VLM.
Each sweep changes only the target parameter while holding the other conditions
fixed.

#block(breakable: false)[
  #table(
    columns: (0.95fr, 1.75fr, 1.25fr, 1.25fr),
    inset: 3.8pt,
    [Experiment], [Swept values], [Fixed conditions], [Selection target],
    [`concurrency`], [`1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16`], [`DPI=250`, `batch_size=15`], [find the throughput plateau and failure point],
    [`batch_size`], [`3, 6, 9, 12, 15, 18, 24, 30, 36, 48, 60, 80, 100, 140, 200`], [`DPI=250`, `concurrency=10`], [find the peak before batching overhead reverses],
    [`DPI` throughput], [`200, 250, 300`], [`concurrency=10`, `batch_size=140`], [estimate speed and resource cost],
    [`DPI` quality], [`200, 250, 300`], [reference pages `313, 320, 435`, `25` fields], [manual review of icon recovery and false positives],
  )
]

Throughput is computed as `requests_per_sec = requests_submitted / elapsed_sec`.
`requests_submitted` is the number of patching fields actually sent to the VLM,
not the page count and not the number of icons. `llm_batches` is the number of
scheduling rounds; `GPU util avg/max`, memory, and power come from runtime
sampling. `status=0` means a complete run. Failed runs remain in the data but
are not eligible as recommended points.

Quality is also scored per submitted field. `all_submitted_fields` tests whether
ordinary text is preserved without hallucinated icons. `icon_required_fields`
contains only fields manually confirmed to have missing UI icons in the source
OCR. `correct` means that icon semantics and insertion location are basically
right and the original text is not materially damaged. `partial` means that the
main missing information is recovered, but icon labels are generic or text/table
preservation is imperfect. `wrong` means a key action icon is wrong or missing,
or ordinary text receives hallucinated icons. The score intentionally does not
only count `[Icon: ...]` markers, because patching needs the right visual
information in the right textual position.

= Concurrency: `concurrency=10`

The concurrency sweep fixes `DPI=250` and `batch_size=15`, then varies the
number of simultaneous requests sent to SGLang. Throughput rises clearly from
`concurrency=1` to `10`; `10-15` forms a plateau; `14` declines; and `16` does
not produce a valid run in this benchmark. The default is therefore `10`, not
the largest possible value. It keeps plateau-level throughput while reducing
queueing variance and server pressure.

#figure(
  image("assets/chart-concurrency-throughput.svg", width: 92%),
  caption: [`concurrency` sweep. `10` is already in the high-throughput plateau; higher concurrency adds little and is less stable.],
)

#table(
  columns: 5,
  inset: 4pt,
  [Setting], [Throughput], [Relative to `c=1`], [GPU util avg/max], [Decision],
  [`c=1`], [`2.76 req/s`], [`1.00x`], [`62.3% / 100%`], [under-concurrent],
  [`c=10`], [`4.57 req/s`], [`1.65x`], [`69.2% / 100%`], [recommended],
  [`c=12`], [`4.57 req/s`], [`1.65x`], [`67.9% / 100%`], [no meaningful gain],
  [`c=14`], [`4.11 req/s`], [`1.49x`], [`68.7% / 100%`], [declines],
)

= Batch Size: `batch_size=140`

`batch_size` controls how many patching fields are grouped into one scheduling
round. It is not the VLM context window; it mainly changes request rounds and
scheduling overhead. With `DPI=250` and `concurrency=10` fixed, very small
batches create too many LLM batches: `b=3` requires `329` batches. Throughput
improves as batch size grows, peaks at `b=140`, and then falls at `b=200`.

#figure(
  image("assets/chart-batch-throughput.svg", width: 92%),
  caption: [`batch_size` sweep. `140` is the observed peak, while `200` is already slower.],
)

#block(breakable: false)[
  #table(
    columns: 5,
    inset: 4pt,
    [Setting], [LLM batches], [Throughput], [Elapsed], [Decision],
    [`b=3`], [`329`], [`3.17 req/s`], [`311.33 s`], [too many rounds],
    [`b=48`], [`21`], [`4.95 req/s`], [`199.41 s`], [near plateau],
    [`b=140`], [`8`], [`5.35 req/s`], [`184.52 s`], [recommended],
    [`b=200`], [`5`], [`5.00 req/s`], [`197.44 s`], [too large],
  )
]

= DPI: `DPI=250`

Throughput alone would choose `200 DPI`, but patching is primarily intended to
recover small UI icons dropped by MinerU. DPI must therefore be selected by
quality. On the three-page quality benchmark, `250 DPI` is the only setting
with no hard errors on icon-required fields. `200 DPI` is faster but misreads
important icons. `300 DPI` preserves some ordinary text and table structure
better, but still mislabels two key action icons.

#figure(
  image("assets/chart-dpi-recognition-ratio.svg", width: 84%),
  caption: [Manual review on the three-page reference set. `250 DPI` gives `6 correct / 2 partial / 0 wrong` on the `8` icon-required fields.],
)

#table(
  columns: 7,
  inset: 3.5pt,
  [DPI], [All fields correct/partial/wrong], [Icon fields correct/partial/wrong], [Throughput], [Elapsed], [Peak VRAM], [Interpretation],
  [`200`], [`15 / 1 / 9`], [`5 / 1 / 2`], [`2.82 req/s`], [`8.87 s`], [`25.7 GiB`], [fastest, but more hallucinated or wrong icons],
  [`250`], [`17 / 2 / 6`], [`6 / 2 / 0`], [`2.76 req/s`], [`9.04 s`], [`26.2 GiB`], [best default: no hard errors on icon-required fields],
  [`300`], [`18 / 0 / 7`], [`6 / 0 / 2`], [`2.50 req/s`], [`10.00 s`], [`26.8 GiB`], [better text/table preservation, but two key icon errors],
)

#text(size: 7.2pt)[
  #table(
    columns: (0.75fr, 1.25fr, 1.25fr, 0.95fr, 0.95fr, 0.95fr, 1.35fr),
    inset: 2.5pt,
    [Field], [Original gap], [Manual reference], [`200 DPI`], [`250 DPI`], [`300 DPI`], [Takeaway],
    [`320/4434`], [`Step 8 Click , ... Attachment`], [`Icon: attachment/link`], [refresh #linebreak() #text(fill: red)[wrong]], [attachment #linebreak() #text(fill: green)[correct]], [refresh #linebreak() #text(fill: red)[wrong]], [`250` correctly distinguishes attachment.],
    [`313/4319`], [`Click at the upper right...`], [`Icon: upload/resume`], [menu #linebreak() #text(fill: red)[wrong]], [upload #linebreak() #text(fill: green)[correct]], [menu #linebreak() #text(fill: red)[wrong]], [`250` avoids menu confusion.],
    [`313/4316`], [Five operation icons missing in a list.], [face group, playback, filter, vehicle group, delete], [5/5 semantic match #linebreak() #text(fill: green)[correct]], [two group labels are generic #linebreak() #text(fill: orange)[partial]], [5/5 semantic match #linebreak() #text(fill: green)[correct]], [`250` is not strongest on every case, but has no hard error.],
    [`435/6210`], [Table icons collapse into `×` and `○`.], [close-all, split-screen, snapshot, stop/pause, frame-by-frame], [rewrites table #linebreak() #text(fill: orange)[partial]], [generic labels #linebreak() #text(fill: orange)[partial]], [best table text #linebreak() #text(fill: green)[correct]], [`300` is useful for table review.],
  )
]

#pagebreak()

The failure cases show that DPI is not the only error source. The main patterns
are:

#block(breakable: false)[
  #table(
    columns: (1.1fr, 1.8fr, 1.8fr),
    inset: 3.5pt,
    [Pattern], [Examples], [Implication],
    [Plain markers become icons], [`313/4312` bullet list, `313/4320` diamond list], [All DPI settings turn list markers into `[Icon: bullet]` or `[Icon: diamond]`. This is more likely a candidate-filtering or prompting issue than a resolution issue.],
    [OCR noise triggers hallucination], [`313/4318` broken `offline`, `320/4435` corrupted `.flv`], [The model interprets corrupted characters as icons. Raising DPI alone does not fix this; future filtering can use text type and character-noise guards.],
    [Fine action-icon confusion], [`313/4319` upload becomes menu, `320/4434` attachment becomes refresh], [This is the strongest DPI-selection evidence: `250 DPI` is correct on both action icons, while `200/300` produce hard errors.],
    [Table-structure fidelity differs], [`435/6210` operation table], [`300 DPI` best preserves table text, but `250 DPI` still recovers the main icons and avoids hard action-icon errors; therefore `250` is the default, while `300` can be used for table review.],
  )
]

#pagebreak()

= Engineering Parameters

The following parameters are not swept as full curves because they are governed
mainly by reliability, output completeness, and system headroom. They are still
part of the final default configuration.

#block(breakable: false)[
  #table(
    columns: 4,
    inset: 4pt,
    [Parameter], [Recommended value], [Risk if too small], [Risk if too large / reason],
    [`SGLang mem_fraction_static`], [`0.75`], [too little KV-cache headroom for long context or concurrency], [too high leaves little room for driver, image processing, and runtime spikes],
    [`max_new_tokens`], [`8000`], [long tables/lists or multi-icon fields may be truncated], [very large values increase runaway-output risk; `8000` is already generous],
    [`checkpoint_interval`], [`30`], [failures require more rerun work], [too small increases IO; `30` is cheap relative to VLM inference],
    [`page_window_size`], [`200`], [cross-page table evidence may be cut], [too large approaches whole-document rasterization and raises CPU/memory/temp-file cost],
    [`request_timeout`], [`120 s`], [complex table requests may time out], [too large delays failure visibility; `120 s` covers normal requests here],
    [`invalid_retry_limit`], [`0`], [no repeated formatting repair attempts], [keeps benchmark and production fast; fallback accepts usable text without slow retries],
  )
]

= Environment Record

The machine and serving identity captured for this run is listed below. Exact
Python package versions were not archived in the original `machine.txt`, so this
report only records the captured system, model, and serving parameters. The two
benchmark scripts now also capture package versions for future reruns, including
`python`, `sglang`, `torch`, and `nvidia-cudnn-cu12`.

#text(size: 8.2pt)[
  #table(
    columns: 2,
    inset: 3.5pt,
    [Throughput run time], [`2026-05-03T22:07:10+08:00`],
    [DPI quality run time], [`2026-05-04T01:35:35+08:00`],
    [Container host], [`autodl-container-jb1uyxxe0t-85d2ca45`],
    [System], [`Ubuntu 22.04.5 LTS`, kernel `5.15.0-78-generic`],
    [CPU / memory], [`Intel Xeon Platinum 8470Q`, `90G` RAM],
    [GPU / driver], [`NVIDIA GeForce RTX 5090`, driver `580.105.08`, `32607 MiB` VRAM],
    [Model snapshot], [`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, snapshot `d1fef185160f938fca00c3c664f21250dd544d63`],
    [SGLang serving args], [`tp=1`, `mem_fraction_static=0.75`, `context=100000`, `quantization=moe_wna16`, `attention=triton`, `kv_cache=fp8_e5m2`],
  )
]

= Conclusion

The recommended patching defaults are:

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

The logic is: `concurrency=10` chooses the stable point in the throughput
plateau; `batch_size=140` chooses the observed batch-size peak before decline;
`DPI=250` chooses the strongest default for icon recovery quality; and the
remaining parameters preserve output completeness, recoverability, and GPU
headroom. This is not the maximum of one metric. It is the best default set for
the current document, VLM backend, and server.

#pagebreak()

= Reproducibility

#table(
  columns: 2,
  inset: 4pt,
  [100-page throughput data], [`experiments/patching-parameter-benchmark/clean-results-100p.csv`],
  [Three-page DPI rerun], [`experiments/patching-parameter-benchmark/data-icon-pages-redo/`],
  [DPI quality ratios], [`data-icon-pages-redo/recognition-ratio.csv`],
  [Complete field comparison], [`data-icon-pages-redo/recognition-comparison-full.csv`],
  [100-page runner], [`experiments/patching-parameter-benchmark/run_remote_benchmark_100p.sh`],
  [Three-page DPI runner], [`experiments/patching-parameter-benchmark/run_remote_dpi_icon_pages.sh`],
)
