# RAG Flow

RAG Flow is a multimodal retrieval augmented generation pipeline for technical
manuals. The default preset uses generic example names, and paths, model names,
and collection names can be changed through environment variables.

## What It Does

1. Parse the source PDF with MinerU into structured `content_list.json`.
2. Recover PDF outline sections into MinerU block metadata when outline data exists.
3. Patch MinerU output with a vision language model:
   - recover small icon text that MinerU/OCR missed
   - add context-aware descriptions to extracted images
4. Build section-aware or fixed token-window chunks from enriched `content_list.json`.
5. Store three retrieval signals in Qdrant:
   - dense text vectors
   - sparse BM25 vectors
   - ColPali page-image multivectors
6. Serve a FastAPI `/retrieve` endpoint with RRF fusion.
7. Use an OpenAI-compatible LLM endpoint for cited terminal chat.

## Project Layout

```text
src/rag_flow/
  config.py                 Environment-driven configuration
  mineru.py                 MinerU install/check/run helpers
  pipeline.py               End-to-end ingestion orchestration
  chunking.py               MinerU JSON to retrieval chunks
  indexing.py               Qdrant collection, text vectors, visual vectors
  retrieval.py              Hybrid retrieval engine and context builder
  api.py                    FastAPI retrieval service
  chat_cli.py               Terminal RAG chat client
  preprocessing/            MinerU post-processing helpers
scripts/                    Shell wrappers for common operations
envs/                       Exported conda environments from the old workspace
docs/                       Notes and migration docs
configs/                    Per-manual env templates
```

## Setup

```bash
mkdir -p .local
cp .env.example .local/rag-flow.env
pip install -e ".[text-retrieval]"
```

The app automatically loads `.local/rag-flow.env` when commands are run from
this repository. You can still point to another local env file with
`RAG_FLOW_ENV_FILE=/path/to/file`.

This core repository still owns its own `.local` private configuration for
pipeline stages, source documents, model paths, and local experiments. The
separate `rag-flow-orchestrator` repository has its own `.local` for deployment
composition and service runtime settings.

Use `pip install -e ".[retrieval,preprocess]"` for the full local visual
retrieval and preprocessing stack. The lighter `text-retrieval` extra installs
the default online text retrieval path without Torch, ColPali, or BitsAndBytes.

For the original AutoDL-style environment, the exported conda YAML files are in
`envs/`. They are intentionally preserved because CUDA, ColPali, MinerU, and
SGLang package compatibility is sensitive.

## Command Line

All operations are available through the unified `rag-flow` command:

```bash
rag-flow init china-all
rag-flow env create-mineru
rag-flow env create-pipeline
rag-flow env create-llm
rag-flow mineru doctor
rag-flow mineru run
rag-flow section
rag-flow serve llm-sglang
rag-flow patch --artifact-dir /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto
rag-flow ingest --to-stage chunking
rag-flow caption
rag-flow chunk
rag-flow index text
rag-flow retriever
rag-flow chat
```

Use `--dry-run` on script-backed commands to see what would run:

```bash
rag-flow init china-all --dry-run
rag-flow env create-mineru --dry-run
rag-flow env create-pipeline --dry-run
rag-flow serve llm-sglang --dry-run
```

## Pipeline

Check whether MinerU is available:

```bash
rag-flow mineru doctor
```

Install the pinned MinerU package into the configured Python environment.
The default install spec is `mineru[all]==3.0.9`, and the configured Python
should be 3.10 through 3.13:

```bash
rag-flow mineru setup
```

Run the default ingestion path from PDF to retrieval chunks:

```bash
rag-flow ingest --pdf .local/source-documents/example-technical-manual.pdf
```

When `--pdf` is omitted, ingestion uses `RAG_FLOW_MINERU_INPUT_PATH`.
`rag-flow mineru run` also uses that same input path and writes to
`RAG_FLOW_MINERU_OUTPUT_DIR`. `rag-flow mineru run` accepts either one PDF or
a folder of PDFs. Folder input is recursive by default and mirrors the input
folder layout into the output root before letting MinerU create each
file-stem output folder:

```bash
rag-flow mineru run \
  --input .local/source-documents \
  --output-dir /root/autodl-tmp/manuals/public
```

For example, `.local/source-documents/network/admin.pdf` is parsed with
`-o /root/autodl-tmp/manuals/public/network`, so MinerU can write its usual
`admin/auto/...` files under that mirrored location. Add `--no-recursive` when
you only want PDFs directly inside the input folder.

Preview the command sequence without running heavy steps:

```bash
rag-flow ingest --dry-run
```

Resume from a later stage:

```bash
rag-flow ingest --from-stage captioning --to-stage chunking
```

Run through indexing too:

```bash
rag-flow ingest --to-stage indexing
```

The individual steps are also available when you need manual control.
If MinerU writes files into a nested output folder, `rag-flow ingest` searches
`RAG_FLOW_MINERU_OUTPUT_DIR` for a `*content_list.json` matching the current
source PDF name, then derives the downstream artifact paths from that folder:
`*_content_list_SECTIONED.json`,
`*_content_list_SECTIONED_PATCHED.json`,
`*_content_list_SECTIONED_PATCHED_CAPTIONED.json`, and
`*_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json`.
Set `RAG_FLOW_MINERU_BACKEND=pipeline` for CPU-friendly parsing. Set
`RAG_FLOW_MINERU_MODEL_SOURCE=modelscope` to run MinerU with
`MINERU_MODEL_SOURCE=modelscope`.

Run MinerU only:

```bash
rag-flow mineru run
```

Recover PDF outline sections into MinerU JSON:

```bash
rag-flow section \
  --input-json /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_content_list.json \
  --input-pdf .local/source-documents/example-technical-manual.pdf
```

This writes `*_content_list_SECTIONED.json` plus `*_SECTIONING_AUDIT.json`.
It uses the original source PDF outline/bookmarks, not MinerU heading labels,
and only adds `section_*` metadata fields to blocks.
Inside `rag-flow ingest`, this stage runs automatically after MinerU parsing and
before patching.

Patch small icons:

```bash
rag-flow serve llm-sglang
rag-flow patch --artifact-dir /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto
```

The artifact-dir form is the preferred patching entrypoint after MinerU has
parsed a PDF. It expects a MinerU output folder containing
`*_content_list.json` and `*_origin.pdf`. If a
`*_content_list_SECTIONED.json` file is present, patching uses that sectioned
file and writes `*_content_list_SECTIONED_PATCHED.json`; otherwise it patches
the raw MinerU file and writes `*_content_list_PATCHED.json`. Patching sends
its crop images to the local OpenAI-compatible vision LLM configured by
`RAG_FLOW_LLM_BASE_URL` and `RAG_FLOW_LLM_MODEL`; start it first with
`rag-flow serve llm-sglang`. If that service is not reachable, patching fails
before rendering PDF pages.

Patching focuses on content blocks instead of page furniture: text, lists, and
table bodies/footnotes are patched, while table captions, headers, footers, page
numbers, and empty fields are skipped. Small uncaptioned `image` blocks are
treated as possible inline icons:
patching links them to nearby text/list blocks or containing table cells, expands
the visual crop to include them, marks them in the JSON, and keeps captioning
from describing them as standalone figures. MinerU represents cross-page table
continuations as empty `table` blocks; those blocks are not copied into the JSON
as duplicate text. Instead, their PDF crops are stacked onto the previous table
crop so the LLM can patch the single complete `table_body` with visual evidence
from every page of the same table.

The source PDF is rendered in page windows instead of loading the whole book at
once; the default is 200 pages per window. When `--artifact-dir` points at a
parent folder, patching finds nested MinerU artifact folders recursively and
processes one PDF at a time to avoid GPU memory spikes:

```bash
rag-flow patch \
  --artifact-dir /root/autodl-tmp/manuals/public \
  --page-window-size 200 \
  --batch-size 9 \
  --concurrency 3
```

The LLM prompt asks the model to add missing `[Icon: ...]` markers without
explanations or surrounding commentary. The run writes a checkpoint every 10
LLM batches by default, also checkpoints at the end of each page window, resumes
from that checkpoint on retry, deletes the checkpoint after success, writes a
`*_PATCHING_VIEW.pdf` overlay that shows the exact crop regions sent to the LLM,
and prints patching statistics at the end. Useful controls:

- `--batch-size`: LLM request group size for checkpoints, default `9`
- `--concurrency`: maximum simultaneous patching LLM requests, default `3`
- `--max-new-tokens`: generation budget, default `8000`
- `--llm-base-url`: OpenAI-compatible LLM endpoint, default `RAG_FLOW_LLM_BASE_URL`
- `--model` / `--llm-model`: model name sent to the LLM endpoint
- `--request-timeout`: per-request timeout, default `RAG_FLOW_PATCH_LLM_TIMEOUT`
- `--dpi`: PDF render DPI for patching crops, default `250`
- `--page-window-size`: PDF render window size, default `200`
- `--checkpoint-interval`: write checkpoint every N LLM batches, default `30`
- `--invalid-retry-limit`: retry only-icon LLM outputs before fallback insertion, default `0`
- `--patching-view-pdf`: custom path for the overlay PDF
- `--no-patching-view`: skip writing the overlay PDF
- `--no-resume`: ignore an existing checkpoint
- `--no-recursive`: only patch the exact artifact folder

Patching needs the pipeline Python environment and Poppler PDF rendering
commands (`pdfinfo` and `pdftoppm`). `rag-flow env create-pipeline` installs the
Python packages, writes `RAG_FLOW_PIPELINE_PYTHON_BIN`, and later `rag-flow
patch` automatically re-runs itself with that Python when the variable is
available.
`rag-flow init china-all` installs `poppler-utils` by default on apt-based
Linux machines.

The same `--max-new-tokens` override is available on `rag-flow ingest` when
running through the patching stage.

You can also regenerate the overlay without running the LLM:

```bash
rag-flow patch-view \
  --input-json /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_content_list_SECTIONED_PATCHED.json \
  --input-pdf /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_origin.pdf
```

Generate image descriptions:

```bash
rag-flow caption
```

After patching a MinerU artifact folder, you can also use the artifact-dir form:

```bash
rag-flow caption --artifact-dir /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto
```

Captioning calls the local OpenAI-compatible SGLang service configured by
`RAG_FLOW_LLM_BASE_URL` / `RAG_FLOW_LLM_MODEL`, so start it first with
`rag-flow serve llm-sglang`. It resumes from
`*_PATCHED_CAPTIONED.checkpoint.json` when available, writes a checkpoint after
each LLM batch by default, and skips image blocks that already have
`image_description_vlm`. Captioning context is taken from nearby text blocks
before and after each image in the patched JSON order, rather than from a fixed
page window. A successful run also writes a `*_CAPTIONING_VIEW.pdf` overlay
showing each captioned image and the nearby context blocks used for it. Useful
controls:

- `--dry-run`: print resolved paths, image counts, and estimated context token stats without calling the LLM
- `--max-context-tokens`: cap nearby text sent with each image, default `10000`
- `--max-new-tokens`: caption generation budget, default `8000`
- `--batch-size`: images grouped per local batch, default `4`
- `--concurrency`: maximum simultaneous captioning LLM requests inside each batch, default `1`
- `--llm-base-url`: OpenAI-compatible endpoint, default `RAG_FLOW_LLM_BASE_URL`
- `--request-timeout`: captioning request timeout in seconds, default `120`
- `--checkpoint-interval`: write checkpoint every N LLM batches, default `1`
- `--captioning-view-pdf`: custom path for the overlay PDF
- `--no-captioning-view`: skip writing the overlay PDF
- `--no-resume`: ignore an existing checkpoint
- `--no-skip-existing`: regenerate descriptions even when `image_description_vlm` exists

You can regenerate the captioning overlay without calling the LLM:

```bash
rag-flow caption-view --artifact-dir /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto
```

Or pass the captioning input JSON and origin PDF explicitly:

```bash
rag-flow caption-view \
  --input-json /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_content_list_SECTIONED_PATCHED.json \
  --input-pdf /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_origin.pdf
```

Build retrieval chunks:

```bash
rag-flow chunk
```

The chunk JSON contains `chunk_content` plus `metadata`; `chunk_content` is the
text used for embeddings and retrieved context.

By default `rag-flow chunk` uses `RAG_FLOW_CHUNK_MODE=auto`. If the input JSON
contains `section_path` metadata from sectioning, chunks are grouped by section
and then split by token budget. If no PDF outline was available and sectioning
was a no-op, chunking falls back to sequential fixed token windows. Useful
controls:

- `--mode`: `auto`, `section`, `token`, or `page`
- `--max-tokens`: target chunk budget, default `5000`
- `--overlap-tokens`: repeated tail context between adjacent token chunks, default `500`
- `--min-tokens`: minimum size before flushing a chunk, default `200`

Cross-page table continuations use the same relation as patching. The empty
continuation `table` blocks are not emitted as duplicate text chunks; instead,
their block indices, pages, and bboxes are attached to the master table chunk's
metadata. That keeps `chunk_content` clean while letting chunking view and
visual retrieval know that the table spans multiple pages.

Successful pipeline chunking also writes a `*_CHUNKING_VIEW.pdf` overlay. It
draws each chunk's source bboxes over the original PDF; adjacent chunks use
different semi-transparent colors so section boundaries and overlap are easier
to inspect. To generate it separately:

```bash
rag-flow chunk-view \
  --input-json /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json \
  --input-pdf /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_origin.pdf
```

Upsert text vectors:

```bash
rag-flow index text
```

Text indexing embeds and upserts chunks in batches. The default batch size is
`RAG_FLOW_INDEX_TEXT_BATCH_SIZE=256`; override it with
`rag-flow index text --batch-size <chunks>`.

Upsert ColPali visual vectors:

```bash
rag-flow index visual
```

By default this renders and embeds `RAG_FLOW_INDEX_VISUAL_BATCH_SIZE=8` PDF
pages per batch at `RAG_FLOW_INDEX_VISUAL_DPI=200`. Override them with
`rag-flow index visual --batch-size <pages> --dpi <dpi>` if the indexing machine
needs a different speed/memory tradeoff. If you point visual indexing at a PDF
outside the configured source, pass `--source-name <file.pdf>` so visual payloads
match the chunk metadata source.

ColPali is still a page-level visual index. When chunk output is available,
visual page payloads inherit page/section metadata from the chunk JSON, so
visual hits can be shown and filtered with the same section context as text
hits. The visual embeddings themselves are not section-level; section metadata
is auxiliary payload for retrieval context and explanation. Visual page payloads
store `chunk_ids_on_page` as pointers, but they do not store aggregated
`chunk_content`; answer context is pulled from the chunk-level text points.
Each `page-image-colpali` point represents exactly one rendered PDF page. It does not
reuse the table-continuation metadata from text chunks, so cross-page table
relations stay in chunk metadata instead of polluting the visual page index.
Visual indexing renders and upserts the PDF in page batches to avoid loading a
large manual as one giant image list in memory.
The collection also indexes `page_indices`, because cross-page chunks can belong
to several pages; this lets a visual hit on one page retrieve the text chunk that
spans into that page.
`rag-flow index inspect` prints the actual ColPali shape for a sampled visual
point, for example `page-image-colpali: 1030 patches x 128 dims`.

ColPali model loading prefers local files before downloading. Set
`RAG_FLOW_COLPALI_MODEL_PATH` to force a specific local directory, or leave it
empty and put a model under `RAG_FLOW_COLPALI_LOCAL_MODEL_ROOT` (default
`/root/autodl-tmp/models`). The resolver checks common layouts such as
`vidore/colpali-v1.3-merged`, `colpali-v1.3-merged`,
`vidore--colpali-v1.3-merged`, and Hugging Face cache directories like
`models--vidore--colpali-v1.3-merged/snapshots/<revision>`. If none exists, it
falls back to `RAG_FLOW_COLPALI_MODEL` and lets `from_pretrained` use the normal
cache/download behavior.

Chunking also records source block indices and page bboxes in chunk metadata.
The retriever uses those bboxes to keep visual evidence page-local: a ColPali
hit first contributes a `visual_page_prior`, then the candidate chunk receives a
conservative `visual_alignment_score` based on whether it belongs to that visual
page and how much of a cross-page chunk actually appears on the hit page. This
avoids blindly giving the same visual score to every chunk near the page. A
token-wise image-patch heatmap ranker exists as an experimental helper, but it is
not enabled in the default retriever until ColPali patch geometry can be mapped
reliably for every model output shape.

Inspect the Qdrant collection:

```bash
rag-flow index inspect
```

By default RAG-Flow opens Qdrant in Python local mode through
`RAG_FLOW_DB_PATH`. For an orchestrated Docker/server deployment, point the same
core retriever at a running Qdrant server instead:

```env
RAG_FLOW_QDRANT_URL=http://127.0.0.1:6333
RAG_FLOW_COLLECTION=technical-manuals
```

Test the retrieval API:

```bash
rag-flow test-retriever "How do I configure alarms?"
```

Start the retriever API:

```bash
rag-flow retriever
```

The default retrieval preset is the low-latency text-only preset selected by
the 220-query benchmark. It intentionally retrieves a large candidate pool, then
uses a hard retrieved-context budget so the downstream answer model is not forced
to consume all candidates:

```env
RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=0
RAG_FLOW_RETRIEVAL_ROUTE_MODE=text
RAG_FLOW_RETRIEVAL_CANDIDATE_MODE=direct
RAG_FLOW_RETRIEVAL_K=150
RAG_FLOW_FINAL_TOP_K=80
RAG_FLOW_RRF_K=10
RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS=10000
RAG_FLOW_RETRIEVAL_CONTEXT_CHARS_PER_TOKEN=4.0
RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO=1.0
```

`final_top_k` remains a maximum number of chunks. The token cap may return fewer
chunks, so retrieval does not need to fill the whole 10k budget when the useful
evidence is already covered. `RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO` is optional
and relative to the top candidate score; the current default keeps it at `1.0`
to disable relative filtering because the benchmark did not show a reliable
quality gain from ratio pruning for the online preset.

For offline high-recall review, keep the same text-only direct mode and raise
the retrieved-context budget. The thesis experiments used
`RAG_FLOW_RETRIEVAL_K=150` plus `RAG_FLOW_FINAL_TOP_K=80`, with no-cap runs for
diagnosis and capped 24k runs for a more practical review preset. Visual
retrieval is kept optional because it can improve visual/UI query ranking, but
it loads the ColPali query encoder and is much slower than the default text path.

Named presets can apply the thesis-recommended retrieval settings without
hand-editing the individual environment variables:

```bash
rag-flow preset list
rag-flow preset show default
rag-flow --preset default retriever
rag-flow --preset precise chat
rag-flow --preset enhanced chat
rag-flow --preset visual-route test-retriever "How do I configure alarms?"
```

The shipped presets are:

- `default`: text-only, `retrieval_k=150`, `final_top_k=80`, 10k retrieved-context cap.
- `precise`: text-only, same retrieval backbone, 5k retrieved-context cap for stricter answer context.
- `enhanced`: text-only, same retrieval backbone, 16k retrieved-context cap.
- `high-recall`: offline text-only review preset, 24k retrieved-context cap, `min_score_ratio=1.0`.
- `visual-route`: optional ColPali route, `visual-naive` plus `visual-page-local-naive`, 16k retrieved-context cap, `visual_weight=2.5`.

You can also set `RAG_FLOW_PRESET=enhanced` in the environment file. Explicit
environment variables still override preset defaults, so remove hand-written
retrieval values when you want the named preset to control them fully.

Retriever visual mode is optional. Dense and sparse text search run on CPU;
`page-image-colpali` visual search also uses Qdrant on CPU, but the ColPali query
encoder is a Torch model and can run on CPU or CUDA:

```env
# Full three-way retrieval, auto-select CUDA when available.
RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=1
RAG_FLOW_RETRIEVAL_DEVICE=auto

# Force GPU visual query encoding; fail if CUDA is unavailable.
RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=1
RAG_FLOW_RETRIEVAL_DEVICE=cuda

# Force CPU visual query encoding. Useful for low-frequency local use.
RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=1
RAG_FLOW_RETRIEVAL_DEVICE=cpu

# Text-only CPU retrieval: dense + sparse, no ColPali model loaded.
RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=0
```

`RAG_FLOW_QUANTIZED_COLPALI=1` only applies when the selected retrieval device
is CUDA. CPU visual mode loads ColPali in float32 instead of BitsAndBytes
quantization.

Start the LLM service on the remote GPU box:

```bash
rag-flow serve llm-sglang
```

The SGLang launcher is optional and reads its own profile settings from
`.local/rag-flow.env`. `rag-flow env create-llm` creates an isolated LLM Python
environment under `RAG_FLOW_ENV_ROOT`, installs `RAG_FLOW_LLM_INSTALL_SPEC`
with uv pip by default, installs the CuDNN compatibility fix package
`RAG_FLOW_LLM_CUDNN_PACKAGE` when `RAG_FLOW_LLM_FIX_CUDNN=1`, and writes
`RAG_FLOW_LLM_PYTHON_BIN` / `RAG_FLOW_SGLANG_PYTHON` back to the env file. The
default SGLang profile is `qwen3.6-35b-a3b-gptq-int4`.

Download the default profile before the first launch. The command first looks
for an existing local model under `RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT`
(`/root/autodl-tmp/models` by default), then the ModelScope cache, then the
Hugging Face cache. If none exist, the default source is `auto`, which tries
ModelScope first and then Hugging Face:

```bash
rag-flow download llm
rag-flow download llm --dry-run
rag-flow download llm --source modelscope
rag-flow download llm --source hf
```

The download command uses `RAG_FLOW_SGLANG_PYTHON` /
`RAG_FLOW_LLM_PYTHON_BIN`, installs the selected downloader package first when
needed (`modelscope` or `huggingface_hub`), writes the resolved source/model
id/path/served model name back to `.local/rag-flow.env`, and keeps startup
explicit. With `RAG_FLOW_SGLANG_DOWNLOAD_SOURCE=auto`, a ModelScope download
failure falls back to Hugging Face. It does not run automatically inside `rag-flow serve llm-sglang`
because model downloads can be very large. For private Hugging Face repos, set
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `RAG_FLOW_SGLANG_HF_TOKEN` in
`.local/rag-flow.env`.
For manually uploaded models, put the directory at either
`/root/autodl-tmp/models/<owner>/<model-name>` or
`/root/autodl-tmp/models/<model-name>`.

Switch profiles or override paths like this:

```bash
rag-flow download llm --profile qwen3.6-35b-a3b-gptq-int4
rag-flow download llm --profile qwen3.5-35b-a3b-gptq-int4
rag-flow download llm --source hf --model-id palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 --model-path /root/.cache/huggingface/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
rag-flow download llm --model-id palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 --model-path /root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
rag-flow serve llm-sglang --profile qwen3.6-35b-a3b-gptq-int4
rag-flow serve llm-sglang --profile qwen3.5-35b-a3b-gptq-int4
rag-flow serve llm-sglang --model-path /root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
rag-flow serve llm-sglang --served-model-name palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
```

Useful launcher/download variables include `RAG_FLOW_SGLANG_MODEL_PROFILE`,
`RAG_FLOW_SGLANG_MODEL_ID`, `RAG_FLOW_SGLANG_MODEL_PATH`,
`RAG_FLOW_SGLANG_SERVED_MODEL_NAME`, `RAG_FLOW_SGLANG_MODEL_REVISION`,
`RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT`, `RAG_FLOW_SGLANG_DOWNLOAD_SOURCE`,
`RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE`,
`RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_HUGGINGFACE_HUB`, `RAG_FLOW_SGLANG_HF_TOKEN`,
`RAG_FLOW_SGLANG_PORT`, `RAG_FLOW_SGLANG_CONTEXT_LENGTH`,
`RAG_FLOW_SGLANG_MEM_FRACTION_STATIC`, `RAG_FLOW_SGLANG_QUANTIZATION`,
`RAG_FLOW_SGLANG_ATTENTION_BACKEND`, and `RAG_FLOW_SGLANG_KV_CACHE_DTYPE`. Set
`RAG_FLOW_LLM_FIX_CUDNN`, `RAG_FLOW_LLM_CUDNN_PACKAGE`,
`RAG_FLOW_LLM_EXTRA_PACKAGES`, and `RAG_FLOW_CREATE_LLM_INSTALL_UV`. Set
`RAG_FLOW_CREATE_LLM_INSTALL_UV=0` if the LLM environment should not install uv
before package installation. Set `RAG_FLOW_LLM_FIX_CUDNN=0` only if you want to
handle SGLang/PyTorch/CuDNN compatibility yourself.

Chat from the terminal:

```bash
rag-flow chat
```

## Security Notes

Keep credentials and private source documents under `.local/`. The whole
directory is ignored by Git and is meant for API keys, SSH details, private env
files, and raw technical documents such as source PDFs or manual folders.

Tracked files should only contain templates, placeholders, or public defaults.
For local configuration, copy `.env.example` to `.local/rag-flow.env` and put
real values there. Runtime artifacts such as Qdrant databases, downloaded
models, and conda environments can stay on the machine paths configured in that
private env file.

If the retriever is exposed beyond localhost, set `RAG_FLOW_RETRIEVER_API_KEY`
and send it as `Authorization: Bearer <token>`. Patching and captioning call the
configured local OpenAI-compatible LLM endpoint instead of loading a VLM inside
the pipeline process.

## Configuration

All core values are environment variables. The important ones are:

- `RAG_FLOW_BASE_DIR`
- `RAG_FLOW_SOURCE_PDF`
- `RAG_FLOW_MINERU_COMMAND`
- `RAG_FLOW_MINERU_INPUT_PATH`
- `RAG_FLOW_MINERU_OUTPUT_DIR`
- `RAG_FLOW_MINERU_BACKEND`
- `RAG_FLOW_MINERU_MODEL_SOURCE`
- `RAG_FLOW_MINERU_VERSION`
- `RAG_FLOW_MINERU_PYTHON`
- `RAG_FLOW_INIT_INSTALL_APT_PACKAGES`
- `RAG_FLOW_INIT_APT_PACKAGES`
- `RAG_FLOW_PATCH_MAX_NEW_TOKENS`
- `RAG_FLOW_PATCH_LLM_TIMEOUT`
- `RAG_FLOW_PATCH_BATCH_SIZE`
- `RAG_FLOW_PATCH_CONCURRENCY`
- `RAG_FLOW_PATCH_CHECKPOINT_INTERVAL`
- `RAG_FLOW_PATCH_INVALID_RETRY_LIMIT`
- `RAG_FLOW_PATCH_DPI`
- `RAG_FLOW_PATCH_PAGE_WINDOW_SIZE`
- `RAG_FLOW_CAPTION_MAX_NEW_TOKENS`
- `RAG_FLOW_CAPTION_MAX_CONTEXT_TOKENS`
- `RAG_FLOW_CAPTION_BATCH_SIZE`
- `RAG_FLOW_CAPTION_CONCURRENCY`
- `RAG_FLOW_CAPTION_LLM_TIMEOUT`
- `RAG_FLOW_CONTENT_JSON`
- `RAG_FLOW_SECTIONED_JSON`
- `RAG_FLOW_PATCHED_JSON`
- `RAG_FLOW_CAPTIONED_JSON`
- `RAG_FLOW_CHUNKS_JSON`
- `RAG_FLOW_CHUNK_MODE`
- `RAG_FLOW_CHUNK_MAX_TOKENS`
- `RAG_FLOW_CHUNK_OVERLAP_TOKENS`
- `RAG_FLOW_CHUNK_MIN_TOKENS`
- `RAG_FLOW_DB_PATH`
- `RAG_FLOW_QDRANT_URL`
- `RAG_FLOW_QDRANT_API_KEY`
- `RAG_FLOW_QDRANT_PREFER_GRPC`
- `RAG_FLOW_QDRANT_TIMEOUT`
- `RAG_FLOW_COLLECTION`
- `RAG_FLOW_PIPELINE_ENV`
- `RAG_FLOW_PIPELINE_PYTHON`
- `RAG_FLOW_PIPELINE_PYTHON_BIN`
- `RAG_FLOW_PIPELINE_TORCH_INDEX_URL`
- `RAG_FLOW_LLM_BASE_URL`
- `RAG_FLOW_LLM_MODEL`
- `RAG_FLOW_LLM_PYTHON_BIN`
- `RAG_FLOW_SGLANG_MODEL_PROFILE`
- `RAG_FLOW_SGLANG_MODEL_ID`
- `RAG_FLOW_SGLANG_MODEL_PATH`
- `RAG_FLOW_SGLANG_SERVED_MODEL_NAME`
- `RAG_FLOW_SGLANG_MODEL_REVISION`
- `RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT`
- `RAG_FLOW_SGLANG_DOWNLOAD_SOURCE`
- `RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE`
- `RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_HUGGINGFACE_HUB`
- `RAG_FLOW_SGLANG_HF_TOKEN`
- `RAG_FLOW_SGLANG_PYTHON`
- `RAG_FLOW_RETRIEVER_API_KEY`
- `RAG_FLOW_RETRIEVER_MAX_QUERY_CHARS`
- `RAG_FLOW_TRUSTED_REMOTE_CODE_MODELS`
- `RAG_FLOW_VLM_MODEL_REVISION`

See `.env.example` for the full list.

The local initialization command `rag-flow init china-all` reads the same
env file and runs `soft-links`, `cpu-cores`, and `china-sources` in that
order. Use `RAG_FLOW_INIT_*` variables to choose apt mirrors, pip/uv indexes,
Hugging Face cache/mirror settings, `MINERU_MODEL_SOURCE`, locale, and conda
channels without editing the script. The individual helpers remain available
as `rag-flow init soft-links`, `rag-flow init cpu-cores`, and
`rag-flow init china-sources`.

`china-sources` also writes `RAG_FLOW_ENV_FILE=<repo>/.local/rag-flow.env` into
the managed shell block. That pins future `rag-flow` commands to the project
env file even when they are launched from `/root` or another directory.

For China mirrors, `RAG_FLOW_INIT_MIRROR_ORDER=aliyun,tencent,tuna` is the
default. `china-sources` probes apt, pip/uv, and conda mirrors independently,
uses the first reachable profile for each category, and writes the selected
managed defaults back to `.local/rag-flow.env` so later `rag-flow env ...`
commands inherit the working source.

For Python environments, use the split setup under `scripts/env/`:

```bash
rag-flow env create-mineru
rag-flow env create-pipeline
rag-flow env create-llm
```

`create-pipeline` installs the CLI plus patching, captioning, chunking,
indexing, retriever, and chat dependencies. It writes the resolved Python path
back to `.local/rag-flow.env`, so module commands can re-enter the right
environment automatically.

On a new AutoDL China machine, run initialization first, then create the MinerU
and pipeline environments, then check MinerU:

```bash
rag-flow init china-all
rag-flow env create-mineru
rag-flow env create-pipeline
rag-flow mineru doctor
rag-flow mineru run --dry-run
```

The default environment strategy is `RAG_FLOW_ENV_MANAGER=auto`:
prefer micromamba/conda for isolated path-based environments under
`RAG_FLOW_ENV_ROOT`. `rag-flow env create-mineru` ensures `uv` is available
by default and then uses it for fast pip installs. Set `RAG_FLOW_USE_UV=0`
to skip installing and using uv, or set `RAG_FLOW_CREATE_MINERU_INSTALL_UV=0`
to only skip the automatic install step. Keep
`RAG_FLOW_ENV_ROOT`, pip/uv caches, conda package caches, Hugging Face cache,
ModelScope cache, and Torch cache under `~/autodl-tmp` on rented GPU machines.
The pipeline setup uses the PyTorch CUDA 12.8 wheel index by default, which
fits current RTX 50-series Linux machines better than mixing all packages into
one environment.
