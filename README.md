# RAG Flow

RAG Flow is a multimodal retrieval augmented generation pipeline for technical
manuals. The default profile uses generic example names, and paths, model names,
and collection names can be changed through environment variables.

## What It Does

1. Parse the source PDF with MinerU into structured `content_list.json`.
2. Patch MinerU output with a vision language model:
   - recover small icon text that MinerU/OCR missed
   - add context-aware descriptions to extracted images
3. Build page-level chunks from enriched `content_list.json`.
4. Store three retrieval signals in Qdrant:
   - dense text vectors
   - sparse BM25 vectors
   - ColPali page-image multivectors
5. Serve a FastAPI `/retrieve` endpoint with RRF fusion.
6. Use an OpenAI-compatible LLM endpoint for cited terminal chat.

## Project Layout

```text
src/rag_flow/
  config.py                 Environment-driven configuration
  mineru.py                 MinerU install/check/run helpers
  pipeline.py               End-to-end ingestion orchestration
  chunking.py               MinerU JSON to page-level chunks
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
pip install -e ".[retrieval,preprocess]"
```

The app automatically loads `.local/rag-flow.env` when commands are run from
this repository. You can still point to another local env file with
`RAG_FLOW_ENV_FILE=/path/to/file`.

For the original AutoDL-style environment, the exported conda YAML files are in
`envs/`. They are intentionally preserved because CUDA, ColPali, MinerU, and
SGLang package compatibility is sensitive.

## Command Line

All operations are available through the unified `rag-flow` command:

```bash
rag-flow init china-all
rag-flow env create-mineru
rag-flow mineru doctor
rag-flow mineru run
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

Run the default ingestion path from PDF to page chunks:

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
source PDF name, then derives the downstream artifact paths from that folder.
Set `RAG_FLOW_MINERU_BACKEND=pipeline` for CPU-friendly parsing. Set
`RAG_FLOW_MINERU_MODEL_SOURCE=modelscope` to run MinerU with
`MINERU_MODEL_SOURCE=modelscope`.

Run MinerU only:

```bash
rag-flow mineru run
```

Patch small icons:

```bash
rag-flow patch --artifact-dir /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto
```

The artifact-dir form is the preferred patching entrypoint after MinerU has
parsed a PDF. It expects a MinerU output folder containing
`*_content_list.json` and `*_origin.pdf`, then writes
`*_content_list_PATCHED.json` in the same folder. The captioning stage then
writes `*_content_list_PATCHED_CAPTIONED.json`.

Patching focuses on content blocks instead of page furniture: text, lists, and
tables are patched, while headers, footers, page numbers, and empty fields are
skipped. Small uncaptioned `image` blocks are treated as possible inline icons:
patching links them to nearby text/list blocks or containing table cells, expands
the visual crop to include them, marks them in the JSON, and keeps captioning
from describing them as standalone figures. MinerU represents cross-page table
continuations as empty `table` blocks; those blocks are not copied into the JSON
as duplicate text. Instead, their PDF crops are stacked onto the previous table
crop so the VLM can patch the single complete `table_body` with visual evidence
from every page of the same table.

The source PDF is rendered in page windows instead of loading the whole book at
once; the default is 200 pages per window. When `--artifact-dir` points at a
parent folder, patching finds nested MinerU artifact folders recursively and
processes one PDF at a time to avoid GPU memory spikes:

```bash
rag-flow patch \
  --artifact-dir /root/autodl-tmp/manuals/public \
  --page-window-size 200 \
  --batch-size 6
```

The VLM prompt is intentionally strict: preserve all existing extracted text and
only insert `[Icon: ...]` markers. The run writes a checkpoint after each VLM
batch by default, resumes from that checkpoint on retry, deletes the checkpoint
after success, writes a `*_PATCHING_VIEW.pdf` overlay that shows the exact crop
regions sent to the VLM, and prints patching statistics at the end. Useful
controls:

- `--batch-size`: VLM request batch size, default `6`
- `--max-new-tokens`: generation budget, default `5000`
- `--page-window-size`: PDF render window size, default `200`
- `--checkpoint-interval`: write checkpoint every N VLM batches, default `1`
- `--patching-view-pdf`: custom path for the overlay PDF
- `--no-patching-view`: skip writing the overlay PDF
- `--no-resume`: ignore an existing checkpoint
- `--no-recursive`: only patch the exact artifact folder

You can also regenerate the overlay without running the VLM:

```bash
rag-flow patch-view \
  --input-json /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_content_list_PATCHED.json \
  --input-pdf /root/autodl-tmp/manuals/public/example-technical-manual/hybrid_auto/example-technical-manual_origin.pdf
```

Generate image descriptions:

```bash
rag-flow caption
```

Build page chunks:

```bash
rag-flow chunk
```

Upsert text vectors:

```bash
rag-flow index text
```

Upsert ColPali visual vectors:

```bash
rag-flow index visual
```

Inspect the Qdrant collection:

```bash
rag-flow index inspect
```

Test the retrieval API:

```bash
rag-flow test-retriever "How do I configure alarms?"
```

Start the retriever API:

```bash
rag-flow retriever
```

Start the LLM service on the remote GPU box:

```bash
rag-flow serve llm-sglang
```

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
and send it as `Authorization: Bearer <token>`. VLM preprocessing executes
model-provided Python code for trusted repositories only; keep
`RAG_FLOW_TRUSTED_REMOTE_CODE_MODELS` narrow and set `RAG_FLOW_VLM_MODEL_REVISION`
when pinning a model snapshot.

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
- `RAG_FLOW_CONTENT_JSON`
- `RAG_FLOW_PATCHED_JSON`
- `RAG_FLOW_CAPTIONED_JSON`
- `RAG_FLOW_CHUNKS_JSON`
- `RAG_FLOW_DB_PATH`
- `RAG_FLOW_COLLECTION`
- `RAG_FLOW_LLM_BASE_URL`
- `RAG_FLOW_LLM_MODEL`
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

For China mirrors, `RAG_FLOW_INIT_MIRROR_ORDER=aliyun,tencent,tuna` is the
default. `china-sources` probes apt, pip/uv, and conda mirrors independently,
uses the first reachable profile for each category, and writes the selected
managed defaults back to `.local/rag-flow.env` so later `rag-flow env ...`
commands inherit the working source.

For Python environments, use the split setup under `scripts/env/`:

```bash
rag-flow env create-core
rag-flow env create-mineru
rag-flow env create-gpu
rag-flow env create-llm
```

On a new AutoDL China machine, run initialization first, then create the MinerU
environment, then check MinerU:

```bash
rag-flow init china-all
rag-flow env create-mineru
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
The GPU setup uses the PyTorch CUDA 12.8 wheel index by default, which fits
current RTX 50-series Linux machines better than mixing all packages into one
environment.
