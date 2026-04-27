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
rag-flow ingest --to-stage chunks
rag-flow preprocess icons
rag-flow preprocess captions
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
`RAG_FLOW_MINERU_OUTPUT_DIR`.

Preview the command sequence without running heavy steps:

```bash
rag-flow ingest --dry-run
```

Resume from a later stage:

```bash
rag-flow ingest --from-stage captions --to-stage chunks
```

Run through text indexing too:

```bash
rag-flow ingest --to-stage index-text
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
rag-flow preprocess icons
```

Generate image descriptions:

```bash
rag-flow preprocess captions
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
`RAG_FLOW_ENV_ROOT`, and use uv for fast package installs. Keep
`RAG_FLOW_ENV_ROOT`, pip/uv caches, conda package caches, Hugging Face cache,
ModelScope cache, and Torch cache under `~/autodl-tmp` on rented GPU machines.
The GPU setup uses the PyTorch CUDA 12.8 wheel index by default, which fits
current RTX 50-series Linux machines better than mixing all packages into one
environment.
