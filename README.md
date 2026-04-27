# RAG Flow

RAG Flow is a multimodal retrieval augmented generation pipeline for technical
manuals. The default profile uses generic example names, and paths, model names,
and collection names can be changed through environment variables.

## What It Does

1. Patch MinerU output with a vision language model:
   - recover small icon text that MinerU/OCR missed
   - add context-aware descriptions to extracted images
2. Build page-level chunks from enriched `content_list.json`.
3. Store three retrieval signals in Qdrant:
   - dense text vectors
   - sparse BM25 vectors
   - ColPali page-image multivectors
4. Serve a FastAPI `/retrieve` endpoint with RRF fusion.
5. Use an OpenAI-compatible LLM endpoint for cited terminal chat.

## Project Layout

```text
src/rag_flow/
  config.py                 Environment-driven configuration
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
mkdir -p .secrets
cp .env.example .secrets/rag-flow.env
pip install -e ".[retrieval,preprocess]"
```

The app automatically loads `.secrets/rag-flow.env` when commands are run from
this repository. You can still point to another local env file with
`RAG_FLOW_ENV_FILE=/path/to/file`.

For the original AutoDL-style environment, the exported conda YAML files are in
`envs/`. They are intentionally preserved because CUDA, ColPali, MinerU, and
SGLang package compatibility is sensitive.

## Pipeline

Patch small icons:

```bash
rag-flow-patch-icons
```

Generate image descriptions:

```bash
rag-flow-caption-images
```

Build page chunks:

```bash
rag-flow-chunk
```

Upsert text vectors:

```bash
rag-flow-index text
```

Upsert ColPali visual vectors:

```bash
rag-flow-index visual
```

Inspect the Qdrant collection:

```bash
rag-flow-index inspect
```

Test the retrieval API:

```bash
rag-flow-test-retriever "How do I configure alarms?"
```

Start the retriever API:

```bash
rag-flow-retriever
```

Start the LLM service on the remote GPU box:

```bash
scripts/serve-llm-sglang.sh
```

Chat from the terminal:

```bash
rag-flow-chat
```

## Security Notes

Keep credentials and private source documents under `.secrets/`. The whole
directory is ignored by Git and is meant for API keys, SSH details, private env
files, and raw technical documents such as source PDFs or manual folders.

Tracked files should only contain templates, placeholders, or public defaults.
For local configuration, copy `.env.example` to `.secrets/rag-flow.env` and put
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
