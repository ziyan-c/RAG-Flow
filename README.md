# RAG Flow

RAG Flow is a multimodal retrieval augmented generation pipeline for technical
manuals. The current profile targets the Dahua DSS Professional manual, but the
code is structured so paths, model names, and collection names can be changed
through environment variables.

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
cp .env.example .env
export RAG_FLOW_ENV_FILE="$PWD/.env"
pip install -e ".[retrieval,preprocess]"
```

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

The old workspace had hard-coded API keys and SSH credentials. They were not
carried over. Put secrets in a local `.env` file or your shell environment, and
rotate any keys/passwords that were previously committed or shared in plaintext.

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

See `.env.example` for the full list.
