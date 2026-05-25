# RAG-Flow

RAG-Flow is a multimodal retrieval-augmented generation pipeline for technical
manuals. It turns PDF manuals into auditable evidence objects, indexes them with
hybrid text and optional visual retrieval, and serves source-grounded answers
through an OpenAI-compatible language model.

The project was built around a practical question: how do we make RAG reliable
on manuals where the answer may live in prose, tables, screenshots, diagrams,
inline icons, page layout, or several pages at once?

## Highlights

- End-to-end PDF pipeline: MinerU parsing, PDF-outline sectioning, small-icon
  repair, image captioning, section-aware chunking, Qdrant indexing, retrieval,
  and answering.
- Hybrid retrieval: dense vectors, sparse BM25-style vectors, reciprocal rank
  fusion, and optional ColPali page-level visual retrieval.
- Source-first design: every block, chunk, retrieved context, and answer payload
  carries source identity, page information, section breadcrumbs, and audit
  metadata.
- Thesis-scale evaluation: 88 answer-bearing runs, 6,001 generated answers, and
  18,003 independent review judgments over a 14-PDF technical-manual corpus.
- Deployable presets: named runtime modes for default online QA, high recall,
  compact low-token use, visual recall, and diagnostic image-input answering.

## System Overview

```mermaid
flowchart LR
  subgraph Offline["Offline ingestion"]
    PDF["Source PDFs"] --> MinerU["MinerU parsing"]
    MinerU --> Sectioning["Outline sectioning<br/>source identity"]
    Sectioning --> Patching["Small-icon patching<br/>VLM crop repair"]
    Patching --> Captioning["Image captioning<br/>answering policy"]
    Captioning --> Chunking["Section-aware chunking"]
    Chunking --> TextIndex["Dense + sparse<br/>text index"]
    PDF --> VisualIndex["Optional ColPali<br/>page index"]
  end

  TextIndex --> Qdrant["Qdrant collection"]
  VisualIndex --> Qdrant

  subgraph Online["Online QA"]
    Query["User query"] --> Retrieval["Hybrid retrieval<br/>RRF + context cap"]
    Qdrant --> Retrieval
    Retrieval --> Payload["Final answer payload<br/>text + optional images"]
    Payload --> Answerer["Qwen/SGLang or<br/>OpenAI-compatible LLM"]
    Answerer --> Answer["Grounded answer"]
  end
```

## Why This Exists

Technical manuals are a poor fit for naive text-only RAG. Important evidence can
appear as:

- a table row with repeated headers,
- a small UI icon dropped by OCR,
- a screenshot caption that explains a workflow,
- a visual dimension drawing with almost no extracted text,
- a procedure split across section boundaries,
- or a comparison across several PDF files.

RAG-Flow treats these cases as first-class system problems. The pipeline enriches
manuals before indexing, keeps the online default conservative, and exposes
heavier multimodal modes only when their cost is justified.

## Pipeline Contract

| Stage | Input | Output | Purpose |
| --- | --- | --- | --- |
| `parsing` | Source PDF | MinerU `content_list.json`, images, origin PDF | Extract layout-aware blocks and assets. |
| `sectioning` | MinerU JSON + source PDF | `*_SECTIONED.json` | Add source identity and PDF-outline breadcrumbs before LLM enrichment. |
| `patching` | Sectioned JSON + source PDF | `*_SECTIONED_PATCHED.json`, patching view PDF | Repair small inline icons and missing visual symbols. |
| `captioning` | Sectioned patched JSON | `*_SECTIONED_PATCHED_CAPTIONED.json`, captioning view PDF | Describe real image blocks and record whether images may be needed during answering. |
| `chunking` | Captioned JSON | `*_CHUNKED.json`, chunking view PDF | Build source-aware, section-aware retrieval units. |
| `indexing` | Chunked JSON + PDF pages | Qdrant text and optional visual points | Store dense text, sparse text, and optional ColPali page vectors. |
| `retrieval` | Qdrant collection + query | Ranked chunks, context text, optional image URLs | Build the final answer payload under explicit top-k and token rules. |
| `answering` | Retrieval final output | LLM answer, usage, latency, review inputs | Generate source-grounded answers without silently choosing extra evidence. |

## Evaluation Snapshot

The final experiments used a mixed technical-manual corpus: long DSS software
manuals plus visually dense camera and switch sheets.

| Item | Count |
| --- | ---: |
| Source PDFs | 14 |
| Source pages | 1,114 |
| Final validation questions | 200 |
| Evidence items in the gold set | 244 |
| Answer-bearing runs | 88 |
| Generated answers | 6,001 |
| Independent review pass judgments | 18,003 |

### Final Presets

Scores are mean 0-5 answer-quality scores from the final 200-question validation
unless noted otherwise.

| Preset | Route | Main settings | Mean score | Avg retrieved context | Use when |
| --- | --- | --- | ---: | ---: | --- |
| `default` | text-only | `k=80`, `top_k=20`, 10k cap | 2.3383 | ~10,072 tokens | Normal online QA with stable latency and simple deployment. |
| `high-recall` | text-only | `k=150`, `top_k=80`, 16k cap | 2.4133 | ~16,585 tokens | Hard questions, manual review, or maximum evidence coverage. |
| `compact` | text-only | `k=150`, `top_k=10`, ratio `0.4`, 10k cap | 2.3833 | ~4,156 tokens | Low-token use, batch previews, and fast screening. |
| `visual-recall` | ColPali-assisted retrieval | naive page visual bonus, `k=150`, `top_k=20`, 10k cap | 2.3483 | ~10,315 tokens | UI-heavy or diagram-heavy queries where slower retrieval is acceptable. |
| `default-with-image-input` | default retrieval + images to answerer | selected evidence images appended | 2.3183 | ~10,251 text tokens | Diagnostics when the answerer must inspect images. |
| `compact-with-image-input` | compact retrieval + images to answerer | selected evidence images appended | not separately run on 200Q | ~4,156 text tokens expected | Image diagnostics with smaller text context. |

Important negative findings:

- `24k` context is not shipped. It produced 124 usage-missing cases in the
  200-question validation under the current serving limit.
- Thinking mode is not enabled by default. It reduced quality and increased
  latency in the measured runs.
- Always sending images is not the default. Repaired image-input runs delivered
  1,378 image URLs to the answerer, but did not improve mean quality.

## Quick Start

### 1. Install the core package

```bash
cd RAG-Flow

mkdir -p .local
cp .env.example .local/rag-flow.env

pip install -e ".[text-retrieval]"
```

Use the full local stack when you need MinerU, patching, captioning, and visual
retrieval on the same machine:

```bash
pip install -e ".[mineru,preprocess,retrieval]"
```

The lighter `text-retrieval` extra is enough for the default online text path.
The visual stack pulls in heavier Torch, ColPali, and related dependencies.

### 2. Configure local paths

RAG-Flow automatically reads `.local/rag-flow.env` from the repository root.
Keep private paths, API keys, source PDFs, model paths, and machine-specific
settings there.

Common local layout:

```text
source-pdfs/   input PDF root
output-pdfs/   MinerU and preprocessing artifacts
qdrant-db/     local Qdrant vector database
.local/        private env file, secrets, local-only state
```

### 3. Run the offline pipeline

```bash
rag-flow mineru doctor
rag-flow mineru run --input source-pdfs --output-dir output-pdfs
rag-flow ingest --to-stage chunking
rag-flow index text
```

For optional visual retrieval, build the page-level ColPali index too:

```bash
rag-flow index visual --chunks /path/to/current_CHUNKED.json
```

Or run through indexing from the staged pipeline:

```bash
RAG_FLOW_INDEX_MODE=both rag-flow ingest --to-stage indexing
```

### 4. Start retrieval and answer questions

```bash
rag-flow --preset default retriever
rag-flow --preset default chat
```

Useful alternatives:

```bash
rag-flow --preset compact chat
rag-flow --preset high-recall chat
rag-flow --preset visual-recall test-retriever "Which port is used for power?"
rag-flow --preset default-with-image-input chat
```

## Command Map

All user-facing operations are exposed through one command:

```bash
rag-flow mineru doctor
rag-flow mineru setup
rag-flow mineru run
rag-flow section
rag-flow patch
rag-flow patch-view
rag-flow caption
rag-flow caption-view
rag-flow chunk
rag-flow chunk-view
rag-flow index text
rag-flow index visual
rag-flow index inspect
rag-flow retriever
rag-flow test-retriever "How do I configure alarms?"
rag-flow chat
rag-flow preset list
rag-flow preset show default
rag-flow benchmark --help
```

Most script-backed setup commands support `--dry-run`:

```bash
rag-flow init china-all --dry-run
rag-flow env create-mineru --dry-run
rag-flow env create-pipeline --dry-run
rag-flow serve llm-sglang --dry-run
```

## Named Presets

Presets set retrieval and answering environment variables without hand-editing
the env file.

```bash
rag-flow preset list
rag-flow preset show high-recall
rag-flow --preset compact retriever
```

You can also set a preset in `.local/rag-flow.env`:

```env
RAG_FLOW_PRESET=high-recall
```

Explicit environment variables override preset values. Remove hand-written
retrieval variables when you want a named preset to control the full policy.

## Configuration

The full environment surface is documented in `.env.example`. The most important
families are:

| Family | Examples | Purpose |
| --- | --- | --- |
| Source and artifacts | `RAG_FLOW_SOURCE_ROOT`, `RAG_FLOW_MINERU_OUTPUT_DIR`, `RAG_FLOW_CONTENT_JSON` | Locate PDFs and stage outputs. |
| Patching | `RAG_FLOW_PATCH_DPI`, `RAG_FLOW_PATCH_CONCURRENCY`, `RAG_FLOW_PATCH_MAX_NEW_TOKENS` | Control small-icon repair. |
| Captioning | `RAG_FLOW_CAPTION_MAX_CONTEXT_TOKENS`, `RAG_FLOW_CAPTION_CONCURRENCY` | Control image-description generation. |
| Chunking | `RAG_FLOW_CHUNK_MODE`, `RAG_FLOW_CHUNK_MAX_TOKENS`, `RAG_FLOW_CHUNK_OVERLAP_TOKENS` | Shape retrieval units. |
| Indexing | `RAG_FLOW_DB_PATH`, `RAG_FLOW_QDRANT_URL`, `RAG_FLOW_COLLECTION`, `RAG_FLOW_INDEX_MODE` | Configure local or server Qdrant. |
| Retrieval | `RAG_FLOW_RETRIEVAL_K`, `RAG_FLOW_FINAL_TOP_K`, `RAG_FLOW_RRF_K`, `RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS` | Select evidence for answering. |
| LLM serving | `RAG_FLOW_LLM_BASE_URL`, `RAG_FLOW_LLM_MODEL`, `RAG_FLOW_SGLANG_MODEL_PATH` | Connect to the OpenAI-compatible answerer. |

## Repository Layout

```text
src/rag_flow/
  cli.py                    Unified command dispatcher
  config.py                 Environment loading and settings
  mineru.py                 MinerU setup and parsing helpers
  sectioning.py             PDF-outline section annotation
  preprocessing/            Small-icon patching, image captioning, view PDFs
  chunking.py               Section-aware and token-window chunking
  indexing.py               Qdrant text and visual indexing
  retrieval.py              Hybrid retrieval and context assembly
  api.py                    FastAPI retrieval service
  chat_cli.py               Terminal answering client
  presets.py                Canonical runtime presets
  benchmark/                Experiment and review utilities

scripts/
  env/                      Environment creation helpers
  experiments/              V2 experiment runners and summarizers
  init/                     Machine/bootstrap helpers
  llm/                      SGLang model download helper
  remote/                   Remote-machine helper scripts

qa-goldset/                 Gold questions, evidence cards, review helpers
source-pdfs/                Local source PDF tree
output-pdfs/                Generated parse/enrichment artifacts
docs/                       Architecture notes and operational docs
tests/                      Unit and regression tests
```

## Development

```bash
pip install -e ".[dev,text-retrieval]"
pytest
```

Run only a focused test while developing:

```bash
pytest tests/test_presets.py
pytest tests/test_chunking.py
```

## Security and Data Hygiene

- Do not commit private manuals, API keys, SSH details, model credentials, or
  raw customer data.
- Put local configuration in `.local/rag-flow.env`; `.local/` is ignored by Git.
- If the retriever is exposed beyond localhost, set
  `RAG_FLOW_RETRIEVER_API_KEY` and send `Authorization: Bearer <token>`.
- Runtime artifacts such as Qdrant databases, downloaded models, and generated
  PDF overlays should stay in ignored local paths unless deliberately published.

## Related Documents

- [Architecture notes](docs/architecture.md)
- [ColPali environment notes](docs/colpali-environment.md)
- [Migration notes](docs/migration.md)
- [Known issues](docs/known-issues.md)
- [QA gold set](qa-goldset/README.md)

The Politecnico thesis artifacts in `thesis-polimi/` document the final
experimental narrative and the complete v1/v2 analysis used to choose the
presets above.
