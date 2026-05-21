# Source PDF QA Goldset

This directory contains an auditable 200-item QA set generated from every PDF under `source-pdfs/`.

## Files

- `source-pdfs-qa-200.json`: final QA set.
- `source-pdfs-qa-200.codex-reviewed.json`: Codex-reviewed final QA set after evidence audit and manual visual review.
- `source-pdfs-qa-50.quick-stratified.json`: stratified 50-item quick set sampled from the Codex-reviewed QA set for broad parameter screening.
- `source-pdfs-qa-50.quick-stratified-report.json`: coverage report for the quick set, including type, difficulty, visual, multi-page, multi-PDF, non-contiguous evidence, and source-PDF counts.
- `source-pdfs-inventory.json`: PDF inventory with page counts, text volume, visual-page signals, and TOC samples.
- `qa-generation-report.json`: validation report with coverage and question-type counts.
- `codex-review/`: review report, evidence audit, and rendered visual pages used for the Codex review pass.
- `evidence-cards/`: page-level evidence cards for each source PDF.

## Quick Parameter Screen

Use `source-pdfs-qa-50.quick-stratified.json` for fast parameter sweeps before running the full 200-item evaluation. The quick set preserves all source-PDF coverage and intentionally includes representative fact, procedure, table/spec, visual, cross-page, multi-PDF, and safety/notice questions. It also keeps the three non-contiguous page-evidence questions so retrieval settings are tested against the hardest context-linking cases.

The quick set is for screening and debugging only. Final Pareto selection should be validated on `source-pdfs-qa-200.codex-reviewed.json`.

## QA Schema

Each item contains:

- `id`
- `question`
- `answer`
- `source_pdfs`
- `evidence`: PDF path, page index, page number, modality, and support text
- `question_type`
- `requires_visual`
- `requires_multiple_pages`
- `requires_multiple_pdfs`
- `difficulty`

## Regeneration

Run:

```bash
.venv/bin/python scripts/qa_goldset/build_source_pdf_goldset.py
```

The generator validates that the final set has exactly 200 items, unique IDs, non-empty evidence, consistent page ranges, no table-of-contents support pollution, visual evidence for visual questions, and coverage for all 14 source PDFs.
