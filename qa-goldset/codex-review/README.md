# Codex Review Notes

This directory records the Codex review pass for `source-pdfs-qa-200.codex-reviewed.json`.

## Review Scope

- Reviewed 200 QA items from `source-pdfs-qa-200.json`.
- Read all 14 source PDFs into the review pass, covering 1114 total pages.
- Checked 244 evidence entries.
- Verified 208 text/table evidence entries against extracted text from the cited original PDF page.
- Rendered and manually inspected 36 visual evidence entries across 7 unique visual PDF pages.
- Confirmed all 14 source PDFs remain covered.

## Visual Pages Reviewed

- `source-pdfs/S3006-4ET-60/S3006-4ET-36_DIMENSIONS.pdf`, page 1
- `source-pdfs/S3006-4ET-60/S3006-4ET-36_INSTALLATION_METHOD_EN.pdf`, page 1
- `source-pdfs/S3006-4ET-60/S3006-4ET-36_PORT.pdf`, page 1
- `source-pdfs/HAC-HF3805G/HAC-HF3231E_Installation_20171121.pdf`, page 1
- `source-pdfs/HAC-HF3805G/HAC-HF3805G_Dimention_20171128.pdf`, page 1
- `source-pdfs/HAC-HF3805G/HDCVI_Box_Camera_Installation_Guide_V1.0.0-Eng.pdf`, pages 1-2

## Correction Made During Review

The HAC-HF3231E installation sheet was rechecked visually. The original generated notes had the mounting accessory IDs reversed. The corrected values are:

- Wall Mount: `PFB121W`
- Ceiling Mount: `PFB110W`

The generator and QA output were regenerated after this correction.

Several text questions also had their evidence support tightened after answer-level review:

- `qa-0031`
- `qa-0033`
- `qa-0040`
- `qa-0041`
- `qa-0049`
- `qa-0058`

## Review Artifacts

- `codex-review-report.json`: automated review summary.
- `codex-reviewed-final-report.json`: final Codex signoff summary after manual visual review.
- `full-source-pdf-read-report.json`: all-source-PDF page/read inventory generated during review.
- `evidence-audit.json`: evidence-level text/visual audit rows.
- `visual-render-manifest.json`: visual QA entries mapped to rendered PDF pages.
- `renders/`: rendered visual evidence pages used for manual review.
