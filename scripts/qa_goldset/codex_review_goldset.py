from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


VISUAL_MODALITIES = {"diagram", "image", "screenshot"}


@dataclass
class EvidenceAudit:
    qa_id: str
    evidence_index: int
    pdf: str
    page_number: int
    modality: str
    support_token_coverage: float | None
    status: str
    note: str


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9][A-Za-z0-9.+/@_-]{1,}", clean_text(text).lower()))


def support_coverage(support: str, page_text: str) -> float:
    support_tokens = token_set(support.replace("...", ""))
    page_tokens = token_set(page_text)
    if not support_tokens:
        return 0.0
    return len(support_tokens & page_tokens) / len(support_tokens)


def render_page(pdf_path: Path, page_idx: int, out_path: Path, zoom: float) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(out_path)


def audit_goldset(goldset_path: Path, out_dir: Path, *, render_zoom: float) -> dict[str, Any]:
    qas = json.loads(goldset_path.read_text())
    page_text_cache: dict[tuple[str, int], str] = {}
    audits: list[EvidenceAudit] = []
    render_manifest: list[dict[str, Any]] = []
    item_notes: dict[str, list[str]] = {}

    for qa in qas:
        for idx, evidence in enumerate(qa["evidence"]):
            pdf = evidence["pdf"]
            page_idx = int(evidence["page_idx"])
            modality = evidence["modality"]
            support = evidence.get("support", "")
            if modality in VISUAL_MODALITIES:
                safe_pdf = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(pdf).stem)
                render_path = out_dir / "renders" / f"{safe_pdf}_p{page_idx + 1}.png"
                render_page(Path(pdf), page_idx, render_path, render_zoom)
                render_manifest.append(
                    {
                        "qa_id": qa["id"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "pdf": pdf,
                        "page_number": page_idx + 1,
                        "render": str(render_path),
                        "support": support,
                    }
                )
                audits.append(
                    EvidenceAudit(
                        qa_id=qa["id"],
                        evidence_index=idx,
                        pdf=pdf,
                        page_number=page_idx + 1,
                        modality=modality,
                        support_token_coverage=None,
                        status="manual_visual_required",
                        note="Rendered page for Codex visual review.",
                    )
                )
                continue

            key = (pdf, page_idx)
            if key not in page_text_cache:
                doc = fitz.open(pdf)
                page_text_cache[key] = clean_text(doc[page_idx].get_text("text") or "")
            coverage = support_coverage(support, page_text_cache[key])
            status = "verified_text" if coverage >= 0.88 else "needs_text_review"
            note = f"Support token coverage against extracted page text: {coverage:.2%}."
            audits.append(
                EvidenceAudit(
                    qa_id=qa["id"],
                    evidence_index=idx,
                    pdf=pdf,
                    page_number=page_idx + 1,
                    modality=modality,
                    support_token_coverage=coverage,
                    status=status,
                    note=note,
                )
            )
            if status != "verified_text":
                item_notes.setdefault(qa["id"], []).append(note)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visual-render-manifest.json").write_text(json.dumps(render_manifest, indent=2, ensure_ascii=False))
    audit_rows = [audit.__dict__ for audit in audits]
    (out_dir / "evidence-audit.json").write_text(json.dumps(audit_rows, indent=2, ensure_ascii=False))

    counter = Counter(audit.status for audit in audits)
    unique_visual_pages = {
        (item["pdf"], item["page_number"])
        for item in render_manifest
    }
    report = {
        "goldset": str(goldset_path),
        "total_questions": len(qas),
        "total_evidence_items": len(audits),
        "status_counts": dict(sorted(counter.items())),
        "unique_visual_pages": len(unique_visual_pages),
        "visual_render_count": len(render_manifest),
        "text_items_needing_review": sorted(item_notes),
    }
    (out_dir / "codex-review-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", default="qa-goldset/source-pdfs-qa-200.json")
    parser.add_argument("--out-dir", default="qa-goldset/codex-review")
    parser.add_argument("--render-zoom", type=float, default=2.5)
    args = parser.parse_args()
    report = audit_goldset(Path(args.goldset), Path(args.out_dir), render_zoom=args.render_zoom)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
