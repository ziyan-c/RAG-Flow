from __future__ import annotations

import argparse
from pathlib import Path


def highlight_pdf_bbox(
    input_pdf: str | Path,
    output_pdf: str | Path,
    page_idx: int,
    bbox_1000: list[float],
    color: tuple[float, float, float] = (1, 0, 0),
    padding: float = 15,
) -> None:
    import fitz

    doc = fitz.open(str(input_pdf))
    if page_idx >= len(doc):
        raise ValueError(f"PDF has {len(doc)} pages; cannot access page {page_idx}.")

    page = doc[page_idx]
    page_width = page.rect.width
    page_height = page.rect.height
    x0, y0, x1, y1 = bbox_1000
    rect = fitz.Rect(
        (x0 / 1000.0) * page_width - padding,
        (y0 / 1000.0) * page_height - padding,
        (x1 / 1000.0) * page_width + padding,
        (y1 / 1000.0) * page_height + padding,
    )
    page.draw_rect(rect, color=color, fill=color, fill_opacity=0.3, width=0)
    doc.save(str(output_pdf))
    doc.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Highlight a MinerU 0-1000 bbox on a PDF page.")
    parser.add_argument("input_pdf")
    parser.add_argument("output_pdf")
    parser.add_argument("--page-idx", type=int, default=0)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--padding", type=float, default=15)
    args = parser.parse_args(argv)

    highlight_pdf_bbox(args.input_pdf, args.output_pdf, args.page_idx, list(args.bbox), padding=args.padding)
    print(f"Wrote highlighted PDF to {args.output_pdf}")


if __name__ == "__main__":
    main()
