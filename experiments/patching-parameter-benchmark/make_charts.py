#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def read_rows() -> list[dict[str, str]]:
    with (ROOT / "clean-results-100p.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_icon_page_quality_rows() -> list[dict[str, str]]:
    path = ROOT / "data-icon-pages" / "quality-scores.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400", fill: str = "#111827") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(text)}</text>"
    )


def line_chart(
    filename: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    data: list[tuple[float, float, str]],
    annotations: list[dict[str, object]],
    x_ticks: list[float] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    x_transform: str = "linear",
) -> None:
    width, height = 900, 470
    left, right, top, bottom = 78, 34, 86, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    if x_transform == "log":
        tx = lambda v: math.log(v)
    else:
        tx = lambda v: v
    min_x, max_x = min(tx(x) for x in xs), max(tx(x) for x in xs)
    min_y = y_min if y_min is not None else min(ys) - 0.15
    max_y = y_max if y_max is not None else max(ys) + 0.25

    def sx(x: float) -> float:
        return left + (tx(x) - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    if x_ticks is None:
        x_ticks = xs

    y_step = 0.5
    y_ticks = []
    y = math.ceil(min_y / y_step) * y_step
    while y <= max_y + 1e-9:
        y_ticks.append(round(y, 2))
        y += y_step

    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    out.append(svg_text(24, 32, title, 20, weight="700"))
    out.append(svg_text(24, 56, subtitle, 12, fill="#4b5563"))
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#e5e7eb"/>')

    for yt in y_ticks:
        y_pos = sy(yt)
        out.append(f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + plot_w}" y2="{y_pos:.1f}" stroke="#e5e7eb"/>')
        out.append(svg_text(left - 10, y_pos + 4, f"{yt:.1f}", 11, anchor="end", fill="#4b5563"))

    for xt in x_ticks:
        x_pos = sx(xt)
        out.append(f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top + plot_h}" stroke="#eef2f7"/>')
        label = str(int(xt)) if abs(xt - round(xt)) < 1e-9 else f"{xt:g}"
        out.append(svg_text(x_pos, top + plot_h + 23, label, 11, anchor="middle", fill="#4b5563"))

    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    out.append(svg_text(left + plot_w / 2, height - 24, x_label, 12, anchor="middle", weight="600"))
    out.append(f'<text transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#111827">{html.escape(y_label)}</text>')

    points = [(sx(x), sy(y), label, x, y) for x, y, label in data]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in points)
    out.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="{poly}"/>')
    for x, y, label, raw_x, raw_y in points:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')

    for ann in annotations:
        x = float(ann["x"])
        yv = float(ann["y"])
        text = str(ann["text"])
        dx = float(ann.get("dx", 0))
        dy = float(ann.get("dy", -36))
        color = str(ann.get("color", "#dc2626"))
        px, py = sx(x), sy(yv)
        txp, typ = px + dx, py + dy
        out.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{txp:.1f}" y2="{typ + 6:.1f}" stroke="{color}" stroke-width="1.4" stroke-dasharray="3,3"/>')
        pad_x, pad_y = 7, 5
        approx_w = max(68, len(text) * 6.4)
        out.append(f'<rect x="{txp - pad_x:.1f}" y="{typ - 15:.1f}" width="{approx_w + 2 * pad_x:.1f}" height="24" rx="5" fill="#ffffff" stroke="{color}" stroke-width="1.2"/>')
        out.append(svg_text(txp, typ + 2, text, 11, fill=color, weight="700"))

    out.append("</svg>")
    (ASSETS / filename).write_text("\n".join(out), encoding="utf-8")


def grouped_bar_chart(filename: str, title: str, subtitle: str, categories: list[str], series: list[tuple[str, list[float], str]], y_label: str) -> None:
    width, height = 900, 430
    left, right, top, bottom = 72, 36, 84, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max(max(vals) for _, vals, _ in series) + 2

    def sy(y: float) -> float:
        return top + (max_y - y) / max_y * plot_h

    group_w = plot_w / len(categories)
    bar_w = group_w / (len(series) + 2)
    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    out.append(svg_text(24, 32, title, 20, weight="700"))
    out.append(svg_text(24, 56, subtitle, 12, fill="#4b5563"))
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#e5e7eb"/>')
    for yt in range(0, int(max_y) + 1, 2):
        y_pos = sy(yt)
        out.append(f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + plot_w}" y2="{y_pos:.1f}" stroke="#e5e7eb"/>')
        out.append(svg_text(left - 10, y_pos + 4, str(yt), 11, anchor="end", fill="#4b5563"))
    for i, cat in enumerate(categories):
        cx = left + i * group_w + group_w / 2
        out.append(svg_text(cx, top + plot_h + 24, cat, 12, anchor="middle", fill="#111827", weight="600"))
        for j, (name, vals, color) in enumerate(series):
            x = left + i * group_w + (j + 1) * bar_w
            y = sy(vals[i])
            h = top + plot_h - y
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{h:.1f}" fill="{color}" rx="4"/>')
            out.append(svg_text(x + bar_w * 0.41, y - 7, f"{vals[i]:.0f}", 11, anchor="middle", fill=color, weight="700"))
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    out.append(f'<text transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="600" text-anchor="middle" fill="#111827">{html.escape(y_label)}</text>')
    legend_x = width - right - 280
    for i, (name, _, color) in enumerate(series):
        y = 28 + i * 22
        out.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{color}" rx="3"/>')
        out.append(svg_text(legend_x + 22, y, name, 12, fill="#111827"))
    out.append("</svg>")
    (ASSETS / filename).write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    valid = [r for r in rows if r["status"] == "0" and float(r["requests_submitted"]) > 0]
    concurrency = sorted([r for r in valid if r["stage"] == "concurrency"], key=lambda r: float(r["concurrency"]))
    batch = sorted([r for r in valid if r["stage"] == "batch"], key=lambda r: float(r["batch_size"]))
    dpi = sorted([r for r in valid if r["stage"] == "dpi"], key=lambda r: float(r["dpi"]))

    line_chart(
        "chart-concurrency-throughput.svg",
        "Concurrency Sweep",
        "Fixed DPI=250 and batch_size=15; 987 submitted requests per run",
        "concurrency",
        "requests / second",
        [(to_float(r, "concurrency"), to_float(r, "requests_per_sec"), r["name"]) for r in concurrency],
        [
            {"x": 10, "y": 4.5692, "text": "observed high 4.57 @ c=10", "dx": -154, "dy": -45},
            {"x": 14, "y": 4.1080, "text": "decline @ c=14", "dx": -54, "dy": 52, "color": "#b45309"},
        ],
        x_ticks=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15],
        y_min=2.5,
        y_max=4.85,
    )

    line_chart(
        "chart-batch-throughput.svg",
        "Batch Size Sweep",
        "Fixed DPI=250 and concurrency=10; x-axis uses log spacing for readability",
        "batch_size (log scale)",
        "requests / second",
        [(to_float(r, "batch_size"), to_float(r, "requests_per_sec"), r["name"]) for r in batch],
        [
            {"x": 140, "y": 5.3492, "text": "observed high 5.35 @ b=140", "dx": -176, "dy": -44},
            {"x": 200, "y": 4.9991, "text": "drops at b=200", "dx": -136, "dy": 48, "color": "#b45309"},
            {"x": 3, "y": 3.1702, "text": "too many batches", "dx": 34, "dy": -36, "color": "#6b7280"},
        ],
        x_ticks=[3, 6, 9, 12, 18, 24, 36, 48, 60, 80, 100, 140, 200],
        y_min=3.0,
        y_max=5.65,
        x_transform="log",
    )

    line_chart(
        "chart-dpi-throughput.svg",
        "DPI Throughput Sweep",
        "Fixed concurrency=10 and batch_size=140",
        "rasterization DPI",
        "requests / second",
        [(to_float(r, "dpi"), to_float(r, "requests_per_sec"), r["name"]) for r in dpi],
        [
            {"x": 200, "y": 5.8218, "text": "fastest 5.82 @ 200", "dx": 28, "dy": -42},
            {"x": 300, "y": 4.5296, "text": "slowest @ 300", "dx": -130, "dy": 48, "color": "#b45309"},
        ],
        x_ticks=[200, 250, 300],
        y_min=4.3,
        y_max=6.05,
    )

    grouped_bar_chart(
        "chart-dpi-quality-case.svg",
        "Single-Page DPI Quality Case",
        "Manual scoring on page_idx=80; lower false positives are better",
        ["200 DPI", "250 DPI", "300 DPI"],
        [
            ("strict hits", [1, 3, 4], "#2563eb"),
            ("false positives", [8, 4, 4], "#dc2626"),
            ("missed targets", [3, 3, 3], "#f59e0b"),
        ],
        "count",
    )

    icon_quality = sorted(read_icon_page_quality_rows(), key=lambda r: float(r["dpi"]))
    if icon_quality:
        line_chart(
            "chart-dpi-icon-pages-quality.svg",
            "Icon-Heavy DPI Quality Probe",
            "Real VLM runs on target technical manual reference pages 313, 320, and 435; higher proxy is better",
            "rasterization DPI",
            "quality proxy",
            [(to_float(r, "dpi"), to_float(r, "quality_proxy"), r["name"]) for r in icon_quality],
            [
                {"x": 300, "y": 7.75, "text": "highest proxy @ 300", "dx": -148, "dy": -45},
                {"x": 250, "y": 1.0, "text": "false positives hurt", "dx": -98, "dy": 48, "color": "#b45309"},
            ],
            x_ticks=[200, 250, 300],
            y_min=0,
            y_max=8.5,
        )


if __name__ == "__main__":
    main()
