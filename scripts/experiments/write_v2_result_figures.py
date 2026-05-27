from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence


WORK_ROOT = Path("thesis-v2/experiments/v2-final")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else default
    except ValueError:
        return default


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text(
    x: float,
    y: float,
    text: Any,
    *,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{_esc(text)}</text>'
    )


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}</style>',
    ]


def _horizontal_bars(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    label_key: str,
    value_key: str,
    extra_key: str | None = None,
    x_label: str = "mean answer score",
    color: str = "#4274e8",
) -> None:
    if not rows:
        return
    width = 1180
    row_h = 42
    height = 150 + row_h * len(rows)
    left, right, top, bottom = 270, 90, 92, 58
    chart_w = width - left - right
    values = [float(row[value_key]) for row in rows]
    v_min = min(values)
    v_max = max(values)
    x_min = max(0.0, min(v_min - 0.08, v_min * 0.96))
    x_max = min(5.0, max(v_max + 0.08, v_max * 1.04))
    if x_max <= x_min:
        x_max = x_min + 1.0

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * chart_w

    parts = _svg_start(width, height)
    parts += [
        _text(width / 2, 34, title, size=25, weight="700"),
        _text(width / 2, 58, subtitle, size=13, fill="#44506a"),
        f'<line x1="{left}" y1="{top - 16}" x2="{left}" y2="{height - bottom}" stroke="#9aa8bf"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9aa8bf"/>',
    ]
    for tick in range(5):
        value = x_min + (x_max - x_min) * tick / 4
        xx = x(value)
        parts.append(f'<line x1="{xx:.1f}" y1="{top - 16}" x2="{xx:.1f}" y2="{height - bottom}" stroke="#edf1f7"/>')
        parts.append(_text(xx, height - bottom + 24, f"{value:.2f}", size=11))

    for idx, row in enumerate(rows):
        yy = top + idx * row_h
        value = float(row[value_key])
        parts.append(_text(left - 14, yy + 18, row[label_key], size=12, anchor="end"))
        parts.append(
            f'<rect x="{left:.1f}" y="{yy:.1f}" width="{max(2, x(value) - left):.1f}" '
            f'height="25" rx="5" fill="{color}"/>'
        )
        parts.append(_text(x(value) + 8, yy + 18, f"{value:.3f}", size=12, anchor="start", weight="700"))
        if extra_key:
            parts.append(_text(width - right + 10, yy + 18, row.get(extra_key, ""), size=11, anchor="start", fill="#526079"))
    parts.append(_text(width / 2, height - 14, x_label, size=12, fill="#44506a"))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _grouped_bars(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    label_key: str,
    series: Sequence[tuple[str, str, str]],
    y_label: str,
) -> None:
    if not rows:
        return
    width, height = 1240, 620
    left, right, top, bottom = 90, 45, 88, 135
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(float(row[key]) for row in rows for key, _name, _color in series)
    y_max = max(1.0, max_value * 1.12)

    def y(value: float) -> float:
        return top + (y_max - value) / y_max * chart_h

    parts = _svg_start(width, height)
    parts += [
        _text(width / 2, 34, title, size=25, weight="700"),
        _text(width / 2, 58, subtitle, size=13, fill="#44506a"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#9aa8bf"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#9aa8bf"/>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#edf1f7"/>')
        parts.append(_text(left - 12, yy + 4, f"{value:.2f}", size=11, anchor="end"))
    slot_w = chart_w / len(rows)
    bar_w = min(48, slot_w / (len(series) + 1))
    for i, row in enumerate(rows):
        center = left + slot_w * i + slot_w / 2
        start = center - (len(series) * bar_w + (len(series) - 1) * 8) / 2
        for j, (key, _name, color) in enumerate(series):
            value = float(row[key])
            xx = start + j * (bar_w + 8)
            parts.append(
                f'<rect x="{xx:.1f}" y="{y(value):.1f}" width="{bar_w:.1f}" '
                f'height="{top + chart_h - y(value):.1f}" rx="5" fill="{color}"/>'
            )
            parts.append(_text(xx + bar_w / 2, y(value) - 8 - j * 2, f"{value:.3f}", size=11, weight="700"))
        label = str(row[label_key])
        for line_idx, line in enumerate(label.split("\\n")):
            parts.append(_text(center, top + chart_h + 28 + line_idx * 16, line, size=11))
    legend_x = left
    for key, name, color in series:
        parts.append(f'<rect x="{legend_x}" y="70" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(_text(legend_x + 18, 81, name, size=12, anchor="start"))
        legend_x += 170
    parts.append(_text(left, top - 18, y_label, size=12, anchor="start", fill="#44506a"))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _line_chart(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    color: str = "#4274e8",
) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row[x_key]))
    width, height = 1100, 560
    left, right, top, bottom = 90, 50, 86, 92
    chart_w = width - left - right
    chart_h = height - top - bottom
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min = max(0.0, min(ys) - 0.08)
    y_max = min(5.0, max(ys) + 0.08)
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * chart_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_h

    parts = _svg_start(width, height)
    parts += [
        _text(width / 2, 34, title, size=25, weight="700"),
        _text(width / 2, 58, subtitle, size=13, fill="#44506a"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#9aa8bf"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#9aa8bf"/>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        yy = sy(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#edf1f7"/>')
        parts.append(_text(left - 12, yy + 4, f"{value:.2f}", size=11, anchor="end"))
    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"/>')
        parts.append(_text(sx(x), sy(y) - 12, f"{y:.3f}", size=11, weight="700"))
        parts.append(_text(sx(x), top + chart_h + 26, f"{x:g}", size=11))
    parts.append(_text(width / 2, height - 24, x_label, size=12, fill="#44506a"))
    parts.append(_text(left, top - 18, y_label, size=12, anchor="start", fill="#44506a"))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _row_by_id(rows: Iterable[dict[str, str]], run_id: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("run_id") == run_id), None)


def _as_result_row(row: dict[str, str], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "run_id": row["run_id"],
        "mean_score": _float(row, "mean_score"),
        "high_quality_rate": _float(row, "high_quality_rate"),
        "low_score_rate": _float(row, "low_score_rate"),
        "avg_context_tokens": _float(row, "avg_context_tokens"),
        "avg_total_tokens": _float(row, "avg_total_tokens"),
        "retrieval_side_efficiency": _float(row, "retrieval_side_efficiency"),
        "latency": _float(row, "avg_total_latency_seconds"),
    }


def main() -> None:
    rows = _read_rows(WORK_ROOT / "tables" / "run_summary.csv")
    figures = WORK_ROOT / "figures"
    tables = WORK_ROOT / "tables"

    chunk_ids = [
        ("800/100/100", "chunk-coarse-m800-o100-n100"),
        ("1200/100/100", "chunk-coarse-m1200-o100-n100"),
        ("1500/150/150", "chunk-coarse-m1500-o150-n150"),
        ("2000/200/200", "chunk-coarse-m2000-o200-n200"),
        ("2500/250/200", "chunk-coarse-m2500-o250-n200"),
        ("3000/300/200", "chunk-coarse-m3000-o300-n200"),
        ("4000/400/200", "chunk-coarse-m4000-o400-n200"),
        ("5000/500/200", "chunk-coarse-m5000-o500-n200"),
    ]
    chunk_rows = [
        _as_result_row(row, label)
        for label, run_id in chunk_ids
        if (row := _row_by_id(rows, run_id))
    ]
    _write_csv(tables / "chunking_key_results.csv", chunk_rows)
    _horizontal_bars(
        figures / "chunking_key_scores.svg",
        chunk_rows,
        title="Chunking profile sweep",
        subtitle="固定 text-only retrieval baseline；50 QA 三轮 Codex 审阅平均分",
        label_key="label",
        value_key="mean_score",
        extra_key="avg_context_tokens",
        x_label="mean answer score; right label = avg retrieved-context tokens",
        color="#4e79e6",
    )

    retrieval_ids = [
        ("tiny\\n3k k50 top5", "retrieval-sweep-cap3k"),
        ("compact\\ntop10", "retrieval-interaction-cap10-rk80-top10-rrf10-r1"),
        ("default\\n10k k80 top20", "retrieval-interaction-cap10-rk80-top20-rrf10-r1"),
        ("ratio-pruned\\n10k top10 r=0.4", "retrieval-interaction-cap10-rk150-top10-rrf10-ratio0p4"),
        ("visual naive\\n10k w=1", "retrieval-interaction-cap10-rk150-top20-naive-w1"),
        ("visual bbox\\n10k w=3", "retrieval-interaction-cap10-rk150-top20-bbox-w3"),
        ("high context\\n16k top80", "retrieval-sweep-cap16k"),
        ("very high\\n24k top80", "retrieval-sweep-cap24k"),
    ]
    retrieval_rows = [
        _as_result_row(row, label)
        for label, run_id in retrieval_ids
        if (row := _row_by_id(rows, run_id))
    ]
    _write_csv(tables / "retrieval_key_results.csv", retrieval_rows)
    _grouped_bars(
        figures / "retrieval_key_scores.svg",
        retrieval_rows,
        title="Retrieval configurations: quality and cost",
        subtitle="50 QA；蓝色为回答评分，绿色为每千 retrieved-context token 效率",
        label_key="label",
        series=(
            ("mean_score", "mean score", "#4e79e6"),
            ("retrieval_side_efficiency", "score / 1k ctx tokens", "#22a06b"),
        ),
        y_label="score",
    )

    max_rows = []
    for run_id in [
        "answering-max-max1000",
        "answering-max-max2000",
        "answering-max-max4000",
        "answering-max-max6000",
        "answering-max-max8000",
    ]:
        row = _row_by_id(rows, run_id)
        if row:
            out = _as_result_row(row, run_id.replace("answering-max-max", ""))
            out["max_tokens"] = int(out["label"])
            max_rows.append(out)
    _write_csv(tables / "answering_max_token_results.csv", max_rows)
    _line_chart(
        figures / "answering_max_tokens.svg",
        max_rows,
        title="Answering max_new_tokens sweep",
        subtitle="固定最佳 text retrieval；输出上限越大不必然更慢，但 8000 分数最高",
        x_key="max_tokens",
        y_key="mean_score",
        x_label="answer max tokens",
        y_label="mean answer score",
    )

    feature_ids = [
        ("text\\nfixed index", "answering-feature-text-fixed"),
        ("real images\\n359 image_url", "answering-feature-images-fixed"),
        ("thinking on", "answering-feature-thinking-on"),
        ("text\\nmax8000 sweep", "answering-feature-text-max8000"),
    ]
    feature_rows = [
        _as_result_row(row, label)
        for label, run_id in feature_ids
        if (row := _row_by_id(rows, run_id))
    ]
    _write_csv(tables / "answering_feature_results.csv", feature_rows)
    _grouped_bars(
        figures / "answering_feature_scores.svg",
        feature_rows,
        title="Answering feature ablation",
        subtitle="fixed-index real image input trades quality/token efficiency for slightly higher high-score rate",
        label_key="label",
        series=(
            ("mean_score", "mean score", "#4e79e6"),
            ("latency", "latency seconds", "#f28c38"),
        ),
        y_label="score / seconds",
    )

    final200 = [row for row in rows if row["run_id"].startswith("final200-")]
    if final200:
        labels = {
            "final200-default-text": "default\\ntext",
            "final200-default-images": "old images flag\\n0 image_url",
            "final200-default-images-fixed": "default\\nreal images",
            "final200-visual-naive-w1": "visual\\nnaive w=1",
            "final200-compact-top10": "compact\\ntop10",
            "final200-compact-ratio0p4": "compact\\nratio 0.4",
            "final200-tiny-cap3-top5": "tiny\\n3k top5",
            "final200-high-context-16k": "high\\n16k",
            "final200-high-context-24k": "very high\\n24k",
            "final200-visual-bbox-w3": "visual\\nbbox w=3",
            "final200-thinking-on": "thinking\\non",
        }
        final_rows = [_as_result_row(row, labels.get(row["run_id"], row["run_id"])) for row in final200]
        final_rows.sort(key=lambda row: row["mean_score"], reverse=True)
        _write_csv(tables / "final200_results.csv", final_rows)
        _grouped_bars(
            figures / "final200_scores.svg",
            final_rows,
            title="Final 200-QA verification",
            subtitle="代表性端到端配置；同一 200 QA set，三轮 Codex 审阅平均分",
            label_key="label",
            series=(
                ("mean_score", "mean score", "#4e79e6"),
                ("high_quality_rate", "score >= 4", "#22a06b"),
                ("low_score_rate", "score <= 2", "#ef4e4e"),
            ),
            y_label="score / rate",
        )


if __name__ == "__main__":
    main()
