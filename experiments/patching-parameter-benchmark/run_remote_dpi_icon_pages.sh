#!/usr/bin/env bash
set -euo pipefail

cd /root/RAG-Flow
export RAG_FLOW_ENV_FILE=/root/RAG-Flow/.local/rag-flow.env
export PYTHONPATH=/root/RAG-Flow/src

PY=${RAG_FLOW_PIPELINE_PYTHON_BIN:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/python}
RAG=${RAG_FLOW_RAG_FLOW_BIN:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/rag-flow}
LLM_PY=${RAG_FLOW_SGLANG_PYTHON:-${RAG_FLOW_LLM_PYTHON_BIN:-/root/autodl-tmp/envs/rag-flow-llm/bin/python}}
BENCH_ROOT=${1:-/root/autodl-tmp/rag-flow-benchmarks/technical-manual/dpi_icon_pages_$(date +%Y%m%d_%H%M%S)}
RESULTS_JSONL=$BENCH_ROOT/results.jsonl
RESULTS_CSV=$BENCH_ROOT/results.csv

PATCH_CONCURRENCY=${RAG_FLOW_BENCH_ICON_CONCURRENCY:-10}
PATCH_BATCH_SIZE=${RAG_FLOW_BENCH_ICON_BATCH_SIZE:-140}
PATCH_MAX_NEW_TOKENS=${RAG_FLOW_BENCH_ICON_MAX_NEW_TOKENS:-8000}
PATCH_TIMEOUT=${RAG_FLOW_BENCH_ICON_TIMEOUT:-120}

mkdir -p "$BENCH_ROOT" "$BENCH_ROOT/inputs" "$BENCH_ROOT/runs" "$BENCH_ROOT/meta"
touch "$RESULTS_JSONL"

{
  echo "timestamp=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "cpu_threads=$(nproc 2>/dev/null || true)"
  echo "memory="
  free -h 2>/dev/null || true
  echo "gpu_query="
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader,nounits 2>/dev/null || true
  echo "pipeline_python=$PY"
  "$PY" -V 2>&1 || true
  echo "pipeline_python_packages="
  "$PY" -c 'import importlib.metadata as m
for p in ("rag-flow", "pdf2image", "pillow", "openai", "pymupdf"):
    try:
        print(f"{p}={m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p}=not-installed")' 2>/dev/null || true
  echo "llm_python=$LLM_PY"
  "$LLM_PY" -V 2>&1 || true
  echo "llm_python_packages="
  "$LLM_PY" -c 'import importlib.metadata as m
for p in ("sglang", "torch", "torchvision", "nvidia-cudnn-cu12", "openai", "pillow"):
    try:
        print(f"{p}={m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p}=not-installed")' 2>/dev/null || true
  echo "sglang_process="
  pgrep -af 'sglang|launch_server' || true
  echo "serve_dry_run="
  "$RAG" serve llm-sglang --dry-run || true
} > "$BENCH_ROOT/meta/machine.txt" 2>&1

"$PY" - "$BENCH_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from rag_flow.config import AppConfig
from rag_flow.preprocessing import small_icons as si

bench_root = Path(sys.argv[1])
config = AppConfig.from_env()
source_pdf = config.paths.source_pdf
content = json.loads(config.paths.content_json.read_text(encoding="utf-8"))
selected_pages = [312, 319, 434]
page_map = {page_idx: idx for idx, page_idx in enumerate(selected_pages)}
chrome_pages = {312: 313, 319: 320, 434: 435}
subset = []
for global_idx, block in enumerate(content):
    if not isinstance(block, dict):
        continue
    page_idx = int(block.get("page_idx", -1))
    if page_idx not in page_map:
        continue
    copied = dict(block)
    copied["benchmark_global_idx"] = global_idx
    copied["benchmark_original_page_idx"] = page_idx
    copied["benchmark_chrome_page"] = chrome_pages[page_idx]
    copied["page_idx"] = page_map[page_idx]
    subset.append(copied)

input_path = bench_root / "inputs" / "chrome_pages_313_320_435_content_list.json"
input_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
selected_pdf = bench_root / "inputs" / "chrome_pages_313_320_435.pdf"
temp_dir = bench_root / "inputs" / "pdf-pages"
temp_dir.mkdir(parents=True, exist_ok=True)
parts = []
for chrome_page in [313, 320, 435]:
    out = temp_dir / f"page_{chrome_page}.pdf"
    subprocess.run(
        ["pdfseparate", "-f", str(chrome_page), "-l", str(chrome_page), str(source_pdf), str(out)],
        check=True,
    )
    parts.append(out)
subprocess.run(["pdfunite", *map(str, parts), str(selected_pdf)], check=True)

fields = empty = ignored = no_bbox = no_fields = 0
type_counts = {}
field_counts = {}
for block in subset:
    block_type = block.get("type", "unknown")
    type_counts[block_type] = type_counts.get(block_type, 0) + 1
    if block.get("type") in si.IGNORE_TYPES:
        ignored += 1
        continue
    if "bbox" not in block:
        no_bbox += 1
        continue
    keys = si._patch_field_keys(block)
    if not keys:
        no_fields += 1
        continue
    for key in keys:
        text = si._join(block.get(key, "")).strip()
        if text:
            fields += 1
            field_counts[key] = field_counts.get(key, 0) + 1
        else:
            empty += 1

metadata = {
    "source_content_json": str(config.paths.content_json),
    "source_pdf": str(source_pdf),
    "input_json": str(input_path),
    "selected_pdf": str(selected_pdf),
    "chrome_pages": [313, 320, 435],
    "original_page_indices": selected_pages,
    "remapped_page_indices": page_map,
    "blocks": len(subset),
    "candidate_fields": fields,
    "empty_fields": empty,
    "ignored_blocks": ignored,
    "no_bbox_blocks": no_bbox,
    "no_text_field_blocks": no_fields,
    "type_counts": type_counts,
    "field_counts": field_counts,
}
(bench_root / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
PY

INPUT=$("$PY" - "$BENCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]) / "dataset.json").read_text())["input_json"])
PY
)
PDF=$("$PY" - "$BENCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]) / "dataset.json").read_text())["selected_pdf"])
PY
)

run_case() {
  local name=$1
  local dpi=$2
  local run_dir=$BENCH_ROOT/runs/$name
  local output=$run_dir/${name}_PATCHED.json
  local patch_log=$run_dir/patch.log
  local gpu_log=$run_dir/gpu.csv

  mkdir -p "$run_dir"
  echo "== case $name dpi=$dpi concurrency=$PATCH_CONCURRENCY batch=$PATCH_BATCH_SIZE =="
  printf '{"stage":"dpi_icon_pages","name":"%s","dpi":%s,"concurrency":%s,"batch_size":%s,"input_json":"%s","output_json":"%s"}\n' \
    "$name" "$dpi" "$PATCH_CONCURRENCY" "$PATCH_BATCH_SIZE" "$INPUT" "$output" > "$run_dir/metadata.json"

  (
    while true; do
      printf '%s,' "$(date +%s.%N)"
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
        --format=csv,noheader,nounits | head -1
      sleep 2
    done
  ) > "$gpu_log" &
  local monitor_pid=$!

  local start_ns end_ns status
  start_ns=$(date +%s%N)
  set +e
  "$RAG" patch \
    --input "$INPUT" \
    --pdf "$PDF" \
    --output "$output" \
    --dpi "$dpi" \
    --batch-size "$PATCH_BATCH_SIZE" \
    --concurrency "$PATCH_CONCURRENCY" \
    --checkpoint-interval 0 \
    --invalid-retry-limit 0 \
    --max-new-tokens "$PATCH_MAX_NEW_TOKENS" \
    --request-timeout "$PATCH_TIMEOUT" \
    --no-resume \
    --no-patching-view \
    > "$patch_log" 2>&1
  status=$?
  end_ns=$(date +%s%N)
  set -e

  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  "$PY" - "$name" "$dpi" "$PATCH_CONCURRENCY" "$PATCH_BATCH_SIZE" "$status" "$start_ns" "$end_ns" "$patch_log" "$gpu_log" "$output" "$RESULTS_JSONL" <<'PY'
import csv
import json
import pathlib
import re
import statistics
import sys

name = sys.argv[1]
dpi, concurrency, batch_size, status = map(int, sys.argv[2:6])
start_ns, end_ns = map(int, sys.argv[6:8])
patch_log = pathlib.Path(sys.argv[8])
gpu_log = pathlib.Path(sys.argv[9])
output = pathlib.Path(sys.argv[10])
results_jsonl = pathlib.Path(sys.argv[11])
elapsed = (end_ns - start_ns) / 1_000_000_000
log = patch_log.read_text(errors="replace") if patch_log.exists() else ""

def find(pattern, cast=float, default=None):
    match = re.search(pattern, log)
    return cast(match.group(1)) if match else default

rows = []
if gpu_log.exists():
    with gpu_log.open() as f:
        for row in csv.reader(f):
            if len(row) >= 7:
                try:
                    rows.append(
                        {
                            "timestamp": float(row[0]),
                            "memory_used_mib": float(row[1].strip()),
                            "memory_total_mib": float(row[2].strip()),
                            "gpu_util_pct": float(row[3].strip()),
                            "memory_util_pct": float(row[4].strip()),
                            "power_w": float(row[5].strip()),
                            "temperature_c": float(row[6].strip()),
                        }
                    )
                except ValueError:
                    pass

requests = find(r"requests submitted: (\d+)", int, 0)
record = {
    "stage": "dpi_icon_pages",
    "name": name,
    "dpi": dpi,
    "concurrency": concurrency,
    "batch_size": batch_size,
    "status": status,
    "elapsed_sec": round(elapsed, 3),
    "requests_submitted": requests,
    "requests_per_sec": round(requests / elapsed, 4) if elapsed and requests else 0,
    "blocks_seen": find(r"blocks seen: (\d+)", int, 0),
    "fields_seen": find(r"fields seen: (\d+)", int, 0),
    "checked": find(r"checked: (\d+)", int, 0),
    "patched": find(r"patched: (\d+)", int, 0),
    "no_missing": find(r"no missing: (\d+)", int, 0),
    "invalid_fallbacks": find(r"invalid fallbacks: (\d+)", int, 0),
    "invalid_rejected": find(r"invalid rejected: (\d+)", int, 0),
    "llm_batches": find(r"LLM batches: (\d+)", int, 0),
    "output_json": str(output),
    "patch_log": str(patch_log),
    "gpu_log": str(gpu_log),
}
if rows:
    for key in ("gpu_util_pct", "memory_used_mib", "memory_util_pct", "power_w", "temperature_c"):
        values = [r[key] for r in rows]
        record[f"{key}_avg"] = round(statistics.mean(values), 3)
        record[f"{key}_max"] = round(max(values), 3)
    record["gpu_samples"] = len(rows)
else:
    record["gpu_samples"] = 0

with results_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
raise SystemExit(status)
PY
}

run_case d200 200 || true
run_case d250 250 || true
run_case d300 300 || true

"$PY" experiments/patching-parameter-benchmark/score_dpi_icon_pages.py "$BENCH_ROOT" > "$BENCH_ROOT/quality-summary.json"

"$PY" - "$RESULTS_JSONL" "$RESULTS_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
keys = [
    "stage",
    "name",
    "dpi",
    "concurrency",
    "batch_size",
    "status",
    "elapsed_sec",
    "requests_submitted",
    "requests_per_sec",
    "checked",
    "patched",
    "no_missing",
    "invalid_fallbacks",
    "invalid_rejected",
    "gpu_util_pct_avg",
    "gpu_util_pct_max",
    "memory_used_mib_avg",
    "memory_used_mib_max",
    "power_w_avg",
    "power_w_max",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in keys})
PY

echo "Benchmark root: $BENCH_ROOT"
echo "Results: $RESULTS_CSV"
echo "Quality: $BENCH_ROOT/quality-scores.csv"
