#!/usr/bin/env bash
set -euo pipefail

cd /root/RAG-Flow
export RAG_FLOW_ENV_FILE=/root/RAG-Flow/.local/rag-flow.env
export PYTHONPATH=/root/RAG-Flow/src

PY=${RAG_FLOW_PIPELINE_PYTHON_BIN:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/python}
RAG=${RAG_FLOW_RAG_FLOW_BIN:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/rag-flow}
LLM_PY=${RAG_FLOW_SGLANG_PYTHON:-${RAG_FLOW_LLM_PYTHON_BIN:-/root/autodl-tmp/envs/rag-flow-llm/bin/python}}
BENCH_ROOT=${1:-/root/autodl-tmp/rag-flow-benchmarks/technical-manual/page50_150_$(date +%Y%m%d_%H%M%S)}
RESULTS_JSONL=$BENCH_ROOT/results.jsonl
RESULTS_CSV=$BENCH_ROOT/results.csv

mkdir -p "$BENCH_ROOT" "$BENCH_ROOT/inputs" "$BENCH_ROOT/runs" "$BENCH_ROOT/meta"
touch "$RESULTS_JSONL"

{
  echo "timestamp=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "os_release="
  cat /etc/os-release 2>/dev/null || true
  echo "cpu_model=$(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || true)"
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
  echo "patch_dry_run="
  "$RAG" patch --dry-run || true
} > "$BENCH_ROOT/meta/machine.txt" 2>&1

"$PY" - "$BENCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from rag_flow.config import AppConfig
from rag_flow.preprocessing import small_icons as si

bench_root = Path(sys.argv[1])
config = AppConfig.from_env()
content = json.loads(config.paths.content_json.read_text(encoding="utf-8"))
subset = [b for b in content if isinstance(b, dict) and 50 <= int(b.get("page_idx", 0)) < 150]
input_path = bench_root / "inputs" / "technical_manual_page_idx_50_149_content_list.json"
input_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

fields = empty = ignored = no_bbox = no_fields = 0
type_counts = {}
field_counts = {}
pages = set()
for block in subset:
    pages.add(int(block.get("page_idx", 0)))
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
    "source_pdf": str(config.paths.source_pdf),
    "input_json": str(input_path),
    "page_idx_start_inclusive": 50,
    "page_idx_end_exclusive": 150,
    "pages_present": sorted(pages),
    "page_count": len(pages),
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

meta = json.loads((Path(sys.argv[1]) / "dataset.json").read_text())
print(meta["input_json"])
PY
)
PDF=$("$PY" - "$BENCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads((Path(sys.argv[1]) / "dataset.json").read_text())
print(meta["source_pdf"])
PY
)

run_case() {
  local stage=$1
  local name=$2
  local dpi=$3
  local concurrency=$4
  local batch=$5
  local run_dir=$BENCH_ROOT/runs/$name
  local output=$run_dir/${name}_PATCHED.json
  local patch_log=$run_dir/patch.log
  local gpu_log=$run_dir/gpu.csv

  mkdir -p "$run_dir"
  if "$PY" - "$RESULTS_JSONL" "$stage" "$name" <<'PY'
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
stage = sys.argv[2]
name = sys.argv[3]
for line in jsonl.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("stage") == stage and row.get("name") == name and row.get("status") == 0:
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo "SKIP case $name already completed"
    return
  fi

  echo "== case $name stage=$stage dpi=$dpi concurrency=$concurrency batch=$batch =="
  printf '{"stage":"%s","name":"%s","dpi":%s,"concurrency":%s,"batch_size":%s,"input_json":"%s","output_json":"%s"}\n' \
    "$stage" "$name" "$dpi" "$concurrency" "$batch" "$INPUT" "$output" > "$run_dir/metadata.json"

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
    --batch-size "$batch" \
    --concurrency "$concurrency" \
    --checkpoint-interval 0 \
    --no-resume \
    --no-patching-view \
    > "$patch_log" 2>&1
  status=$?
  end_ns=$(date +%s%N)
  set -e

  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  "$PY" - "$stage" "$name" "$dpi" "$concurrency" "$batch" "$status" "$start_ns" "$end_ns" "$patch_log" "$gpu_log" "$output" "$RESULTS_JSONL" <<'PY'
import csv
import json
import pathlib
import re
import statistics
import sys

stage, name = sys.argv[1], sys.argv[2]
dpi, concurrency, batch_size, status = map(int, sys.argv[3:7])
start_ns, end_ns = map(int, sys.argv[7:9])
patch_log = pathlib.Path(sys.argv[9])
gpu_log = pathlib.Path(sys.argv[10])
output = pathlib.Path(sys.argv[11])
results_jsonl = pathlib.Path(sys.argv[12])
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
    "stage": stage,
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
    "invalid_retries": find(r"invalid retries: (\d+)", int, 0),
    "invalid_fallbacks": find(r"invalid fallbacks: (\d+)", int, 0),
    "invalid_rejected": find(r"invalid rejected: (\d+)", int, 0),
    "llm_batches": find(r"LLM batches: (\d+)", int, 0),
    "checkpoints_written": find(r"checkpoints written: (\d+)", int, 0),
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
print("SUMMARY " + json.dumps(record, ensure_ascii=False), flush=True)
PY

  if [[ "$status" != "0" ]]; then
    echo "case $name failed with status $status" >&2
  fi
}

choose_best() {
  local stage=$1
  "$PY" - "$RESULTS_JSONL" "$stage" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
candidates = [
    r
    for r in rows
    if r["stage"] == sys.argv[2] and r["status"] == 0 and r["requests_submitted"] > 0
]
print(json.dumps(max(candidates, key=lambda r: r["requests_per_sec"])))
PY
}

for c in 1 2 3 4 5 6 7 8 10 12 14 15; do
  run_case concurrency c${c}_d250_b15 250 "$c" 15
done
BEST_CONCURRENCY=$(choose_best concurrency | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["concurrency"])')
echo "BEST_CONCURRENCY=$BEST_CONCURRENCY"

for b in 3 6 9 12 15 18 24 30 36 48 60 80 100 140 200; do
  run_case batch c${BEST_CONCURRENCY}_d250_b${b} 250 "$BEST_CONCURRENCY" "$b"
done
BEST_BATCH=$(choose_best batch | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["batch_size"])')
echo "BEST_BATCH=$BEST_BATCH"

for dpi in 200 250 300; do
  run_case dpi c${BEST_CONCURRENCY}_d${dpi}_b${BEST_BATCH} "$dpi" "$BEST_CONCURRENCY" "$BEST_BATCH"
done
BEST_DPI=$(choose_best dpi | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["dpi"])')
echo "BEST_DPI_THROUGHPUT=$BEST_DPI"

"$PY" - "$RESULTS_JSONL" "$RESULTS_CSV" "$BENCH_ROOT/summary.json" "$BEST_CONCURRENCY" "$BEST_BATCH" "$BEST_DPI" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl, csv_path, summary_path = map(Path, sys.argv[1:4])
best_concurrency, best_batch, best_dpi = map(int, sys.argv[4:7])
rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
fieldnames = [
    "stage",
    "name",
    "dpi",
    "concurrency",
    "batch_size",
    "status",
    "elapsed_sec",
    "requests_submitted",
    "requests_per_sec",
    "blocks_seen",
    "fields_seen",
    "checked",
    "patched",
    "no_missing",
    "invalid_retries",
    "invalid_fallbacks",
    "invalid_rejected",
    "llm_batches",
    "checkpoints_written",
    "gpu_util_pct_avg",
    "gpu_util_pct_max",
    "memory_used_mib_avg",
    "memory_used_mib_max",
    "memory_util_pct_avg",
    "memory_util_pct_max",
    "power_w_avg",
    "power_w_max",
    "temperature_c_avg",
    "temperature_c_max",
    "gpu_samples",
    "output_json",
    "patch_log",
    "gpu_log",
]
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
summary = {
    "best_concurrency_by_throughput": best_concurrency,
    "best_batch_by_throughput": best_batch,
    "best_dpi_by_throughput": best_dpi,
    "runs": len(rows),
    "results_csv": str(csv_path),
    "results_jsonl": str(jsonl),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "BENCH_ROOT=$BENCH_ROOT"
