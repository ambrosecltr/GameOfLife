#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_DIR POD_HOURLY_COST" >&2
  exit 2
fi

output_dir=$1
pod_hourly_cost=$2
mkdir -p "$output_dir"

uv run python - <<'PY' | tee "$output_dir/hardware.txt"
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two CUDA devices, found {torch.cuda.device_count()}")

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        print(
            f"gpu={index} name={torch.cuda.get_device_name(index)} "
            f"capability={torch.cuda.get_device_capability(index)} "
            f"memory_bytes={torch.cuda.get_device_properties(index).total_memory} "
            f"bf16_supported={torch.cuda.is_bf16_supported()}"
        )
PY

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader | tee "$output_dir/gpu.csv"

uv run python scripts/bench_contention.py \
  --brain configs/brain/aion_01_s5.yaml \
  --devices cuda:0 cuda:1 \
  --precision amp_bf16 \
  --brains 2 \
  --updates 20 \
  --fill 4096 \
  --headroom 0.85 \
  --action-deadline-ms 250 \
  --gpu-hourly-cost "$pod_hourly_cost" \
  --json | tee "$output_dir/contention-amp_bf16.log"

uv run python - "$output_dir/contention-amp_bf16.log" <<'PY'
import json
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text().splitlines()
result = json.loads(lines[-1])
failures = []

if result.get("devices") != "cuda:0,cuda:1":
    failures.append(f"unexpected device assignment {result.get('devices')!r}")
if result.get("precision") != "amp_bf16":
    failures.append(f"unexpected precision {result.get('precision')!r}")
if int(float(result.get("brains", 0))) != 2:
    failures.append(f"expected two brains, found {result.get('brains')!r}")

aggregate_safe_rate = float(result["sustainable_ticks_per_second_with_headroom"])
individual_safe_rate = float(
    result["slowest_brain_sustainable_ticks_per_second_with_headroom"]
)
safe_rate = min(aggregate_safe_rate, individual_safe_rate)
if safe_rate < 25.0:
    failures.append(f"safe tick rate {safe_rate:.2f} is below 25")
if float(result.get("action_deadline_misses", 0.0)) != 0.0:
    failures.append("action deadline misses were observed")
if float(result.get("action_p95_seconds", 1.0)) >= 0.250:
    failures.append("action p95 reached the 250 ms deadline")

for index in range(2):
    reserved = float(result[f"peak_vram_reserved_mb_cuda_{index}"])
    total = float(result[f"total_vram_mb_cuda_{index}"])
    if reserved > total * 0.85:
        failures.append(f"cuda:{index} reserved {reserved:.0f} MiB, leaving under 15% headroom")

if failures:
    raise SystemExit("two-GPU preflight failed: " + "; ".join(failures))
print(f"two-GPU preflight passed: {safe_rate:.2f} safe ticks/s with 0 deadline misses")
PY
