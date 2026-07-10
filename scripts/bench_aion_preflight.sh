#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_DIR GPU_HOURLY_COST" >&2
  exit 2
fi

output_dir=$1
gpu_hourly_cost=$2
mkdir -p "$output_dir"
started_at=$SECONDS
deadline_seconds=300

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required to enforce the five-minute pod gate" >&2
  exit 2
fi

check_deadline() {
  if (( SECONDS - started_at >= deadline_seconds )); then
    echo "preflight exceeded five minutes; reject this unit for the long run" >&2
    exit 3
  fi
}

run_before_deadline() {
  local remaining=$((deadline_seconds - (SECONDS - started_at)))
  if (( remaining <= 0 )); then
    check_deadline
  fi
  if timeout --foreground --kill-after=5s "${remaining}s" "$@"; then
    return 0
  else
    local rc=$?
    if (( rc == 124 || SECONDS - started_at >= deadline_seconds )); then
      echo "preflight exceeded five minutes; reject this unit for the long run" >&2
      return 3
    fi
    return "$rc"
  fi
}

nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader > "$output_dir/gpu.csv"
uv run python - <<'PY' > "$output_dir/torch.txt"
import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
print(f"capability={torch.cuda.get_device_capability()}")
PY

for precision in ieee_fp32 tf32 amp_bf16; do
  run_before_deadline uv run python scripts/bench_learn.py \
    --brain configs/brain/aion_01_s5.yaml \
    --devices cuda \
    --precision "$precision" \
    --updates 10 \
    --fill 4096 \
    --headroom 0.85 \
    --gpu-hourly-cost "$gpu_hourly_cost" \
    --json | tee "$output_dir/single-$precision.log"
  check_deadline

  run_before_deadline uv run python scripts/bench_contention.py \
    --brain configs/brain/aion_01_s5.yaml \
    --device cuda \
    --precision "$precision" \
    --brains 3 \
    --updates 10 \
    --fill 4096 \
    --headroom 0.85 \
    --gpu-hourly-cost "$gpu_hourly_cost" \
    --json | tee "$output_dir/contention-$precision.log"
  check_deadline
done

run_before_deadline uv run python scripts/bench_learn.py \
  --brain configs/brain/aion_01_s5.yaml \
  --devices cuda \
  --precision amp_bf16 \
  --updates 3 \
  --fill 4096 \
  --headroom 0.85 \
  --profile "$output_dir/profile" \
  --json | tee "$output_dir/profile.log"
check_deadline

runtime_save="$output_dir/runtime-save"
if [[ -e "$runtime_save" ]]; then
  echo "$runtime_save already exists; use a fresh output directory" >&2
  exit 2
fi
run_before_deadline uv run gol-run --new "$runtime_save" \
  --config configs/run/aion_01.yaml \
  --headless \
  --ticks 200 \
  --set checkpoint_interval_ticks=100 \
  --set observability.metrics_every_ticks=50 \
  --profile "$output_dir/profile/runtime.json"
check_deadline
