#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
save_dir=${1:-saves/aion_01_2gpu}
cd "$repo_root"

uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Aion 01 requires exactly two CUDA devices, found {torch.cuda.device_count()}")
for index in range(2):
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f"cuda:{index} does not support BF16")
        print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
PY

if [[ -e "$save_dir/manifest.json" ]]; then
  echo "$save_dir already contains a world; use gol-run --resume explicitly" >&2
  exit 2
fi

exec env PYTHONUNBUFFERED=1 uv run gol-run \
  --new "$save_dir" \
  --config configs/run/aion_01_2gpu.yaml \
  --headless
