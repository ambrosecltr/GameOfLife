#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_CONFIG SAVE_DIR FINITE_TICKS" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
run_config=$1
save_dir=$2
ticks=$3
cd "$repo_root"

if [[ ! "$ticks" =~ ^[1-9][0-9]*$ ]]; then
  echo "FINITE_TICKS must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "$run_config" ]]; then
  echo "run config does not exist: $run_config" >&2
  exit 2
fi
if ! git ls-files --error-unmatch "$run_config" >/dev/null 2>&1; then
  echo "run config must be tracked before launch: $run_config" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to launch from a dirty worktree; commit the exact source first" >&2
  exit 2
fi

uv run pytest -q \
  packages/brains/tests/test_swift.py::test_policy_standard_deviation_is_smoothly_bounded \
  packages/brains/tests/test_swift.py::test_reinforce_sample_is_detached_but_log_prob_still_trains_policy \
  packages/brains/tests/test_dreamer.py::test_imagination_includes_regulated_wellbeing_in_viability_affect \
  packages/brains/tests/test_dreamer.py::test_imagination_continuation_obeys_predicted_coma_and_death_boundaries

uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Aion requires exactly two CUDA devices, found {torch.cuda.device_count()}")
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

args=(
  --new "$save_dir"
  --config "$run_config"
  --headless
  --ticks "$ticks"
)

exec env PYTHONUNBUFFERED=1 uv run gol-run "${args[@]}"
