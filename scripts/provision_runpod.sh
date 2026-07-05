#!/usr/bin/env bash
# Provision a rented GPU box (RunPod/Vast/any ubuntu+CUDA host) and ship the
# project to it. Spot-instance discipline: checkpoints rsync home on a cadence,
# so a killed instance costs at most one checkpoint interval.
#
#   scripts/provision_runpod.sh root@1.2.3.4 -p 22 [saves/alpha]
#
# Then on the box:
#   tmux new -s world
#   cd ~/GameOfLife && uv run gol-run saves/alpha --resume --config configs/run/cloud_gpu.yaml
#
# And from the laptop, keep checkpoints flowing home:
#   scripts/sync_back.sh root@1.2.3.4 -p 22 saves/alpha
set -euo pipefail

HOST=${1:?usage: provision_runpod.sh user@host [-p port] [save_dir]}
shift
PORT=22
SAVE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PORT="$2"; shift 2 ;;
    *) SAVE="$1"; shift ;;
  esac
done
SSH="ssh -p $PORT"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> installing uv + rsync on $HOST"
$SSH "$HOST" 'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync tmux); command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

echo "==> shipping repo"
rsync -az -e "ssh -p $PORT" \
  --exclude .venv --exclude .git --exclude saves --exclude '*.rrd' \
  --exclude .mypy_cache --exclude .ruff_cache --exclude .pytest_cache \
  "$REPO_ROOT/" "$HOST:~/GameOfLife/"

if [[ -n "$SAVE" ]]; then
  echo "==> shipping $SAVE (world + brains resume where they left off)"
  rsync -az -e "ssh -p $PORT" "$REPO_ROOT/$SAVE/" "$HOST:~/GameOfLife/$SAVE/"
fi

echo "==> installing dependencies (CUDA torch resolves automatically on linux)"
$SSH "$HOST" 'cd ~/GameOfLife && ~/.local/bin/uv sync 2>&1 | tail -2 && ~/.local/bin/uv run python -c "import torch; print(\"cuda:\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")"'

cat <<EOF

Provisioned. Next:
  $SSH $HOST
  tmux new -s world
  cd ~/GameOfLife && uv run gol-run ${SAVE:-saves/alpha} ${SAVE:+--resume} --headless --config configs/run/cloud_gpu.yaml
Stream the viewer home instead of --headless: forward Rerun over SSH, or record
.rrd files (rotated per sim-6h) and sync them back with scripts/sync_back.sh.
EOF
