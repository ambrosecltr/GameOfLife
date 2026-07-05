#!/usr/bin/env bash
# Long-running stability soak: run a world headless, restart on crash, forever
# (or until you ctrl-C). Everything is resumable, so restarts lose at most one
# checkpoint interval.
#
#   scripts/soak.sh saves/soak_001 [--config configs/run/local_m1.yaml]
set -uo pipefail

SAVE=${1:?usage: soak.sh save_dir [gol-run args...]}
shift || true
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$(dirname "$SAVE")"
LOG="$SAVE.soak.log"

echo "soaking $SAVE (log: $LOG). ctrl-C to stop."
while true; do
  if [[ -f "$SAVE/manifest.json" ]]; then
    uv run gol-run "$SAVE" --resume --headless "$@" >>"$LOG" 2>&1
  else
    uv run gol-run "$SAVE" --new --headless "$@" >>"$LOG" 2>&1
  fi
  code=$?
  if [[ $code -eq 0 || $code -eq 130 ]]; then
    echo "run exited cleanly ($code); stopping soak."
    break
  fi
  echo "$(date '+%F %H:%M:%S') crashed (exit $code); resuming in 5s" | tee -a "$LOG"
  sleep 5
done
