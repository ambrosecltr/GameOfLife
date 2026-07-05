#!/usr/bin/env bash
# Pull a remote run's checkpoints, logs, and recordings home on a loop.
# Run this on the laptop while a cloud soak is in progress; a killed spot
# instance then costs at most one sync interval of artifacts.
#
#   scripts/sync_back.sh root@1.2.3.4 -p 22 saves/alpha [interval_seconds]
set -euo pipefail

HOST=${1:?usage: sync_back.sh user@host [-p port] save_dir [interval]}
shift
PORT=22
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PORT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
SAVE=${ARGS[0]:?save_dir required}
INTERVAL=${ARGS[1]:-600}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

while true; do
  rsync -az -e "ssh -p $PORT" --delete-after \
    "$HOST:~/GameOfLife/$SAVE/" "$REPO_ROOT/$SAVE/" \
    && echo "$(date '+%H:%M:%S') synced $SAVE" \
    || echo "$(date '+%H:%M:%S') sync failed (instance gone?); will retry"
  sleep "$INTERVAL"
done
