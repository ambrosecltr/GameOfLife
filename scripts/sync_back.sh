#!/usr/bin/env bash
# Pull a remote run's checkpoints, logs, and recordings home on a loop.
# Run this on the laptop while a cloud soak is in progress; a killed spot
# instance then costs at most one sync interval of artifacts.
#
# Dreamer brain checkpoints (~560MB per agent per checkpoint) are SKIPPED by
# default — the mirror is for metrics/events/world analysis, not resume. Pass
# --full to mirror brains too (do this once at round close if you want a
# locally resumable copy; per invariant 4 a mirror without brains is not one).
#
# Needs a direct TCP SSH endpoint — rsync cannot run through the interactive
# ssh.runpod.io proxy (PTY-only, no exec channel).
#
#   scripts/sync_back.sh root@1.2.3.4 -p 22 saves/alpha [interval_seconds] [--full]
set -euo pipefail

HOST=${1:?usage: sync_back.sh user@host [-p port] save_dir [interval] [--full]}
shift
PORT=22
FULL=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PORT="$2"; shift 2 ;;
    --full) FULL=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
SAVE=${ARGS[0]:?save_dir required}
INTERVAL=${ARGS[1]:-600}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EXCLUDES=()
# rsync protects excluded paths from --delete-after, so a --full pull's local
# brains survive later light syncs.
[[ $FULL -eq 1 ]] || EXCLUDES=(--exclude 'checkpoints/*/brains/')

while true; do
  rsync -az -e "ssh -p $PORT" --delete-after "${EXCLUDES[@]}" \
    "$HOST:~/GameOfLife/$SAVE/" "$REPO_ROOT/$SAVE/" \
    && echo "$(date '+%H:%M:%S') synced $SAVE" \
    || echo "$(date '+%H:%M:%S') sync failed (instance gone?); will retry"
  sleep "$INTERVAL"
done
