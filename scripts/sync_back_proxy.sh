#!/usr/bin/env bash
# Pull a remote run's save dir home through the RunPod interactive SSH proxy
# (ssh.runpod.io), which is PTY-only — no exec channel, so rsync/scp/sftp all
# fail. Works by tarring on the pod and streaming base64 through the PTY.
#
# Dreamer brain checkpoints (~560MB per agent per checkpoint) are always
# skipped — this mirror is for metrics/events/world analysis, not resume.
# Use sync_back.sh --full against a direct TCP SSH endpoint for brains.
#
#   scripts/sync_back_proxy.sh <podid>-<hash>@ssh.runpod.io saves/beta_09 [interval_seconds]
#
# interval_seconds=0 (default) does a single pull and exits.
set -euo pipefail

HOST=${1:?usage: sync_back_proxy.sh user@ssh.runpod.io save_dir [interval]}
SAVE=${2:?save_dir required}
INTERVAL=${3:-0}
KEY=${SYNC_KEY:-$HOME/.ssh/id_ed25519}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$SAVE")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pull_once() {
  printf 'cd ~/GameOfLife/%s/.. && tar czf /tmp/%s_light.tgz --exclude "%s/checkpoints/*/brains" %s && echo BEGIN64 && base64 /tmp/%s_light.tgz && echo END64\nexit\n' \
    "$SAVE" "$NAME" "$NAME" "$NAME" "$NAME" |
    ssh -tt -o ConnectTimeout=15 -i "$KEY" "$HOST" >"$TMP/dump.txt" 2>&1 || true
  # PTY output carries CRs and ANSI escapes; strip them, then decode the
  # payload between the markers (echoed command lines never match ^BEGIN64$).
  perl -pe 's/\e\][^\a]*\a//g; s/\e\[[0-9;?]*[a-zA-Z]//g; s/\r//g' "$TMP/dump.txt" |
    awk '/^BEGIN64$/{f=1;next} /^END64$/{f=0} f' | base64 -d >"$TMP/$NAME.tgz"
  [[ -s "$TMP/$NAME.tgz" ]] || { echo "$(date '+%H:%M:%S') pull failed (instance gone?)"; return 1; }
  tar xzf "$TMP/$NAME.tgz" -C "$(dirname "$REPO_ROOT/$SAVE")"
  echo "$(date '+%H:%M:%S') synced $SAVE ($(du -sh "$REPO_ROOT/$SAVE" | cut -f1))"
}

if [[ "$INTERVAL" -eq 0 ]]; then
  pull_once
else
  while true; do
    pull_once || true
    sleep "$INTERVAL"
  done
fi
