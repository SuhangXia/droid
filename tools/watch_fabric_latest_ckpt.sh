#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/home/suhang/datasets2/rivermind-data}"
CKPT_ROOT="${CKPT_ROOT:-$STORAGE_ROOT/checkpoints/fabric_droid_lora_10k/fabric_droid_v001_lora_10k}"
LATEST_LINK="$CKPT_ROOT/latest"

while true; do
  latest="$(find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\n' 2>/dev/null | sort -n | tail -n 1 || true)"
  if [[ -n "$latest" && -d "$CKPT_ROOT/$latest" ]]; then
    ln -sfn "$CKPT_ROOT/$latest" "$LATEST_LINK"
  fi
  sleep 15
done
