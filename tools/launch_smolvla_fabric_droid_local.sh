#!/usr/bin/env bash
# Direct local-GPU launcher for the immutable Fabric-DROID v001 SmolVLA run.
# It keeps setup visible in the invoking terminal, then moves conversion and
# training into a durable nohup process with logs and a PID below STORAGE_ROOT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROID_ROOT="${DROID_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/suhang/datasets2}"
RAW_ROOT="${RAW_ROOT:-$STORAGE_ROOT/frabric_pi}"
MANIFEST="${MANIFEST:-$DROID_ROOT/curation/exports/fabric_droid_v001/manifest.parquet}"
RUN_NAME="${RUN_NAME:-fabric_droid_v001_smolvla_20k}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESUME="${RESUME:-0}"

SETUP="$SCRIPT_DIR/setup_smolvla_env.sh"
RUNNER="$SCRIPT_DIR/run_smolvla_fabric_droid.sh"
PYTHON_BIN="$STORAGE_ROOT/smolvla/.venv/bin/python"
LOG_DIR="$STORAGE_ROOT/smolvla/runlogs"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOG_DIR/$RUN_NAME.pid"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail "STEPS must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be a positive integer"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || fail "NUM_WORKERS must be a non-negative integer"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fail "RESUME must be 0 or 1"
[[ -d "$RAW_ROOT" ]] || fail "raw Fabric-DROID data not found: $RAW_ROOT"
[[ -f "$MANIFEST" ]] || fail "curated v001 manifest not found: $MANIFEST"
[[ -f "$SETUP" && -f "$RUNNER" ]] || fail "SmolVLA helper scripts are missing from $SCRIPT_DIR"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable; a local NVIDIA GPU driver is required"
nvidia-smi -L >/dev/null 2>&1 || fail "the local NVIDIA GPU is not available to this shell"

mkdir -p "$LOG_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  fail "a local SmolVLA process is already alive (PID $(cat "$PID_FILE")); log: $LOG_FILE"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[INFO] Creating the local SmolVLA environment below $STORAGE_ROOT"
  STORAGE_ROOT="$STORAGE_ROOT" bash "$SETUP"
fi

echo "[INFO] Starting local SmolVLA bootstrap in the background."
echo "[INFO] Log: $LOG_FILE"
nohup env \
  DROID_ROOT="$DROID_ROOT" \
  STORAGE_ROOT="$STORAGE_ROOT" \
  RAW_ROOT="$RAW_ROOT" \
  MANIFEST="$MANIFEST" \
  RUN_NAME="$RUN_NAME" \
  STEPS="$STEPS" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  RESUME="$RESUME" \
  SAVE_FREQ=2000 \
  bash "$RUNNER" > "$LOG_FILE" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$PID_FILE"

sleep 3
if ! kill -0 "$train_pid" 2>/dev/null; then
  tail -n 80 "$LOG_FILE" >&2 || true
  fail "local SmolVLA bootstrap exited early; see $LOG_FILE"
fi

echo "[OK] Local SmolVLA bootstrap PID: $train_pid"
echo "[OK] Follow progress: tail -f $LOG_FILE"
echo "[OK] Checkpoints: $STORAGE_ROOT/smolvla/runs/$RUN_NAME/checkpoints/ (every 2000 steps)"
