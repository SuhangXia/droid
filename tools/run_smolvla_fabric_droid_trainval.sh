#!/usr/bin/env bash
# Leakage-free Fabric-DROID SmolVLA fine-tuning: source-level train/val split,
# 2k checkpoint cadence, and post-training offline validation of every checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROID_ROOT="${DROID_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/suhang/datasets2}"
PYTHON_BIN="${PYTHON_BIN:-$STORAGE_ROOT/smolvla/.venv/bin/python}"

SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-$STORAGE_ROOT/smolvla/fabric_droid_v001_v3}"
SOURCE_REPO_ID="${SOURCE_REPO_ID:-local/fabric_droid_v001_smolvla_v3}"
SPLIT_FILE="${SPLIT_FILE:-$DROID_ROOT/curation/exports/fabric_droid_v001_smolvla_trainval_v1/split.json}"
SPLIT_ROOT="${SPLIT_ROOT:-$STORAGE_ROOT/smolvla/fabric_droid_v001_source_split_v1}"
TRAIN_DATASET_ROOT="$SPLIT_ROOT/train"
VAL_DATASET_ROOT="$SPLIT_ROOT/val"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-${SOURCE_REPO_ID}_train}"
VAL_REPO_ID="${VAL_REPO_ID:-${SOURCE_REPO_ID}_val}"
MATERIALIZER="$SCRIPT_DIR/materialize_fabric_droid_smolvla_train_val.py"
VALIDATOR="$SCRIPT_DIR/evaluate_smolvla_offline_val.py"

RUN_NAME="${RUN_NAME:-fabric_droid_v001_train90_val10_smolvla_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/smolvla/runs/$RUN_NAME}"
STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
RESUME="${RESUME:-0}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

# Pin the exact SmolVLA revision already downloaded for the completed 20k run.
HF_HOME="${HF_HOME:-$STORAGE_ROOT/smolvla/cache/huggingface}"
BASE_REPO_ID="${BASE_REPO_ID:-lerobot/smolvla_base}"
BASE_REVISION="${BASE_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
DEFAULT_BASE_POLICY_PATH="$HF_HOME/hub/models--lerobot--smolvla_base/snapshots/$BASE_REVISION"
BASE_POLICY_PATH="${BASE_POLICY_PATH:-$DEFAULT_BASE_POLICY_PATH}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ "$SAVE_FREQ" == "2000" ]] || fail "SAVE_FREQ is locked to 2000 for this run, got: $SAVE_FREQ"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail "STEPS must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be a positive integer"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || fail "NUM_WORKERS must be a non-negative integer"
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]] || fail "WARMUP_STEPS must be a non-negative integer"
[[ "$VAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "VAL_BATCH_SIZE must be a positive integer"
[[ "$VAL_NUM_WORKERS" =~ ^[0-9]+$ ]] || fail "VAL_NUM_WORKERS must be a non-negative integer"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fail "RESUME must be 0 or 1"
[[ "$RUN_VALIDATION" == "0" || "$RUN_VALIDATION" == "1" ]] || fail "RUN_VALIDATION must be 0 or 1"
[[ "$VAL_MAX_BATCHES" =~ ^[0-9]+$ ]] || fail "VAL_MAX_BATCHES must be a non-negative integer"
[[ -x "$PYTHON_BIN" ]] || fail "SmolVLA Python is not executable: $PYTHON_BIN"
[[ -d "$SOURCE_DATASET_ROOT" ]] || fail "source LeRobot v3 dataset not found: $SOURCE_DATASET_ROOT"
[[ -f "$SPLIT_FILE" ]] || fail "source-level split file not found: $SPLIT_FILE"
[[ -f "$MATERIALIZER" && -f "$VALIDATOR" ]] || fail "train/val helper scripts are missing from $SCRIPT_DIR"

mkdir -p "$STORAGE_ROOT/smolvla" "$(dirname "$OUTPUT_DIR")"
export HF_HOME
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$STORAGE_ROOT/smolvla/lerobot_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

AVAILABLE_GIB="$(df -Pk "$STORAGE_ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
if [[ "$AVAILABLE_GIB" -lt "$MIN_FREE_GIB" ]]; then
  fail "only ${AVAILABLE_GIB} GiB free below $STORAGE_ROOT; need at least ${MIN_FREE_GIB} GiB for 30 retained checkpoints"
fi
echo "[INFO] ${AVAILABLE_GIB} GiB free under $STORAGE_ROOT"

"$PYTHON_BIN" - <<'PY'
import torch
import lerobot

print(f"[INFO] LeRobot {lerobot.__version__}")
print(f"[INFO] PyTorch {torch.__version__}; CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start a CPU SmolVLA run.")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
PY

if [[ ! -f "$BASE_POLICY_PATH/config.json" ]]; then
  [[ "$BASE_POLICY_PATH" == "$DEFAULT_BASE_POLICY_PATH" ]] || fail "BASE_POLICY_PATH has no config.json: $BASE_POLICY_PATH"
  echo "[INFO] Downloading pinned base policy $BASE_REPO_ID@$BASE_REVISION"
  "$PYTHON_BIN" - "$BASE_REPO_ID" "$BASE_REVISION" "$HF_HOME" <<'PY'
from huggingface_hub import snapshot_download
import sys

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]))
PY
fi
[[ -f "$BASE_POLICY_PATH/config.json" ]] || fail "pinned SmolVLA policy was not materialized: $BASE_POLICY_PATH"

if [[ ! -e "$SPLIT_ROOT" ]]; then
  echo "[INFO] Materializing independent train/val roots with separate stats.json files."
  "$PYTHON_BIN" "$MATERIALIZER" \
    --source-root "$SOURCE_DATASET_ROOT" \
    --source-repo-id "$SOURCE_REPO_ID" \
    --split-file "$SPLIT_FILE" \
    --output-root "$SPLIT_ROOT"
else
  echo "[INFO] Reusing existing materialized train/val roots after validation: $SPLIT_ROOT"
  "$PYTHON_BIN" "$MATERIALIZER" \
    --source-root "$SOURCE_DATASET_ROOT" \
    --source-repo-id "$SOURCE_REPO_ID" \
    --split-file "$SPLIT_FILE" \
    --output-root "$SPLIT_ROOT" \
    --validate-only
fi

if [[ "$RESUME" == "1" ]]; then
  CONFIG_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  [[ -f "$CONFIG_PATH" ]] || fail "RESUME=1 but no resumable checkpoint config at $CONFIG_PATH"
  echo "[INFO] Resuming from the latest checkpoint; desired total steps: $STEPS"
  "$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
    --config_path="$CONFIG_PATH" \
    --resume=true \
    --output_dir="$OUTPUT_DIR" \
    --steps="$STEPS" \
    --save_freq=2000
else
  [[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists: $OUTPUT_DIR (set RESUME=1 to continue it)"
  RUN_SPEC="$OUTPUT_DIR.run_spec.json"
  "$PYTHON_BIN" - "$RUN_SPEC" "$SPLIT_FILE" "$BASE_POLICY_PATH" "$BASE_REPO_ID" "$BASE_REVISION" "$STEPS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_spec = Path(sys.argv[1])
split_file = Path(sys.argv[2])
base_policy_path = Path(sys.argv[3])
base_repo_id = sys.argv[4]
base_revision = sys.argv[5]
steps = int(sys.argv[6])

digest = hashlib.sha256(split_file.read_bytes()).hexdigest()
payload = {
    "schema": "fabric-droid-smolvla-trainval-run-v1",
    "split_file": str(split_file),
    "split_file_sha256": digest,
    "base_policy_repo_id": base_repo_id,
    "base_policy_revision": base_revision,
    "base_policy_local_snapshot": str(base_policy_path),
    "steps": steps,
    "checkpoint_cadence_steps": 2000,
}
temporary = run_spec.with_name(f".{run_spec.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(run_spec)
PY
  echo "[INFO] Starting new train-only SmolVLA fine-tune: $RUN_NAME"
  echo "[INFO] Train root: $TRAIN_DATASET_ROOT ($TRAIN_REPO_ID)"
  echo "[INFO] Val root:   $VAL_DATASET_ROOT ($VAL_REPO_ID; evaluated after training)"
  echo "[INFO] Steps: $STEPS; batch size: $BATCH_SIZE; checkpoints: every 2000 steps"
  "$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
    --policy.path="$BASE_POLICY_PATH" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
    --policy.scheduler_decay_steps="$STEPS" \
    --dataset.repo_id="$TRAIN_REPO_ID" \
    --dataset.root="$TRAIN_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --rename_map='{"observation.images.exterior_1_left":"observation.images.camera1","observation.images.wrist_left":"observation.images.camera2"}' \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --num_workers="$NUM_WORKERS" \
    --log_freq="$LOG_FREQ" \
    --eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=2000 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$RUN_NAME" \
    --seed=20260822 \
    --wandb.enable=false
fi

# In LeRobot v0.4.4, eval_freq refers to simulator/robot-environment rollout,
# not held-out offline BC loss. Evaluate the saved checkpoints explicitly after
# the trainer has closed its CUDA resources.
if [[ "$RUN_VALIDATION" == "1" ]]; then
  echo "[INFO] Computing deterministic held-out VAL loss for every saved checkpoint."
  "$PYTHON_BIN" "$VALIDATOR" \
    --checkpoints-dir "$OUTPUT_DIR/checkpoints" \
    --dataset-root "$VAL_DATASET_ROOT" \
    --repo-id "$VAL_REPO_ID" \
    --output "$OUTPUT_DIR/validation/val_loss_by_checkpoint.json" \
    --batch-size "$VAL_BATCH_SIZE" \
    --num-workers "$VAL_NUM_WORKERS" \
    --max-batches "$VAL_MAX_BATCHES" \
    --seed 20260822 \
    --device cuda \
    --overwrite
fi

echo "[OK] Train/val run complete: $OUTPUT_DIR"
