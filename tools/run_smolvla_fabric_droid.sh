#!/usr/bin/env bash
# Materialize the exact curated Fabric-DROID export as LeRobot v3 and fine-tune
# lerobot/smolvla_base. Checkpoints are deliberately retained every 2,000 steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROID_ROOT="${DROID_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/suhang/datasets2}"
PYTHON_BIN="${PYTHON_BIN:-$STORAGE_ROOT/smolvla/.venv/bin/python}"

RAW_ROOT="${RAW_ROOT:-$STORAGE_ROOT/frabric_pi}"
MANIFEST="${MANIFEST:-$DROID_ROOT/curation/exports/fabric_droid_v001/manifest.parquet}"
CONVERTER="${CONVERTER:-$SCRIPT_DIR/convert_curator_manifest_to_smolvla_v3.py}"
REPO_ID="${REPO_ID:-local/fabric_droid_v001_smolvla_v3}"
DATASET_ROOT="${DATASET_ROOT:-$STORAGE_ROOT/smolvla/fabric_droid_v001_v3}"

RUN_NAME="${RUN_NAME:-fabric_droid_v001_smolvla_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_ROOT/smolvla/runs/$RUN_NAME}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
RESUME="${RESUME:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ "$SAVE_FREQ" == "2000" ]] || fail "SAVE_FREQ is locked to 2000 for this run, got: $SAVE_FREQ"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail "STEPS must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be a positive integer"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || fail "NUM_WORKERS must be a non-negative integer"
[[ -x "$PYTHON_BIN" ]] || fail "SmolVLA Python is not executable: $PYTHON_BIN. Run tools/setup_smolvla_env.sh first."
[[ -f "$CONVERTER" ]] || fail "converter not found: $CONVERTER"
[[ -f "$MANIFEST" ]] || fail "curator manifest not found: $MANIFEST"
[[ -d "$RAW_ROOT" ]] || fail "raw Fabric-DROID root not found: $RAW_ROOT"

mkdir -p "$STORAGE_ROOT/smolvla" "$(dirname "$OUTPUT_DIR")"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/smolvla/cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$STORAGE_ROOT/smolvla/lerobot_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

AVAILABLE_GIB="$(df -Pk "$STORAGE_ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
if [[ "$AVAILABLE_GIB" -lt "$MIN_FREE_GIB" ]]; then
  fail "only ${AVAILABLE_GIB} GiB free below $STORAGE_ROOT; need at least ${MIN_FREE_GIB} GiB because all 2k checkpoints are retained"
fi
echo "[INFO] ${AVAILABLE_GIB} GiB free under $STORAGE_ROOT"

"$PYTHON_BIN" - <<'PY'
import h5py
import torch
import lerobot
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

print(f"[INFO] LeRobot {lerobot.__version__}")
print(f"[INFO] PyTorch {torch.__version__}; CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start a CPU SmolVLA run.")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] Policy config: {SmolVLAConfig.__name__}")
PY

if [[ ! -e "$DATASET_ROOT" ]]; then
  echo "[INFO] Building a new LeRobot v3 dataset from the curated Pi0.5 export."
  "$PYTHON_BIN" "$CONVERTER" \
    --manifest "$MANIFEST" \
    --raw-root "$RAW_ROOT" \
    --output-root "$DATASET_ROOT" \
    --repo-id "$REPO_ID" \
    --video-codec h264
else
  echo "[INFO] Reusing existing dataset after strict validation: $DATASET_ROOT"
fi

"$PYTHON_BIN" "$CONVERTER" \
  --validate-only \
  --output-root "$DATASET_ROOT" \
  --repo-id "$REPO_ID"

# The public SmolVLA base model was trained with camera1/2/3. Keep the new
# dataset semantically named and map exterior/wrist into stable checkpoint slots.
RENAME_MAP='{"observation.images.exterior_1_left":"observation.images.camera1","observation.images.wrist_left":"observation.images.camera2"}'

if [[ "$RESUME" == "1" ]]; then
  CONFIG_PATH="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
  [[ -f "$CONFIG_PATH" ]] || fail "RESUME=1 but no resumable checkpoint config at $CONFIG_PATH"
  echo "[INFO] Resuming SmolVLA. STEPS is the desired total update count: $STEPS"
  exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
    --config_path="$CONFIG_PATH" \
    --resume=true \
    --output_dir="$OUTPUT_DIR" \
    --steps="$STEPS" \
    --save_freq=2000
fi

[[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists: $OUTPUT_DIR (set RESUME=1 to continue it)"

echo "[INFO] Starting new SmolVLA fine-tune: $RUN_NAME"
echo "[INFO] Dataset: $DATASET_ROOT ($REPO_ID)"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Steps: $STEPS; batch size: $BATCH_SIZE; checkpoint cadence: every 2000 steps"

exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --rename_map="$RENAME_MAP" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --num_workers="$NUM_WORKERS" \
  --log_freq="$LOG_FREQ" \
  --eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false
