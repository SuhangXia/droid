#!/usr/bin/env bash
set -euo pipefail

# Continuous Fabric-DROID LoRA training (2k save interval) for sc02 deployment.

OPENPI_ROOT="${OPENPI_ROOT:-/workspace/openpi}"
PYTHON_BIN="${PYTHON_BIN:-/root/rivermind-data/venvs/openpi/bin/python}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/suhang/datasets2/rivermind-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$STORAGE_ROOT/lerobot}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$STORAGE_ROOT/checkpoints/fabric_droid_v001_lora_10k}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$STORAGE_ROOT/assets}"
LOG_DIR="${LOG_DIR:-$STORAGE_ROOT/runlogs/openpi}"
EXP_NAME="${EXP_NAME:-fabric_droid_v001_lora_sc02}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"

mkdir -p "$CHECKPOINT_BASE_DIR" "$ASSETS_BASE_DIR" "$LOG_DIR" "$HF_LEROBOT_HOME"
cd "$OPENPI_ROOT"

export HF_HOME="${HF_HOME:-$STORAGE_ROOT/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$STORAGE_ROOT/cache/jax}"
mkdir -p "$HF_HOME" "$JAX_COMPILATION_CACHE_DIR"

export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$PYTHON_BIN" "$OPENPI_ROOT/scripts/train.py" fabric_droid_lora_10k \
  --exp-name "$EXP_NAME" \
  --data.repo-id local/fabric_droid_v001 \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size 8 \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 50 \
  --keep-period 1000000000 \
  --no-wandb-enabled \
  --resume
