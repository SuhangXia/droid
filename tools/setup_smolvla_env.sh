#!/usr/bin/env bash
# Build an isolated, GPU-capable LeRobot/SmolVLA environment on the training host.
#
# It deliberately lives below STORAGE_ROOT so it neither changes the existing
# OpenPI environment nor writes outside the user-approved storage location.
# The defaults target the CUDA 12.8 / RTX 5090 server used for the prior Pi0.5
# run. Set INSTALL_TORCH=0 when a compatible CUDA PyTorch is already present.

set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/home/suhang/datasets2}"
VENV_DIR="${VENV_DIR:-$STORAGE_ROOT/smolvla/.venv}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$STORAGE_ROOT/smolvla/lerobot}"
LEROBOT_REF="${LEROBOT_REF:-v0.4.4}"
# LeRobot v0.4.4 supports Python >=3.10. Override this only when the server
# has several interpreters and a specific one should own the isolated venv.
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1 || fail "Python not found: $BOOTSTRAP_PYTHON"
"$BOOTSTRAP_PYTHON" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is too old; LeRobot v0.4.4 requires Python >= 3.10. "
        "Set BOOTSTRAP_PYTHON to a compatible interpreter."
    )
PY
command -v git >/dev/null 2>&1 || fail "git is required to fetch LeRobot"
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is required to write LeRobot video shards"

mkdir -p "$(dirname "$VENV_DIR")" "$(dirname "$LEROBOT_ROOT")"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade \
    --index-url "$TORCH_INDEX_URL" \
    'torch>=2.7,<2.8' \
    'torchvision>=0.22,<0.23'
fi

if [[ ! -d "$LEROBOT_ROOT/.git" ]]; then
  git clone --depth 1 --branch "$LEROBOT_REF" \
    https://github.com/huggingface/lerobot.git "$LEROBOT_ROOT"
else
  # Fetching the user-selected ref directly supports both release tags (the
  # default) and an explicitly requested branch/commit.
  git -C "$LEROBOT_ROOT" fetch --depth 1 origin "$LEROBOT_REF"
  git -C "$LEROBOT_ROOT" checkout --detach FETCH_HEAD
fi

# h5py is needed only by the Fabric-DROID reader; the SmolVLA extra supplies
# transformers, accelerate, safetensors, and the policy implementation.
"$PYTHON_BIN" -m pip install --upgrade -e "$LEROBOT_ROOT[smolvla]" h5py

"$PYTHON_BIN" - <<'PY'
import cv2
import h5py
import torch
import transformers
import accelerate
import lerobot
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

print(f"LeRobot: {lerobot.__version__}")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformers: {transformers.__version__}; Accelerate: {accelerate.__version__}")
print(f"OpenCV: {cv2.__version__}; h5py: {h5py.__version__}")
print(f"SmolVLA config: {SmolVLAConfig.__name__}")
PY

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not usable in this environment. Check the NVIDIA driver and the selected PyTorch CUDA wheel."
    )
PY

echo "[OK] SmolVLA environment ready: $VENV_DIR"
echo "[OK] Use this interpreter: $PYTHON_BIN"
