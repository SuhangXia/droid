#!/usr/bin/env bash
set -euo pipefail

SCRIPT_VERSION="sc02_5090_redeploy_and_train_v15"
echo "[INFO] ${SCRIPT_VERSION} loaded."
echo "[INFO] Script path: $(readlink -f "$0")"

# Purpose:
# 1) Sync dataset from ah01 -> sc02 directly on sc02 host (server-to-server rsync).
# 2) Start LoRA training on sc02 with SAVE_INTERVAL default 2000 and keep running until PID is alive.

SC02_HOST="root@sc02-ssh.gpuhome.cc"
SC02_PORT=30102
AH01_HOST="root@ah01-ssh.gpuhome.cc"
AH01_PORT=30163
KEY_PATH="${HOME}/.ssh/id_ed25519_sc02_5090"
KEY_PATH_PUB="${KEY_PATH}.pub"

SRC_DATA_DIR="/home/suhang/datasets2/rivermind-data/lerobot/local/fabric_droid_v001"
DST_DATA_DIR="/home/suhang/datasets2/rivermind-data/lerobot/local/fabric_droid_v001"
SC02_DATA_STAGING_ROOT="${SC02_DATA_STAGING_ROOT:-/tmp/fabric_droid_v001_sync}"
USE_LOCAL_FALLBACK="${USE_LOCAL_FALLBACK:-1}"

OPENPI_ROOT="/workspace/openpi"
PYTHON_BIN="/root/rivermind-data/venvs/openpi/bin/python"
STORAGE_ROOT="/home/suhang/datasets2/rivermind-data"
HF_HOME="${STORAGE_ROOT}/cache/huggingface"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EXP_NAME="${EXP_NAME:-fabric_droid_v001_lora_sc02}"
CHECKPOINT_BASE_DIR="${STORAGE_ROOT}/checkpoints/fabric_droid_v001_lora_10k"
ASSETS_BASE_DIR="${STORAGE_ROOT}/assets"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"

FORCE_SYNC="${SC02_FORCE_SYNC:-0}"
PREFER_RSYNC="${PREFER_RSYNC_SYNC:-1}"
TRAIN_START_TIMEOUT="${TRAIN_START_TIMEOUT:-0}"
TRAIN_POLL_INTERVAL="${TRAIN_POLL_INTERVAL:-10}"

SC02_SSH_OPTS=(
  -i "${KEY_PATH}"
  -p "${SC02_PORT}"
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
)

AH01_SSH_OPTS=(
  -i "${KEY_PATH}"
  -p "${AH01_PORT}"
  -o StrictHostKeyChecking=accept-new
  -o BatchMode=yes
)

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERROR] $*"; }

ensure_local_key() {
  if [ -f "${KEY_PATH}" ] && [ -f "${KEY_PATH_PUB}" ]; then
    log "Reusing existing key ${KEY_PATH}"
    return 0
  fi
  err "Missing SSH key pair ${KEY_PATH}(/.pub). Create and copy first:"
  err "  ssh-keygen -t ed25519 -C fabric-sc02-5090 -f ${KEY_PATH} -N ''"
  exit 1
}

ensure_access_to_sc02() {
  if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "hostname" >/tmp/sc02-hostname.out 2>&1; then
    log "sc02 key authentication works."
    rm -f /tmp/sc02-hostname.out
    return 0
  fi
  rm -f /tmp/sc02-hostname.out
  return 1
}

add_key_to_host() {
  local host="$1"
  local port="$2"
  if ssh-copy-id -i "${KEY_PATH_PUB}" -p "${port}" "${host}" >/tmp/sc02-keycopy.out 2>&1; then
    rm -f /tmp/sc02-keycopy.out
    return 0
  fi
  warn "ssh-copy-id to ${host} failed:"
  cat /tmp/sc02-keycopy.out || true
  rm -f /tmp/sc02-keycopy.out
  return 1
}

ensure_sc02_to_ah01() {
  if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" \
    "ssh -p ${AH01_PORT} -o StrictHostKeyChecking=accept-new -o BatchMode=yes ${AH01_HOST} 'hostname' >/dev/null 2>&1"; then
    log "sc02 can reach ah01 by key."
    return 0
  fi
  warn "sc02 cannot SSH ah01 with current key yet."
  log "Trying to install key into ah01 from local."
  add_key_to_host "${AH01_HOST}" "${AH01_PORT}" || true
  if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" \
    "ssh -p ${AH01_PORT} -o StrictHostKeyChecking=accept-new -o BatchMode=yes ${AH01_HOST} 'hostname' >/dev/null 2>&1"; then
    log "sc02 now reaches ah01 by key."
    return 0
  fi
  return 1
}

dataset_exists_sc02() {
  if [ "${FORCE_SYNC}" = "1" ]; then
    return 1
  fi
  if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" \
    "test -d '${DST_DATA_DIR}' && ls -1A '${DST_DATA_DIR}' | grep -q ."; then
    return 0
  fi
  return 1
}

install_rsync_on_sc02() {
  if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "command -v rsync >/dev/null 2>&1"; then
    return 0
  fi
  log "rsync not installed on sc02, try install..."
  ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" \
    "command -v apt-get >/dev/null 2>&1 && apt-get update && apt-get install -y rsync || (command -v yum >/dev/null 2>&1 && yum -y install rsync)" \
    || true
  if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "command -v rsync >/dev/null 2>&1"; then
    return 0
  fi
  return 1
}

sync_via_sc02_rsync() {
  log "Syncing via sc02-side rsync: ${AH01_HOST}:${SRC_DATA_DIR} -> ${SC02_HOST%:}/${DST_DATA_DIR}"
  if ! install_rsync_on_sc02; then
    warn "rsync still unavailable on sc02."
    return 1
  fi
  if ! ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "command -v ssh >/dev/null 2>&1"; then
    err "ssh binary missing on sc02."
    return 1
  fi
  if ! ensure_sc02_to_ah01; then
    return 1
  fi

  ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" \
    "mkdir -p '${DST_DATA_DIR%/}' && \
     rsync -az --progress --partial --delete \
     -e \"ssh -p ${AH01_PORT} -o StrictHostKeyChecking=accept-new -o BatchMode=yes\" \
     '${AH01_HOST}:${SRC_DATA_DIR%/}/' '${DST_DATA_DIR%/}/'"
}

sync_via_sc02_tar() {
  log "Syncing via sc02-side tar stream."
  if ! ssh "${AH01_SSH_OPTS[@]}" "${AH01_HOST}" "command -v tar >/dev/null 2>&1"; then
    err "tar missing on ah01."
    return 1
  fi
  if ! ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "command -v tar >/dev/null 2>&1"; then
    err "tar missing on sc02."
    return 1
  fi
  local src_parent
  local src_base
  local dst_parent
  src_parent="$(dirname "${SRC_DATA_DIR}")"
  src_base="$(basename "${SRC_DATA_DIR}")"
  dst_parent="$(dirname "${DST_DATA_DIR}")"

  ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "rm -rf '${DST_DATA_DIR}' && mkdir -p '${dst_parent}'"
  ssh "${AH01_SSH_OPTS[@]}" "${AH01_HOST}" "tar -C '${src_parent}' -cf - '${src_base}'" | \
    ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "mkdir -p '${dst_parent}' && tar -C '${dst_parent}' -xf -"
}

sync_via_local_staging() {
  local staging_root="${SC02_DATA_STAGING_ROOT%/}"
  local staging_dir="${staging_root}/fabric_droid_v001"
  local dst_parent
  dst_parent="$(dirname "${DST_DATA_DIR}")"

  log "Syncing via local staging: ah01 -> local:${staging_dir} -> sc02:${SC02_HOST}:${DST_DATA_DIR}"
  mkdir -p "${staging_root}"
  rm -rf "${staging_dir}"

  if command -v rsync >/dev/null 2>&1; then
    rsync -az --progress --partial --delete \
      -e "ssh -p ${AH01_PORT} -i ${KEY_PATH} -o StrictHostKeyChecking=accept-new -o BatchMode=yes" \
      "${AH01_HOST}:${SRC_DATA_DIR%/}/" "${staging_dir}/"
  else
    warn "rsync unavailable locally, fallback to scp ah01 -> local staging."
    scp -r -i "${KEY_PATH}" -P "${AH01_PORT}" -o StrictHostKeyChecking=accept-new \
      "${AH01_HOST}:${SRC_DATA_DIR%/}" "${staging_root}/"
  fi

  if command -v rsync >/dev/null 2>&1; then
    rsync -az --progress --partial --delete \
      -e "ssh -p ${SC02_PORT} -i ${KEY_PATH} -o StrictHostKeyChecking=accept-new -o BatchMode=yes" \
      "${staging_dir}/" "${SC02_HOST}:${dst_parent%/}/"
  else
    warn "rsync unavailable locally, fallback to scp local staging -> sc02."
    scp -r -i "${KEY_PATH}" -P "${SC02_PORT}" -o StrictHostKeyChecking=accept-new \
      "${staging_dir}" "${SC02_HOST}:${dst_parent%/}/"
  fi
}

sync_dataset() {
  if [ "${PREFER_RSYNC}" = "1" ]; then
    sync_via_sc02_rsync && return 0
    if [ "${USE_LOCAL_FALLBACK}" = "1" ]; then
      sync_via_local_staging && return 0
    fi
    sync_via_sc02_tar && return 0
    return 1
  fi
  if [ "${USE_LOCAL_FALLBACK}" = "1" ]; then
    sync_via_local_staging && return 0
  fi
  sync_via_sc02_tar && return 0
  return 1
}

validate_sync() {
  log "Validate dataset on sc02:"
  ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "ls -lh '${DST_DATA_DIR}' | head"
}

train_cmd_remote() {
  cat <<'EOF'
set -euo pipefail
cd /workspace/openpi

source /home/suhang/openpi_env.sh || true

export OPENPI_ROOT="/workspace/openpi"
export PYTHON_BIN="/root/rivermind-data/venvs/openpi/bin/python"
export STORAGE_ROOT="/home/suhang/datasets2/rivermind-data"
export HF_LEROBOT_HOME="$STORAGE_ROOT/lerobot"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$HF_LEROBOT_HOME" "$HF_HOME" "$CHECKPOINT_BASE_DIR" "$ASSETS_BASE_DIR"
mkdir -p "$(dirname "$HF_HOME")" "$(dirname "$HF_LEROBOT_HOME")" || true

cd "$OPENPI_ROOT"
exec /root/rivermind-data/venvs/openpi/bin/python "scripts/train.py" fabric_droid_lora_10k \
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
EOF
}

start_training_once() {
  local log_dir="${STORAGE_ROOT}/runlogs/openpi/fabric_droid_v001_lora_sc02"
  local log_file="${log_dir}/$(date +%F_%H%M%S)_sc02_train.log"

  local cmd
  cmd="$(train_cmd_remote)"

  local pid
  pid="$(ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" \
    "OPENPI_ROOT='${OPENPI_ROOT}' \
     PYTHON_BIN='${PYTHON_BIN}' \
     STORAGE_ROOT='${STORAGE_ROOT}' \
     HF_HOME='${HF_HOME}' \
     HF_ENDPOINT='${HF_ENDPOINT}' \
     NUM_TRAIN_STEPS='${NUM_TRAIN_STEPS}' \
     SAVE_INTERVAL='${SAVE_INTERVAL}' \
     CHECKPOINT_BASE_DIR='${CHECKPOINT_BASE_DIR}' \
     ASSETS_BASE_DIR='${ASSETS_BASE_DIR}' \
     EXP_NAME='${EXP_NAME}' \
     mkdir -p '${log_dir}' && \
     nohup bash -lc ${cmd@Q} > '${log_file}' 2>&1 & echo \$!")"

  log "Training PID on sc02: ${pid}"
  log "Training log file: ${log_file}"

  local elapsed=0
  while true; do
    if ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "kill -0 ${pid} 2>/dev/null"; then
      log "Process ${pid} is alive."
      ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "tail -n 30 '${log_file}' 2>/dev/null || true"
      return 0
    fi
    if [ "${TRAIN_START_TIMEOUT}" = "0" ]; then
      warn "PID ${pid} not alive yet, wait ${TRAIN_POLL_INTERVAL}s and check again..."
      sleep "${TRAIN_POLL_INTERVAL}"
      continue
    fi
    sleep "${TRAIN_POLL_INTERVAL}"
    elapsed=$((elapsed + TRAIN_POLL_INTERVAL))
    if [ "${elapsed}" -ge "${TRAIN_START_TIMEOUT}" ]; then
      err "Training process failed to stay alive within timeout."
      ssh "${SC02_SSH_OPTS[@]}" "${SC02_HOST}" "tail -n 120 '${log_file}' 2>/dev/null || true"
      return 1
    fi
    warn "PID ${pid} not alive yet (${elapsed}/${TRAIN_START_TIMEOUT}s)..."
  done
}

start_training() {
  local attempt=0
  while true; do
    attempt=$((attempt+1))
    log "Start training attempt ${attempt} with save interval ${SAVE_INTERVAL}"
    if start_training_once; then
      return 0
    fi
    warn "Training not up yet, retrying in 15 seconds..."
    sleep 15
  done
}

main() {
  ensure_local_key

  if ! ensure_access_to_sc02; then
    err "sc02 authentication failed. install key first:"
    err "  ssh-copy-id -i ${KEY_PATH_PUB} -p ${SC02_PORT} ${SC02_HOST}"
    exit 1
  fi

  if ! dataset_exists_sc02; then
    if ! sync_dataset; then
      warn "Primary sync methods failed, fallback to tar stream as last resort."
      if ! sync_via_sc02_tar; then
        err "All sync methods failed."
        exit 1
      fi
    fi
    validate_sync
  else
    log "Dataset already exists on sc02."
  fi

  start_training
}

main "$@"
