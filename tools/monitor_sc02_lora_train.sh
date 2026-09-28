#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-sc02-ssh.gpuhome.cc}"
PORT="${PORT:-30102}"
USER="${USER_NAME:-root}"
KEY="${KEY_PATH:-${HOME}/.ssh/id_ed25519_sc02_5090}"
LOG_DIR="${LOG_DIR:-/home/suhang/datasets2/rivermind-data/runlogs/openpi/fabric_droid_v001_lora_sc02}"
CKPT_DIR="${CKPT_DIR:-/home/suhang/datasets2/rivermind-data/checkpoints/fabric_droid_v001_lora_10k}"
INTERVAL="${INTERVAL:-20}"

if [ ! -f "${KEY}" ]; then
  echo "[ERROR] SSH key not found: ${KEY}"
  exit 1
fi

echo "[INFO] Monitoring sc02 training every ${INTERVAL}s"
echo "[INFO] Host: ${USER}@${HOST}:${PORT}"

while true; do
  NOW="$(date '+%F %T')"
  echo "==== ${NOW} ===="

  echo "[check] process"
  ssh -i "${KEY}" -p "${PORT}" "${USER}@${HOST}" "pgrep -af 'fabric_droid_lora_10k|run_fabric_lora_sc02_until_shutdown|train.py' || true"

  echo "[check] latest logs"
  ssh -i "${KEY}" -p "${PORT}" "${USER}@${HOST}" "LATEST=\$(ls -1t '${LOG_DIR}'/*.log 2>/dev/null | head -n 1); if [ -n \"\${LATEST}\" ]; then echo \"$LATEST\"; tail -n 25 \"\${LATEST}\"; else echo 'NO_LOG'; fi"

  echo "[check] latest ckpt"
  ssh -i "${KEY}" -p "${PORT}" "${USER}@${HOST}" "ls -lt '${CKPT_DIR}' 2>/dev/null | head -n 12 || true"

  echo
  sleep "${INTERVAL}"
done
