#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_env/elta10/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ELTA/AVA}"
META_ROOT="${META_ROOT:-/root/autodl-tmp/ELTA/elta2_ava_impl}"
CKPT="${CKPT:-${ROOT_DIR}/weights/elta2_ava_target_adapted.pth}"
OUT_CSV="${OUT_CSV:-/root/autodl-tmp/ELTA/elta2_ava_runs/inference/predictions.csv}"

mkdir -p "$(dirname "${OUT_CSV}")"

"${PYTHON_BIN}" "${ROOT_DIR}/code/elta2_ava.py" eval \
  --csv "${META_ROOT}/ava_valtest_meta.csv" \
  --tail_config "${META_ROOT}/tail_config.json" \
  --image_dir "${DATA_ROOT}/AVA_dataset/image" \
  --ckpt "${CKPT}" \
  --output_csv "${OUT_CSV}" \
  --gpu_id 0 \
  --batch_size 128 \
  --num_workers 8

