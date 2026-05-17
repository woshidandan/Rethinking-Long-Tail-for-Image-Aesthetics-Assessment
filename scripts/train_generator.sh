#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_env/elta10/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ELTA/AVA}"
META_ROOT="${META_ROOT:-/root/autodl-tmp/ELTA/elta2_ava_impl}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/ELTA/elta2_ava_generator}"

"${PYTHON_BIN}" "${ROOT_DIR}/code/elta2_ava_generator.py" train \
  --train_csv "${META_ROOT}/ava_train_meta.csv" \
  --image_dir "${DATA_ROOT}/AVA_dataset/image" \
  --output_dir "${OUT_DIR}" \
  --gpu_id 0 \
  --epochs 10 \
  --batch_size 64 \
  --num_workers 8

