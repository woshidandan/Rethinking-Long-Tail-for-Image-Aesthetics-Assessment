#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_env/elta10/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-${ROOT_DIR}/weights/elta2_ava_generator.pth}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/ELTA/elta2_ava_generation/images}"

"${PYTHON_BIN}" "${ROOT_DIR}/code/elta2_ava_generator.py" generate \
  --ckpt "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --gpu_id 0 \
  --num_images 100 \
  --batch_size 16

