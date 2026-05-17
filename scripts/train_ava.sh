#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_env/elta10/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ELTA/AVA}"
META_ROOT="${META_ROOT:-/root/autodl-tmp/ELTA/elta2_ava_impl}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/ELTA/elta2_ava_runs/train}"
RESUME_CKPT="${RESUME_CKPT:-${ROOT_DIR}/weights/elta2_ava_target_adapted.pth}"

"${PYTHON_BIN}" "${ROOT_DIR}/code/elta2_ava.py" train \
  --train_csv "${META_ROOT}/ava_train_meta.csv" \
  --val_csv "${META_ROOT}/ava_valtest_meta.csv" \
  --tail_config "${META_ROOT}/tail_config.json" \
  --image_dir "${DATA_ROOT}/AVA_dataset/image" \
  --output_dir "${OUT_DIR}" \
  --gpu_id 0 \
  --epochs 10 \
  --batch_size 48 \
  --num_workers 8 \
  --lr 1e-6 \
  --resume "${RESUME_CKPT}" \
  --reset_epoch \
  --scene_balanced_sampler \
  --label_tail_weight 0.5 \
  --scene_tail_weight 0.5 \
  --joint_tail_weight 0.5 \
  --sample_multiplier 0.50 \
  --no-tfa \
  --score_loss_weight 0.10 \
  --rank_loss_weight 0.05

