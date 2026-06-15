[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
Rethinking-Long-Tail-for-Image-Aesthetics-Assessment
</b>
</h1>
<h4>
<b>
Shuai He, Yuxin Chen, Limin Liu, Anlong Ming, Huadong Ma

      
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

[[国内的小伙伴请看更详细的中文说明]](https://github.com/woshidandan/Rethinking-Long-Tail-for-Image-Aesthetics-Assessment/blob/main/ELTA2_AVA_TRAINING_AND_EVALUATION_ZH.md)

<img width="3935" height="1874" alt="Overall structure_01" src="https://github.com/user-attachments/assets/5c2d31f4-769d-482c-995a-4ffe0b93271a" />


# Rethinking-Long-Tail-for-Image-Aesthetics-Assessment
ELTA 2.0 employs an aesthetics-guided Text-to-Image model to generate new data to solve long-tail issue. 利用文生图模型解决IAA中的长尾问题。

# ELTA 2.0 AVA Training, Inference, and Metrics Guide

This document combines the ELTA 2.0 AVA training workflow, `val+test` inference workflow, paper-style metrics, and scene-level evaluation steps. A typical workflow is:

1. Prepare the environment and data paths.
2. Train or continue training on the AVA training split.
3. Run inference on the AVA `val+test` split.
4. Inspect overall metrics, label-bucket metrics, and scene-bucket metrics.
5. Optionally apply scene calibration and evaluate the calibrated predictions.

## 1. Environment Setup

Use the preconfigured Conda environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_env/elta10
```

Alternatively, point directly to the Python executable in that environment:

```bash
export PYTHON_BIN=/root/autodl-tmp/conda_env/elta10/bin/python
```

Enter the ELTA 2.0 AVA package directory:

```bash
cd /root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE
```

## 2. Data and Metadata Paths

Default AVA data path:

```bash
export DATA_ROOT=/root/autodl-tmp/ELTA/AVA
```

The package already includes the metadata required for training and evaluation:

```bash
export META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels
```

Main metadata files:

```text
labels/ava_train_meta.csv
labels/ava_valtest_meta.csv
labels/tail_config.json
labels/ava_scene_stats.csv
```

These files are used as follows:

- `ava_train_meta.csv` is used for training.
- `ava_valtest_meta.csv` is used for inference and evaluation on the merged `val+test` split.
- `tail_config.json` is used for label-tail buckets and scene evaluation.
- `ava_scene_stats.csv` is used for scene statistics and scene-tail grouping.

## 3. Continue Training from the Packaged Checkpoint

To continue training from the default packaged checkpoint, run:

```bash
DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
OUT_DIR=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package \
RESUME_CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
bash scripts/train_ava.sh
```

Training output directory:

```text
/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package
```

Main output files:

```text
checkpoint/ckpt.pth
history.jsonl
val_epoch*.csv
```

## 4. Train from Scratch or Another Checkpoint

To train from scratch, or to train from another checkpoint, call the core training script directly:

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_ava.py train \
  --train_csv labels/ava_train_meta.csv \
  --val_csv labels/ava_valtest_meta.csv \
  --tail_config labels/tail_config.json \
  --image_dir /root/autodl-tmp/ELTA/AVA/AVA_dataset/image \
  --output_dir /root/autodl-tmp/ELTA/elta2_train_runs/train_clean \
  --gpu_id 0 \
  --epochs 10 \
  --batch_size 48 \
  --num_workers 8 \
  --lr 1e-6 \
  --scene_balanced_sampler \
  --label_tail_weight 0.5 \
  --scene_tail_weight 0.5 \
  --joint_tail_weight 0.5 \
  --sample_multiplier 0.50 \
  --no-tfa \
  --score_loss_weight 0.10 \
  --rank_loss_weight 0.05
```

To resume from a specific checkpoint, add:

```bash
--resume /path/to/ckpt.pth --reset_epoch
```

## 5. Training Logs and Metrics

Each epoch writes the following metrics to `history.jsonl`:

```text
plcc, srcc, mae, mse, LL_mae, LM_mae, LH_mae, SN_mae, SJ_mae
```

Metric definitions:

- `plcc`: Pearson Linear Correlation Coefficient.
- `srcc`: Spearman Rank Correlation Coefficient.
- `mae`: Mean Absolute Error.
- `mse`: Mean Squared Error.
- `LL_mae/LM_mae/LH_mae`: MAE for low-score, mid-score, and high-score aesthetic label buckets.
- `SN_mae/SJ_mae`: MAE for scene-tail and scene-head buckets.

## 6. Inference on the val+test Split

Run inference with the packaged default checkpoint [download from this link](https://drive.google.com/drive/folders/1FVZBY6rym90FIOyO691cPlHQi9QwtH8w?usp=sharing):

```bash
DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
OUT_CSV=/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
bash scripts/infer_ava.sh
```

To evaluate your own trained model, replace `CKPT` with the corresponding checkpoint path, for example:

```bash
CKPT=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package/checkpoint/ckpt.pth
```

Inference outputs:

```text
/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv
/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.metrics.json
```

The `*.metrics.json` file contains:

```text
plcc, srcc, mae, mse, LL_mae, LM_mae, LH_mae, SN_mae, SJ_mae
```

## 7. Generate the Scene-Level Report

To further summarize results by scene, scene-tail, and scene-head buckets, run:

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/scene_eval
```

Output files:

```text
scene_eval/scene_summary.json
scene_eval/per_scene_metrics.csv
scene_eval/predictions_with_scene_buckets.csv
```

Key fields in `scene_summary.json`:

```text
overall.plcc
overall.srcc
overall.mae
label.label_low.mae
label.label_mid.mae
label.label_high.mae
scene.scene_tail.mae
scene.scene_head.mae
```

For paper tables, these fields can be read as:

```text
P  = overall.plcc
S  = overall.srcc
LL = label.label_low.mae
LM = label.label_mid.mae
LH = label.label_high.mae
SN = scene.scene_tail.mae
SJ = scene.scene_head.mae
```

## 8. Use the Scene Calibration File

The package includes an optional scene bias table:

```text
weights/scene_bias.csv
```

To reuse the scene calibration workflow, first fit calibration on training predictions, then apply it to the `val+test` predictions:

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_calibrate.py \
  --train_predictions_csv /path/to/train_predictions.csv \
  --test_predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated \
  --shrink_k 80
```

Then run scene evaluation again on the calibrated predictions:

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated/predictions_calibrated.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated_scene_eval
```

## 9. Recommended Workflows

To reproduce evaluation metrics only, run:

```bash
cd /root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE

DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
OUT_CSV=/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
bash scripts/infer_ava.sh

/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/scene_eval
```

To retrain first and then evaluate the trained checkpoint, run:

```bash
cd /root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE

DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
OUT_DIR=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package \
RESUME_CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
bash scripts/train_ava.sh

DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
CKPT=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package/checkpoint/ckpt.pth \
OUT_CSV=/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
bash scripts/infer_ava.sh
```


