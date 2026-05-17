# ELTA 2.0 AVA 训练教程

本文档说明如何只使用 AVA 训练集进行训练。

## 1. 环境

推荐使用已经配置好的 Conda 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_env/elta10
```

或者直接使用环境中的 Python：

```bash
export PYTHON_BIN=/root/autodl-tmp/conda_env/elta10/bin/python
```

## 2. 数据路径

默认 AVA 图像路径：

```bash
export DATA_ROOT=/root/autodl-tmp/ELTA/AVA
```

包内已经包含训练和评估需要的标签元数据：

```bash
export META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels
```

其中：

```text
labels/ava_train_meta.csv
labels/ava_valtest_meta.csv
labels/tail_config.json
labels/ava_scene_stats.csv
```

## 3. 从已有权重继续训练

进入包目录：

```bash
cd /root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE
```

运行：

```bash
DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
OUT_DIR=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package \
RESUME_CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
bash scripts/train_ava.sh
```

训练输出在：

```text
/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package
```

主要文件：

```text
checkpoint/ckpt.pth
history.jsonl
val_epoch*.csv
```

## 4. 从头训练或从其他权重训练

直接调用核心训练脚本：

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

如果要从某个 checkpoint 继续训练，增加：

```bash
--resume /path/to/ckpt.pth --reset_epoch
```

## 5. 训练日志说明

每个 epoch 会在 `history.jsonl` 中记录：

```text
plcc, srcc, mae, mse, LL_mae, LM_mae, LH_mae, SN_mae, SJ_mae
```

其中：

- `LL/LM/LH` 是按美学分数长尾划分的 MAE。
- `SN/SJ` 是按 scene 长尾/非长尾划分的 MAE。

