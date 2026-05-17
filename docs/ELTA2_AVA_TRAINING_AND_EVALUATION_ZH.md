# ELTA 2.0 AVA 训练、推理与指标教程

本文档整合了 ELTA 2.0 AVA 包中的训练流程、`val+test` 合并集推理流程，以及论文常用指标和 scene 细分评估方法。推荐按以下顺序使用：

1. 配置环境与数据路径。
2. 使用 AVA 训练集训练或继续训练模型。
3. 在 AVA `val+test` 合并集上推理。
4. 查看整体指标、label 分段指标和 scene 分段指标。
5. 如有需要，使用 scene 校准流程进一步评估。

## 1. 环境准备

推荐使用已经配置好的 Conda 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_env/elta10
```

也可以直接指定环境中的 Python：

```bash
export PYTHON_BIN=/root/autodl-tmp/conda_env/elta10/bin/python
```

进入 ELTA 2.0 AVA 包目录：

```bash
cd /root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE
```

## 2. 数据与元数据路径

默认 AVA 数据路径：

```bash
export DATA_ROOT=/root/autodl-tmp/ELTA/AVA
```

包内已经包含训练和评估需要的标签元数据：

```bash
export META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels
```

主要元数据文件包括：

```text
labels/ava_train_meta.csv
labels/ava_valtest_meta.csv
labels/tail_config.json
labels/ava_scene_stats.csv
```

其中：

- `ava_train_meta.csv` 用于训练。
- `ava_valtest_meta.csv` 用于 `val+test` 合并集推理和评估。
- `tail_config.json` 用于 label 长尾分段和 scene 评估。
- `ava_scene_stats.csv` 用于 scene 统计和 scene 长尾划分。

## 3. 从已有权重继续训练

如需基于包内默认权重继续训练，运行：

```bash
DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
OUT_DIR=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package \
RESUME_CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
bash scripts/train_ava.sh
```

训练输出目录：

```text
/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package
```

主要输出文件：

```text
checkpoint/ckpt.pth
history.jsonl
val_epoch*.csv
```

## 4. 从头训练或从其他 checkpoint 训练

如果需要从头训练，或从其他 checkpoint 开始训练，可以直接调用核心训练脚本：

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

如果要从某个 checkpoint 继续训练，增加以下参数：

```bash
--resume /path/to/ckpt.pth --reset_epoch
```

## 5. 训练日志与指标

每个 epoch 会在 `history.jsonl` 中记录：

```text
plcc, srcc, mae, mse, LL_mae, LM_mae, LH_mae, SN_mae, SJ_mae
```

指标含义：

- `plcc`：Pearson Linear Correlation Coefficient。
- `srcc`：Spearman Rank Correlation Coefficient。
- `mae`：Mean Absolute Error。
- `mse`：Mean Squared Error。
- `LL_mae/LM_mae/LH_mae`：按美学分数区间划分的低分段、中分段、高分段 MAE。
- `SN_mae/SJ_mae`：按 scene 长尾和非长尾划分的 MAE。

## 6. 在 val+test 合并集上推理

使用包内默认权重推理：

```bash
DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
OUT_CSV=/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
bash scripts/infer_ava.sh
```

如果要评估自己训练得到的模型，请将 `CKPT` 替换为对应 checkpoint，例如：

```bash
CKPT=/root/autodl-tmp/ELTA/elta2_train_runs/train_from_package/checkpoint/ckpt.pth
```

推理输出：

```text
/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv
/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.metrics.json
```

`*.metrics.json` 中包含：

```text
plcc, srcc, mae, mse, LL_mae, LM_mae, LH_mae, SN_mae, SJ_mae
```

## 7. 生成 scene 细分报告

如果需要按 scene、长尾 scene 和非长尾 scene 进一步统计，运行：

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/scene_eval
```

输出文件：

```text
scene_eval/scene_summary.json
scene_eval/per_scene_metrics.csv
scene_eval/predictions_with_scene_buckets.csv
```

`scene_summary.json` 中的关键字段：

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

对应论文表格可读为：

```text
P  = overall.plcc
S  = overall.srcc
LL = label.label_low.mae
LM = label.label_mid.mae
LH = label.label_high.mae
SN = scene.scene_tail.mae
SJ = scene.scene_head.mae
```

## 8. 使用 scene 校准文件

包内包含可选的 scene bias 表：

```text
weights/scene_bias.csv
```

如果需要复用 scene 校准流程，应先在训练集预测上拟合校准，再对 `val+test` 预测应用：

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_calibrate.py \
  --train_predictions_csv /path/to/train_predictions.csv \
  --test_predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated \
  --shrink_k 80
```

然后对校准后的预测再运行 scene 评估：

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated/predictions_calibrated.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated_scene_eval
```

## 9. 推荐工作流

如果只是复现实验指标，可以直接执行：

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

如果需要重新训练后再评估，可以执行：

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

