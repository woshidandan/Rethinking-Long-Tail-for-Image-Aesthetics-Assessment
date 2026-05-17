# ELTA 2.0 AVA 验证集+测试集推理与指标教程

本文档说明如何在 AVA `val+test` 合并集上推理，并得到论文中常用指标。

## 1. 进入包目录

```bash
cd /root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE
```

## 2. 推理

使用包内默认权重：

```bash
DATA_ROOT=/root/autodl-tmp/ELTA/AVA \
META_ROOT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/labels \
CKPT=/root/autodl-tmp/ELTA/ELTA2_AVA_PACKAGE/weights/elta2_ava_target_adapted.pth \
OUT_CSV=/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
bash scripts/infer_ava.sh
```

输出：

```text
/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv
/root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.metrics.json
```

`*.metrics.json` 中包含：

```text
plcc, srcc, mae, mse, LL_mae, LM_mae, LH_mae, SN_mae, SJ_mae
```

## 3. 生成 scene 细分报告

如果需要按 scene、长尾 scene、非长尾 scene 进一步统计：

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/scene_eval
```

输出：

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

## 4. 使用 scene 校准文件

包内包含可选的 scene bias 表：

```text
weights/scene_bias.csv
```

如果你需要复用校准流程，应先在训练集预测上拟合校准，再对 val+test 预测应用：

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_calibrate.py \
  --train_predictions_csv /path/to/train_predictions.csv \
  --test_predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/valtest_predictions.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated \
  --shrink_k 80
```

然后对校准后的预测再跑 scene 评估：

```bash
/root/autodl-tmp/conda_env/elta10/bin/python code/elta2_scene_eval.py \
  --predictions_csv /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated/predictions_calibrated.csv \
  --tail_config labels/tail_config.json \
  --train_meta labels/ava_train_meta.csv \
  --scene_stats labels/ava_scene_stats.csv \
  --output_dir /root/autodl-tmp/ELTA/elta2_eval_runs/calibrated_scene_eval
```

