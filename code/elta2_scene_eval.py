import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def safe_corr(fn, x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    value = fn(x, y)[0]
    return None if not np.isfinite(value) else float(value)


def metric_block(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return {
        "count": int(len(gt)),
        "plcc": safe_corr(pearsonr, pred, gt),
        "srcc": safe_corr(spearmanr, pred, gt),
        "mae": float(mean_absolute_error(gt, pred)) if len(gt) else None,
        "mse": float(mean_squared_error(gt, pred)) if len(gt) else None,
    }


def load_tail_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_scene_frequency(pred, train_meta=None, scene_stats=None):
    pred = pred.copy()
    if scene_stats:
        stats = pd.read_csv(scene_stats)
        if "category" in stats.columns and "count" in stats.columns:
            pred = pred.merge(stats[["category", "count"]].rename(columns={"count": "train_scene_count"}), on="category", how="left")
    elif train_meta:
        train = pd.read_csv(train_meta, usecols=["category"])
        counts = train["category"].astype(str).value_counts().rename_axis("category").reset_index(name="train_scene_count")
        pred = pred.merge(counts, on="category", how="left")
    pred["train_scene_count"] = pred["train_scene_count"].fillna(0).astype(int)
    return pred


def group_metrics(df, masks):
    out = {}
    for name, mask in masks.items():
        sub = df.loc[mask]
        out[name] = metric_block(sub["score"], sub["gt"])
    return out


def main(args):
    pred = pd.read_csv(args.predictions_csv)
    required = {"category", "gt", "score"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    pred["category"] = pred["category"].astype(str).fillna("unknown")
    config = load_tail_config(args.tail_config)
    pred = add_scene_frequency(pred, args.train_meta, args.scene_stats)

    low_cut = float(config["low_cut"])
    high_cut = float(config["high_cut"])
    minority_scenes = set(str(x) for x in config["minority_scenes"])
    scene_counts = pred[["category", "train_scene_count"]].drop_duplicates()
    nonzero = scene_counts["train_scene_count"][scene_counts["train_scene_count"] > 0]
    if len(nonzero) >= 3:
        q33, q66 = np.percentile(nonzero, [33.333, 66.667])
    else:
        q33, q66 = 0, 0

    pred["label_bucket"] = np.select(
        [pred["gt"] <= low_cut, pred["gt"] >= high_cut],
        ["label_low", "label_high"],
        default="label_mid",
    )
    pred["scene_bucket"] = np.select(
        [pred["category"].isin(minority_scenes), pred["train_scene_count"] >= q66],
        ["scene_tail", "scene_head"],
        default="scene_mid",
    )
    pred["scene_label_bucket"] = pred["scene_bucket"] + "__" + pred["label_bucket"]

    summary = {
        "overall": metric_block(pred["score"], pred["gt"]),
        "cuts": {
            "low_cut": low_cut,
            "high_cut": high_cut,
            "scene_count_q33": float(q33),
            "scene_count_q66": float(q66),
            "minority_scene_count": len(minority_scenes),
        },
        "label": group_metrics(pred, {
            "label_low": pred["label_bucket"] == "label_low",
            "label_mid": pred["label_bucket"] == "label_mid",
            "label_high": pred["label_bucket"] == "label_high",
        }),
        "scene": group_metrics(pred, {
            "scene_tail": pred["scene_bucket"] == "scene_tail",
            "scene_mid": pred["scene_bucket"] == "scene_mid",
            "scene_head": pred["scene_bucket"] == "scene_head",
        }),
        "scene_x_label": {},
    }
    for scene_bucket in ["scene_tail", "scene_mid", "scene_head"]:
        for label_bucket in ["label_low", "label_mid", "label_high"]:
            name = f"{scene_bucket}__{label_bucket}"
            summary["scene_x_label"][name] = metric_block(
                pred.loc[pred["scene_label_bucket"] == name, "score"],
                pred.loc[pred["scene_label_bucket"] == name, "gt"],
            )

    per_scene_rows = []
    for scene, sub in pred.groupby("category"):
        item = {
            "category": scene,
            "test_count": int(len(sub)),
            "train_scene_count": int(sub["train_scene_count"].iloc[0]),
            "scene_bucket": sub["scene_bucket"].iloc[0],
            "gt_mean": float(sub["gt"].mean()),
            "pred_mean": float(sub["score"].mean()),
        }
        item.update(metric_block(sub["score"], sub["gt"]))
        per_scene_rows.append(item)
    per_scene = pd.DataFrame(per_scene_rows).sort_values(["scene_bucket", "mae", "test_count"], ascending=[False, False, False])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / args.summary_name
    scene_path = out_dir / args.per_scene_name
    bucket_path = out_dir / args.predictions_with_buckets_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    per_scene.to_csv(scene_path, index=False)
    pred.to_csv(bucket_path, index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary={summary_path}")
    print(f"per_scene={scene_path}")
    print(f"predictions_with_buckets={bucket_path}")


def parse_args():
    parser = argparse.ArgumentParser("Scene-wise ELTA 2.0 evaluation")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--tail_config", default="/root/autodl-tmp/ELTA/elta2_ava_impl/tail_config.json")
    parser.add_argument("--train_meta", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_train_meta.csv")
    parser.add_argument("--scene_stats", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_scene_stats.csv")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/ELTA/elta2_scene_eval")
    parser.add_argument("--summary_name", default="scene_summary.json")
    parser.add_argument("--per_scene_name", default="per_scene_metrics.csv")
    parser.add_argument("--predictions_with_buckets_name", default="predictions_with_scene_buckets.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
