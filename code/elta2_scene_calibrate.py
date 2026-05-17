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


def metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return {
        "plcc": safe_corr(pearsonr, pred, gt),
        "srcc": safe_corr(spearmanr, pred, gt),
        "mae": float(mean_absolute_error(gt, pred)),
        "mse": float(mean_squared_error(gt, pred)),
    }


def fit_affine(pred, gt):
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(gt, dtype=np.float64)
    var = np.var(x)
    if var < 1e-12:
        return 1.0, float(y.mean() - x.mean())
    a = float(np.cov(x, y, bias=True)[0, 1] / var)
    b = float(y.mean() - a * x.mean())
    return a, b


def main(args):
    train = pd.read_csv(args.train_predictions_csv)
    test = pd.read_csv(args.test_predictions_csv)
    for df in (train, test):
        df["category"] = df["category"].astype(str).fillna("unknown")

    global_a, global_b = fit_affine(train["score"], train["gt"])
    train_global = np.clip(global_a * train["score"].to_numpy() + global_b, 1.0, 10.0)
    residual = train["gt"].to_numpy() - train_global
    global_bias = float(residual.mean())

    scene_rows = []
    for scene, sub in train.assign(global_pred=train_global, residual=residual).groupby("category"):
        n = len(sub)
        scene_bias = float(sub["residual"].mean())
        shrink = float(n / (n + args.shrink_k))
        scene_rows.append({
            "category": scene,
            "count": int(n),
            "bias": scene_bias,
            "shrink": shrink,
            "effective_bias": shrink * scene_bias + (1.0 - shrink) * global_bias,
        })
    scene_bias = pd.DataFrame(scene_rows)
    test_cal = test.merge(scene_bias[["category", "effective_bias", "count"]], on="category", how="left")
    test_cal["effective_bias"] = test_cal["effective_bias"].fillna(global_bias)
    test_cal["raw_score"] = test_cal["score"]
    test_cal["calibrated_score"] = np.clip(global_a * test_cal["score"].to_numpy() + global_b + test_cal["effective_bias"].to_numpy(), 1.0, 10.0)

    summary = {
        "global_affine": {"a": global_a, "b": global_b, "global_bias_after_affine": global_bias},
        "train_before": metrics(train["score"], train["gt"]),
        "train_after_global": metrics(train_global, train["gt"]),
        "test_before": metrics(test_cal["score"], test_cal["gt"]),
        "test_after_scene_calibration": metrics(test_cal["calibrated_score"], test_cal["gt"]),
        "shrink_k": args.shrink_k,
    }
    test_cal["score"] = test_cal["calibrated_score"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pred = out_dir / "predictions_calibrated.csv"
    out_scene = out_dir / "scene_bias.csv"
    out_summary = out_dir / "calibration_summary.json"
    test_cal.to_csv(out_pred, index=False)
    scene_bias.to_csv(out_scene, index=False)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"calibrated_predictions={out_pred}")
    print(f"scene_bias={out_scene}")
    print(f"summary={out_summary}")


def parse_args():
    parser = argparse.ArgumentParser("Scene-aware prediction calibration")
    parser.add_argument("--train_predictions_csv", required=True)
    parser.add_argument("--test_predictions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shrink_k", type=float, default=80.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
