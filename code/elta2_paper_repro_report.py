import argparse
import json
from pathlib import Path

import pandas as pd


PAPER_AVA = {
    "ELTA_1.0": {
        "P": 0.777,
        "S": 0.764,
        "LL": 0.438,
        "LM": 0.302,
        "LH": 0.426,
        "SN": 0.390,
        "SJ": 0.331,
    },
    "ELTA_2.0": {
        "P": 0.793,
        "S": 0.810,
        "LL": 0.412,
        "LM": 0.286,
        "LH": 0.410,
        "SN": 0.339,
        "SJ": 0.316,
    },
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_from_metrics(name, metrics, source):
    return {
        "model": name,
        "source": source,
        "P": metrics.get("plcc"),
        "S": metrics.get("srcc"),
        "LL": metrics.get("LL_mae"),
        "LM": metrics.get("LM_mae"),
        "LH": metrics.get("LH_mae"),
        "SN": metrics.get("SN_mae"),
        "SJ": metrics.get("SJ_mae"),
    }


def row_from_scene_summary(name, summary, source):
    return {
        "model": name,
        "source": source,
        "P": summary["overall"]["plcc"],
        "S": summary["overall"]["srcc"],
        "LL": summary["label"]["label_low"]["mae"],
        "LM": summary["label"]["label_mid"]["mae"],
        "LH": summary["label"]["label_high"]["mae"],
        "SN": summary["scene"]["scene_tail"]["mae"],
        "SJ": summary["scene"]["scene_head"]["mae"],
    }


def main(args):
    rows = []
    for model, metrics in PAPER_AVA.items():
        rows.append({"model": model, "source": "paper_table_3_4", **metrics})

    if args.baseline_metrics and Path(args.baseline_metrics).exists():
        rows.append(row_from_metrics("ELTA_1.0", load_json(args.baseline_metrics), "available_split_eval"))
    if args.available_scene_summary and Path(args.available_scene_summary).exists():
        rows.append(row_from_scene_summary("ELTA_2.0_scene_boost_calibrated", load_json(args.available_scene_summary), "available_split_eval"))

    df = pd.DataFrame(rows)
    metric_cols = ["P", "S", "LL", "LM", "LH", "SN", "SJ"]
    for metric in metric_cols:
        if f"paper_delta_{metric}" in df.columns:
            continue
    paper2 = PAPER_AVA["ELTA_2.0"]
    for metric in metric_cols:
        df[f"delta_to_paper_elta2_{metric}"] = df[metric] - paper2[metric]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paper_reproduction_comparison.csv"
    json_path = out_dir / "paper_reproduction_notes.json"
    df.to_csv(csv_path, index=False)
    notes = {
        "paper_ava_protocol": {
            "train_images": 235528,
            "test_images": 20000,
            "self_training_generated_samples": 35000,
            "label_boundary": "bottom 20% = LL, middle 60% = LM, top 20% = LH",
            "scene_boundary": "minority scenes are scenes with train-frequency percentage in [0,20) relative to the head scene",
            "optimizer": "Adam",
            "batch_size": 48,
            "ava_loss": "EMD",
            "self_training_rounds": 1,
        },
        "available_server_protocol": {
            "train_images": int(args.available_train_count),
            "test_images": int(args.available_test_count),
            "generated_pseudo_samples_current": int(args.available_generated_count),
            "limitation": [
                "Current server split is not the paper's 235528/20000 AVA split.",
                "The paper's 35000 AG-T2I generated AVA supplement is not present in the repository or uploaded data.",
                "Current generated data uses a self-trained compact AVA generator, not the paper's AG-T2I/UltraEdit model.",
            ],
        },
        "target_ava_metrics_from_paper": PAPER_AVA,
    }
    json_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"comparison={csv_path}")
    print(f"notes={json_path}")


def parse_args():
    parser = argparse.ArgumentParser("Build AVA paper reproduction comparison report")
    parser.add_argument("--baseline_metrics", default="/root/autodl-tmp/ELTA/elta2_scene_eval/baseline/predictions.metrics.json")
    parser.add_argument("--available_scene_summary", default="/root/autodl-tmp/ELTA/elta2_scene_eval/scene_boost/calibrated_scene_eval/scene_summary.json")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/ELTA/elta2_paper_reproduction")
    parser.add_argument("--available_train_count", type=int, default=229951)
    parser.add_argument("--available_test_count", type=int, default=12776)
    parser.add_argument("--available_generated_count", type=int, default=28)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
