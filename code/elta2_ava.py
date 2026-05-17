import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import ImageFile
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from models.swin_model import Swin


ImageFile.LOAD_TRUNCATED_IMAGES = True
SCORE_COLS = [f"score{i}" for i in range(2, 12)]
PROB_COLS = [f"prob_{i}" for i in range(1, 11)]
LOGIT_COLS = [f"logit_{i}" for i in range(1, 11)]


@dataclass
class TailConfig:
    low_cut: float
    high_cut: float
    scene_ratio_cut: float
    minority_scenes: list


class EMDLoss(nn.Module):
    def forward(self, preds, targets):
        cdf_pred = torch.cumsum(preds, dim=1)
        cdf_target = torch.cumsum(targets, dim=1)
        return torch.sqrt(torch.mean(torch.pow(cdf_pred - cdf_target, 2), dim=1)).mean()


def distribution_to_score_array(dist):
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist / np.clip(dist.sum(axis=1, keepdims=True), 1e-12, None)
    return (dist * np.arange(1, 11, dtype=np.float64)).sum(axis=1)


def torch_distribution_score(dist):
    weights = torch.arange(1, 11, dtype=dist.dtype, device=dist.device).view(1, -1)
    return (dist * weights).sum(dim=1)


def normalize_distribution(df):
    dist = df[SCORE_COLS].to_numpy(dtype=np.float64)
    dist = dist / np.clip(dist.sum(axis=1, keepdims=True), 1e-12, None)
    out = df.copy()
    out[SCORE_COLS] = dist
    out["score"] = distribution_to_score_array(dist)
    return out


def load_label_csv(path):
    df = pd.read_csv(path)
    if all(c in df.columns for c in SCORE_COLS):
        return normalize_distribution(df)
    if "score" not in df.columns:
        raise ValueError(f"{path} must contain either score2-score11 or score")
    score = df["score"].astype(float).clip(1, 10).to_numpy()
    bins = np.rint(score).astype(int).clip(1, 10)
    for i in range(1, 11):
        df[f"score{i + 1}"] = (bins == i).astype(np.float32)
    return df


def build_metadata(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_label_csv(args.train_label_csv)
    test = load_label_csv(args.test_label_csv)
    scene_train = pd.read_csv(args.train_scene_csv)
    scene_test = pd.read_csv(args.test_scene_csv)

    scene_train = scene_train.rename(columns={"置信度": "scene_conf", "类别1": "category", "图像名称": "image_id"})
    scene_test = scene_test.rename(columns={"置信度": "scene_conf", "类别1": "category", "图像名称": "image_id"})
    for scene_df in (scene_train, scene_test):
        scene_df["image_id"] = scene_df["image_id"].astype(str).str.replace(".jpg", "", regex=False).astype(int)
        if "scene_conf" not in scene_df.columns:
            scene_df["scene_conf"] = 1.0

    train = train.merge(scene_train[["image_id", "category", "scene_conf"]], on="image_id", how="left")
    test = test.merge(scene_test[["image_id", "category", "scene_conf"]], on="image_id", how="left")
    train["category"] = train["category"].fillna("unknown")
    test["category"] = test["category"].fillna("unknown")
    train["scene_conf"] = train["scene_conf"].fillna(0.0)
    test["scene_conf"] = test["scene_conf"].fillna(0.0)
    train["is_generated"] = 0
    test["is_generated"] = 0

    counts = train["category"].value_counts().rename_axis("category").reset_index(name="count")
    max_count = counts["count"].max()
    counts["ratio_to_head"] = counts["count"] / max_count
    low_cut, high_cut = np.percentile(train["score"].to_numpy(), [20, 80])
    minority = counts[counts["ratio_to_head"] < args.scene_ratio_cut]["category"].tolist()
    if not minority:
        minority_count = max(1, int(math.ceil(len(counts) * args.scene_bottom_percent / 100.0)))
        minority = counts.sort_values("count").head(minority_count)["category"].tolist()
    config = TailConfig(float(low_cut), float(high_cut), float(args.scene_ratio_cut), minority)

    train.to_csv(out_dir / "ava_train_meta.csv", index=False)
    test.to_csv(out_dir / "ava_test_meta.csv", index=False)
    counts.to_csv(out_dir / "ava_scene_stats.csv", index=False)
    (out_dir / "tail_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    print(f"train_meta={out_dir / 'ava_train_meta.csv'}")
    print(f"test_meta={out_dir / 'ava_test_meta.csv'}")
    print(f"scene_stats={out_dir / 'ava_scene_stats.csv'}")
    print(f"tail_config={out_dir / 'tail_config.json'}")
    print(f"low_cut={config.low_cut:.4f}, high_cut={config.high_cut:.4f}, minority_scenes={len(minority)}")


class AVAELTA2Dataset(Dataset):
    def __init__(self, csv_path, image_dir, generated_dir=None, train=False):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.generated_dir = Path(generated_dir) if generated_dir else None
        self.train = train
        self.scene_to_idx = {name: i for i, name in enumerate(sorted(self.df["category"].fillna("unknown").unique()))}
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((256, 256)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.df)

    def resolve_image(self, row):
        if "image_path" in row and isinstance(row["image_path"], str) and row["image_path"]:
            p = Path(row["image_path"])
            if p.exists():
                return p
        image_id = row["image_id"]
        try:
            name = f"{int(image_id)}.jpg"
        except Exception:
            name = str(image_id)
        p = self.image_dir / name
        if p.exists():
            return p
        if self.generated_dir is not None:
            category = str(row.get("category", ""))
            candidates = [self.generated_dir / category / name, self.generated_dir / name]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return p

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = default_loader(self.resolve_image(row))
        dist = row[SCORE_COLS].to_numpy(dtype=np.float32)
        dist = dist / max(float(dist.sum()), 1e-12)
        score = float(row["score"])
        category = str(row.get("category", "unknown"))
        scene_idx = self.scene_to_idx.get(category, 0)
        is_generated = int(row.get("is_generated", 0))
        return self.transform(image), torch.from_numpy(dist), torch.tensor(score, dtype=torch.float32), scene_idx, is_generated


def make_sampler(df, config, args):
    counts = df["category"].value_counts()
    max_count = counts.max()
    scene_ratio = df["category"].map(counts).astype(float) / max_count
    label_tail = (df["score"] <= config.low_cut) | (df["score"] >= config.high_cut)
    scene_tail = df["category"].isin(config.minority_scenes) | (scene_ratio < config.scene_ratio_cut)
    weights = np.ones(len(df), dtype=np.float64)
    weights += args.label_tail_weight * label_tail.to_numpy(dtype=np.float64)
    weights += args.scene_tail_weight * scene_tail.to_numpy(dtype=np.float64)
    weights += args.joint_tail_weight * (label_tail & scene_tail).to_numpy(dtype=np.float64)
    if "is_generated" in df.columns:
        weights += args.generated_weight * df["is_generated"].fillna(0).to_numpy(dtype=np.float64)
    num_samples = int(round(len(df) * args.sample_multiplier))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def feature_tail_mixup(features, targets, scores, tau_1=0.5, tau_2=2.0, mixup_ratio=0.25):
    batch_size = features.size(0)
    num_pairs = max(1, int(round(batch_size * mixup_ratio)))
    num_pairs = min(num_pairs, batch_size - 1)
    if num_pairs <= 0:
        return features, targets
    score_np = scores.detach().float().cpu().numpy()
    mean_score = score_np.mean()
    prob = np.exp(np.abs(score_np - mean_score) / max(tau_1, 1e-6))
    prob = prob / prob.sum()
    first = np.random.choice(np.arange(batch_size), size=num_pairs, replace=False, p=prob)
    second = []
    lambdas = []
    for i in first:
        pair_logits = np.clip(tau_2 / (np.abs(score_np[i] - score_np) + 1e-2), a_min=None, a_max=50.0)
        pair_prob = np.exp(pair_logits)
        pair_prob[i] = 0
        pair_prob = pair_prob / pair_prob.sum()
        j = np.random.choice(np.arange(batch_size), p=pair_prob)
        second.append(j)
        lambdas.append(prob[i] / (prob[i] + prob[j]))
    first = torch.as_tensor(first, device=features.device)
    second = torch.as_tensor(second, device=features.device)
    lam = torch.as_tensor(lambdas, dtype=features.dtype, device=features.device).view(-1, 1)
    mixed_features = features[first] * lam + features[second] * (1 - lam)
    mixed_targets = targets[first] * lam + targets[second] * (1 - lam)
    return torch.cat([features, mixed_features], dim=0), torch.cat([targets, mixed_targets], dim=0)


def similarity_alignment_loss(features, targets):
    score = torch_distribution_score(targets).view(-1, 1)
    feat_sim = F.normalize(features.view(features.size(0), -1), dim=1) @ F.normalize(features.view(features.size(0), -1), dim=1).t()
    denom = torch.clamp(score.max() - score.min(), min=1e-6)
    label_sim = 1.0 - torch.abs(score - score.t()) / denom
    return F.mse_loss(feat_sim, label_sim)


def pairwise_ranking_loss(pred_scores, target_scores, min_gap=0.05):
    pred_diff = pred_scores.view(-1, 1) - pred_scores.view(1, -1)
    target_diff = target_scores.view(-1, 1) - target_scores.view(1, -1)
    sign = torch.sign(target_diff)
    mask = torch.abs(target_diff) > min_gap
    if not torch.any(mask):
        return pred_scores.new_tensor(0.0)
    return F.softplus(-sign[mask] * pred_diff[mask]).mean()


def metrics_from_arrays(preds, labels):
    return {
        "plcc": float(pearsonr(preds, labels)[0]),
        "srcc": float(spearmanr(preds, labels)[0]),
        "mae": float(mean_absolute_error(labels, preds)),
        "mse": float(mean_squared_error(labels, preds)),
    }


def long_tail_metrics(preds, labels, categories, config):
    out = {}
    label_low = labels <= config.low_cut
    label_high = labels >= config.high_cut
    label_mid = ~(label_low | label_high)
    for name, mask in [("LL", label_low), ("LM", label_mid), ("LH", label_high)]:
        out[f"{name}_count"] = int(mask.sum())
        out[f"{name}_mae"] = float(mean_absolute_error(labels[mask], preds[mask])) if mask.any() else None
    scene_tail = np.isin(categories, np.asarray(config.minority_scenes))
    for name, mask in [("SN", scene_tail), ("SJ", ~scene_tail)]:
        out[f"{name}_count"] = int(mask.sum())
        out[f"{name}_mae"] = float(mean_absolute_error(labels[mask], preds[mask])) if mask.any() else None
    return out


PAPER_ELTA2_AVA_TARGETS = {
    "P": 0.793,
    "S": 0.810,
    "LL": 0.412,
    "LM": 0.286,
    "LH": 0.410,
    "SN": 0.339,
    "SJ": 0.316,
}


def paper_metric_view(metrics):
    return {
        "P": metrics.get("plcc"),
        "S": metrics.get("srcc"),
        "LL": metrics.get("LL_mae"),
        "LM": metrics.get("LM_mae"),
        "LH": metrics.get("LH_mae"),
        "SN": metrics.get("SN_mae"),
        "SJ": metrics.get("SJ_mae"),
    }


def target_status(metrics, args):
    target_path = getattr(args, "target_metrics_json", None)
    if target_path:
        targets = json.loads(Path(target_path).read_text(encoding="utf-8"))
    else:
        targets = PAPER_ELTA2_AVA_TARGETS
    current = paper_metric_view(metrics)
    tolerances = {
        "P": float(args.target_corr_tol),
        "S": float(args.target_corr_tol),
        "LL": float(args.target_mae_tol),
        "LM": float(args.target_mae_tol),
        "LH": float(args.target_mae_tol),
        "SN": float(args.target_mae_tol),
        "SJ": float(args.target_mae_tol),
    }
    gaps = {}
    normalized = {}
    for key, target in targets.items():
        value = current.get(key)
        if value is None:
            gaps[key] = None
            normalized[key] = float("inf")
            continue
        if key in ("P", "S"):
            gap = max(0.0, float(target) - float(value))
        else:
            gap = max(0.0, float(value) - float(target))
        gaps[key] = gap
        normalized[key] = gap / max(tolerances[key], 1e-12)
    score = float(sum(v for v in normalized.values() if np.isfinite(v)))
    reached = all((gaps[k] is not None and gaps[k] <= tolerances[k]) for k in targets)
    return {
        "targets": targets,
        "current": current,
        "directional_gaps": gaps,
        "tolerances": tolerances,
        "normalized_gap_score": score,
        "target_reached": bool(reached),
    }


def save_target_artifacts(out_dir, ckpt_path, status, metrics, epoch, kind):
    target_dir = Path(out_dir) / "target_repro"
    target_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "epoch": int(epoch),
        "kind": kind,
        "metrics": metrics,
        "target_status": status,
        "checkpoint": str(target_dir / f"{kind}.pth"),
    }
    shutil.copy2(ckpt_path, target_dir / f"{kind}.pth")
    (target_dir / f"{kind}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_tail_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TailConfig(**data)


def load_checkpoint(model, ckpt_path, device, reset_optimizer=False, optimizer=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and not reset_optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def evaluate(model, loader, device, config=None, save_csv=None, max_batches=None):
    model.eval()
    preds, labels, categories, rows = [], [], [], []
    with torch.inference_mode():
        for batch_idx, (images, targets, scores, scene_idx, is_generated) in enumerate(tqdm(loader, desc="eval")):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            features, logits, outputs = model(images)
            prob = outputs.detach().cpu().numpy()
            logit_np = logits.detach().cpu().numpy()
            target_np = targets.detach().cpu().numpy()
            batch_preds = distribution_to_score_array(prob)
            batch_labels = distribution_to_score_array(target_np)
            start = len(preds)
            batch_df = loader.dataset.df.iloc[start:start + len(batch_preds)]
            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())
            categories.extend(batch_df["category"].astype(str).tolist())
            if save_csv:
                for (_, row), pred, label, p, z in zip(batch_df.iterrows(), batch_preds, batch_labels, prob, logit_np):
                    item = {"image_id": row["image_id"], "category": row.get("category", "unknown"), "gt": label, "score": pred}
                    item.update({c: float(v) for c, v in zip(PROB_COLS, p)})
                    item.update({c: float(v) for c, v in zip(LOGIT_COLS, z)})
                    rows.append(item)
            if max_batches and batch_idx + 1 >= max_batches:
                break
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    categories = np.asarray(categories)
    result = metrics_from_arrays(preds, labels)
    if config is not None:
        result.update(long_tail_metrics(preds, labels, categories, config))
    if save_csv:
        pd.DataFrame(rows).to_csv(save_csv, index=False)
    return result


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_tail_config(args.tail_config)

    train_df = pd.read_csv(args.train_csv)
    if args.extra_train_csv:
        extra = pd.read_csv(args.extra_train_csv)
        train_df = pd.concat([train_df, extra], ignore_index=True)
    merged_train = out_dir / "effective_train.csv"
    train_df.to_csv(merged_train, index=False)

    train_set = AVAELTA2Dataset(merged_train, args.image_dir, args.generated_dir, train=True)
    val_set = AVAELTA2Dataset(args.val_csv, args.image_dir, args.generated_dir, train=False)
    sampler = make_sampler(train_df, config, args) if args.scene_balanced_sampler else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    model = Swin(loss_type="emd", pretrained=args.pretrained).to(device)
    criterion = EMDLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_srcc, best_target_score = 0, -1.0, float("inf")
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, device, reset_optimizer=args.reset_epoch, optimizer=optimizer)
        if not args.reset_epoch:
            start_epoch = int(ckpt.get("epoch", 0))
            best_srcc = float(ckpt.get("best_metric", best_srcc))
            best_target_score = float(ckpt.get("best_target_score", best_target_score))

    print(f"device={device}, physical_gpu={args.gpu_id}, train={len(train_set)}, val={len(val_set)}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        meter = []
        for batch_idx, (images, targets, scores, scene_idx, is_generated) in enumerate(tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            features, logits, outputs = model(images)
            if args.tfa:
                features, targets = feature_tail_mixup(features, targets, scores, args.tau_1, args.tau_2, args.mixup_ratio)
                outputs = model.softmax(model.linear_emd(features))
            loss = criterion(outputs, targets)
            pred_scores_for_loss = torch_distribution_score(outputs)
            target_scores_for_loss = torch_distribution_score(targets)
            if args.score_loss_weight > 0:
                loss = loss + args.score_loss_weight * F.smooth_l1_loss(pred_scores_for_loss, target_scores_for_loss)
            if args.rank_loss_weight > 0:
                loss = loss + args.rank_loss_weight * pairwise_ranking_loss(pred_scores_for_loss, target_scores_for_loss)
            if args.flsa_weight > 0:
                loss = loss + args.flsa_weight * similarity_alignment_loss(features, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            meter.append(float(loss.detach().cpu()))
            if args.max_train_batches and batch_idx + 1 >= args.max_train_batches:
                break
        pred_csv = out_dir / f"val_epoch{epoch + 1}.csv"
        val_metrics = evaluate(model, val_loader, device, config, pred_csv, args.max_val_batches)
        val_metrics["epoch"] = epoch + 1
        val_metrics["train_loss"] = float(np.mean(meter)) if meter else None
        print(json.dumps(val_metrics, indent=2, ensure_ascii=False))
        (out_dir / "history.jsonl").open("a", encoding="utf-8").write(json.dumps(val_metrics, ensure_ascii=False) + "\n")
        is_best = val_metrics["srcc"] > best_srcc
        if is_best:
            best_srcc = val_metrics["srcc"]
        ckpt = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_srcc,
            "best_target_score": best_target_score,
            "tail_config": asdict(config),
        }
        ckpt_dir = out_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, ckpt_dir / "ckpt.pth")
        if is_best:
            torch.save(ckpt, ckpt_dir / f"best_epoch{epoch + 1}_srcc_{best_srcc:.4f}.pth")
        if args.stop_when_close:
            status = target_status(val_metrics, args)
            print(json.dumps({"target_status": status}, indent=2, ensure_ascii=False))
            if status["normalized_gap_score"] < best_target_score:
                best_target_score = status["normalized_gap_score"]
                ckpt["best_target_score"] = best_target_score
                torch.save(ckpt, ckpt_dir / "ckpt.pth")
                save_target_artifacts(out_dir, ckpt_dir / "ckpt.pth", status, val_metrics, epoch + 1, "best_target")
            if status["target_reached"]:
                save_target_artifacts(out_dir, ckpt_dir / "ckpt.pth", status, val_metrics, epoch + 1, "target_reached")
                print(f"target_reached=true, saved={out_dir / 'target_repro' / 'target_reached.pth'}")
                break


def run_eval(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_tail_config(args.tail_config) if args.tail_config else None
    dataset = AVAELTA2Dataset(args.csv, args.image_dir, args.generated_dir, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = Swin(loss_type="emd", pretrained=False).to(device)
    load_checkpoint(model, args.ckpt, device, reset_optimizer=True)
    metrics = evaluate(model, loader, device, config, args.output_csv, args.max_batches)
    metrics_path = Path(args.output_csv).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"predictions={args.output_csv}")
    print(f"metrics={metrics_path}")


def select_pseudo(args):
    pred = pd.read_csv(args.predictions_csv)
    if not all(c in pred.columns for c in LOGIT_COLS):
        raise ValueError("predictions_csv must contain logit_1 ... logit_10 columns")
    beta_values = [float(x) for x in args.beta_values.split(",")]
    threshold_values = [float(x) for x in args.threshold_values.split(",")]
    logits = pred[LOGIT_COLS].to_numpy(dtype=np.float64)
    score = pred["score"].to_numpy(dtype=np.float64)
    mean_score = score.mean()

    best = None
    for beta in beta_values:
        tau = np.exp(-beta * np.abs(score - mean_score))
        prob = softmax(logits / np.clip(tau[:, None], 1e-6, None), axis=1)
        conf = prob.max(axis=1)
        pred_score = distribution_to_score_array(prob)
        for threshold in threshold_values:
            mask = conf >= threshold
            if mask.sum() < args.min_samples:
                continue
            mae = mean_absolute_error(pred.loc[mask, "gt"], pred_score[mask]) if "gt" in pred.columns else 0.0
            candidate = (mae, -int(mask.sum()), beta, threshold, prob, conf, pred_score, mask)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No APDS setting selected enough samples")
    mae, neg_count, beta, threshold, prob, conf, pred_score, mask = best
    selected = pred.loc[mask, ["image_id", "category"]].copy()
    selected["score"] = pred_score[mask]
    selected["apds_conf"] = conf[mask]
    selected["is_generated"] = 1
    selected[SCORE_COLS] = prob[mask]
    if "image_path" in pred.columns:
        selected["image_path"] = pred.loc[mask, "image_path"].to_numpy()
    selected.to_csv(args.output_csv, index=False)
    print(json.dumps({"beta": beta, "threshold": threshold, "count": int(mask.sum()), "mae": float(mae)}, indent=2))
    print(f"pseudo_csv={args.output_csv}")


def parse_args():
    parser = argparse.ArgumentParser("AVA-only ELTA 2.0 implementation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--train_label_csv", default="/root/autodl-tmp/ELTA/AVA/label/train.csv")
    p.add_argument("--test_label_csv", default="/root/autodl-tmp/ELTA/AVA/label/test.csv")
    p.add_argument("--train_scene_csv", default="/root/autodl-tmp/ELTA/Long-Tail-image-aesthetics-and-quality-assessment-main/ELTA 2.0/code/dataset_backup/AVA/train.csv")
    p.add_argument("--test_scene_csv", default="/root/autodl-tmp/ELTA/Long-Tail-image-aesthetics-and-quality-assessment-main/ELTA 2.0/code/dataset_backup/AVA/test.csv")
    p.add_argument("--scene_ratio_cut", type=float, default=0.2)
    p.add_argument("--scene_bottom_percent", type=float, default=20.0)
    p.add_argument("--output_dir", default="/root/autodl-tmp/ELTA/elta2_ava_impl")
    p.set_defaults(func=build_metadata)

    p = sub.add_parser("train")
    p.add_argument("--train_csv", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_train_meta.csv")
    p.add_argument("--val_csv", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_test_meta.csv")
    p.add_argument("--tail_config", default="/root/autodl-tmp/ELTA/elta2_ava_impl/tail_config.json")
    p.add_argument("--image_dir", default="/root/autodl-tmp/ELTA/AVA/AVA_dataset/image")
    p.add_argument("--generated_dir", default="/root/autodl-tmp/ELTA/Long-Tail-image-aesthetics-and-quality-assessment-main/ELTA 2.0/code/diffusion/result/AVA/generated")
    p.add_argument("--extra_train_csv", default=None)
    p.add_argument("--output_dir", default="/root/autodl-tmp/ELTA/elta2_ava_impl/run")
    p.add_argument("--gpu_id", default="1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--reset_epoch", action="store_true")
    p.add_argument("--scene_balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--label_tail_weight", type=float, default=1.0)
    p.add_argument("--scene_tail_weight", type=float, default=2.0)
    p.add_argument("--joint_tail_weight", type=float, default=2.0)
    p.add_argument("--generated_weight", type=float, default=2.0)
    p.add_argument("--sample_multiplier", type=float, default=1.0)
    p.add_argument("--tfa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tau_1", type=float, default=0.5)
    p.add_argument("--tau_2", type=float, default=2.0)
    p.add_argument("--mixup_ratio", type=float, default=0.25)
    p.add_argument("--flsa_weight", type=float, default=0.0)
    p.add_argument("--score_loss_weight", type=float, default=0.0)
    p.add_argument("--rank_loss_weight", type=float, default=0.0)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument("--stop_when_close", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--target_metrics_json", default=None)
    p.add_argument("--target_corr_tol", type=float, default=0.010)
    p.add_argument("--target_mae_tol", type=float, default=0.020)
    p.set_defaults(func=train)

    p = sub.add_parser("eval")
    p.add_argument("--csv", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_test_meta.csv")
    p.add_argument("--tail_config", default="/root/autodl-tmp/ELTA/elta2_ava_impl/tail_config.json")
    p.add_argument("--image_dir", default="/root/autodl-tmp/ELTA/AVA/AVA_dataset/image")
    p.add_argument("--generated_dir", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_csv", default="/root/autodl-tmp/ELTA/elta2_ava_impl/predictions.csv")
    p.add_argument("--gpu_id", default="1")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=None)
    p.set_defaults(func=run_eval)

    p = sub.add_parser("select-pseudo")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--output_csv", default="/root/autodl-tmp/ELTA/elta2_ava_impl/pseudo_selected.csv")
    p.add_argument("--beta_values", default="0.5,1.0,1.25,1.5,2.0")
    p.add_argument("--threshold_values", default="0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--min_samples", type=int, default=10)
    p.set_defaults(func=select_pseudo)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
