import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_path_for_id(image_dir, image_id):
    try:
        name = f"{int(image_id)}.jpg"
    except Exception:
        name = str(image_id)
    return Path(image_dir) / name


def normalize_score(score):
    return (float(score) - 5.5) / 4.5


def tensor_to_save_range(x):
    return (x.clamp(-1, 1) + 1.0) * 0.5


class AVAGeneratorDataset(Dataset):
    def __init__(self, meta_csv, image_dir, scene_to_idx=None, image_size=128, max_samples=None):
        self.df = pd.read_csv(meta_csv)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=2026).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        if scene_to_idx is None:
            scenes = sorted(str(x) for x in self.df["category"].fillna("unknown").unique())
            scene_to_idx = {scene: idx for idx, scene in enumerate(scenes)}
        self.scene_to_idx = scene_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = default_loader(image_path_for_id(self.image_dir, row["image_id"]))
        scene = str(row.get("category", "unknown"))
        scene_idx = self.scene_to_idx.get(scene, self.scene_to_idx.get("unknown", 0))
        score = normalize_score(row["score"])
        return {
            "image": self.transform(image),
            "scene_idx": torch.tensor(scene_idx, dtype=torch.long),
            "score": torch.tensor([score], dtype=torch.float32),
        }


class PlanGenerationDataset(Dataset):
    def __init__(self, plan_csv, scene_to_idx, image_size=128):
        self.df = pd.read_csv(plan_csv)
        self.scene_to_idx = scene_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = default_loader(row["source_path"])
        scene = str(row["target_scene"])
        scene_idx = self.scene_to_idx.get(scene, self.scene_to_idx.get("unknown", 0))
        return {
            "image": self.transform(image),
            "scene_idx": torch.tensor(scene_idx, dtype=torch.long),
            "score": torch.tensor([normalize_score(row["anchor_score"])], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


class FiLMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.cond = nn.Linear(cond_dim, out_channels * 4)

    def forward(self, x, cond):
        scale1, bias1, scale2, bias2 = self.cond(cond).chunk(4, dim=1)
        scale1 = scale1[:, :, None, None]
        bias1 = bias1[:, :, None, None]
        scale2 = scale2[:, :, None, None]
        bias2 = bias2[:, :, None, None]
        x = self.conv1(x)
        x = self.norm1(x) * (1 + scale1) + bias1
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x) * (1 + scale2) + bias2
        return F.silu(x)


class ConditionalUNetGenerator(nn.Module):
    def __init__(self, num_scenes, base_channels=32, scene_dim=64, cond_dim=128):
        super().__init__()
        self.scene_embed = nn.Embedding(num_scenes, scene_dim)
        self.cond = nn.Sequential(
            nn.Linear(scene_dim + 1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        c = base_channels
        self.enc1 = FiLMBlock(3, c, cond_dim)
        self.enc2 = FiLMBlock(c, c * 2, cond_dim)
        self.enc3 = FiLMBlock(c * 2, c * 4, cond_dim)
        self.mid = FiLMBlock(c * 4, c * 4, cond_dim)
        self.dec2 = FiLMBlock(c * 6, c * 2, cond_dim)
        self.dec1 = FiLMBlock(c * 3, c, cond_dim)
        self.out = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, 3, 3, padding=1),
            nn.Tanh(),
        )

    def condition(self, scene_idx, score):
        return self.cond(torch.cat([self.scene_embed(scene_idx), score], dim=1))

    def forward(self, x, scene_idx, score):
        cond = self.condition(scene_idx, score)
        e1 = self.enc1(x, cond)
        e2 = self.enc2(F.avg_pool2d(e1, 2), cond)
        e3 = self.enc3(F.avg_pool2d(e2, 2), cond)
        mid = self.mid(e3, cond)
        d2 = F.interpolate(mid, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond)
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), cond)
        return self.out(d1)


def make_corrupted_input(clean, min_noise=0.05, max_noise=0.35, drop_prob=0.15):
    b = clean.size(0)
    noise_level = torch.empty(b, 1, 1, 1, device=clean.device).uniform_(min_noise, max_noise)
    noisy = clean + noise_level * torch.randn_like(clean)
    if drop_prob > 0:
        mask = torch.rand(b, 1, 1, 1, device=clean.device) < drop_prob
        noisy = torch.where(mask, torch.randn_like(clean), noisy)
    return noisy.clamp(-1, 1)


def total_variation_loss(x):
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


def save_checkpoint(path, model, optimizer, epoch, scene_to_idx, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scene_to_idx": scene_to_idx,
        "args": vars(args),
    }, path)


def load_generator_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    scene_to_idx = {str(k): int(v) for k, v in ckpt["scene_to_idx"].items()}
    train_args = ckpt.get("args", {})
    model = ConditionalUNetGenerator(
        num_scenes=max(scene_to_idx.values()) + 1,
        base_channels=int(train_args.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, scene_to_idx, train_args


def train_generator(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = AVAGeneratorDataset(
        args.train_meta,
        args.image_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = ConditionalUNetGenerator(
        num_scenes=max(dataset.scene_to_idx.values()) + 1,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "scene_to_idx.json").write_text(json.dumps(dataset.scene_to_idx, indent=2), encoding="utf-8")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"generator-train:{epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(progress, start=1):
            clean = batch["image"].to(device, non_blocking=True)
            scene_idx = batch["scene_idx"].to(device, non_blocking=True)
            score = batch["score"].to(device, non_blocking=True)
            noisy = make_corrupted_input(clean, args.min_noise, args.max_noise, args.drop_prob)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
                pred = model(noisy, scene_idx, score)
                l1 = F.l1_loss(pred, clean)
                mse = F.mse_loss(pred, clean)
                tv = total_variation_loss(pred)
                loss = l1 + args.mse_weight * mse + args.tv_weight * tv
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().cpu())
            global_step += 1
            progress.set_postfix(loss=running / batch_idx)
            if args.debug_max_batches and batch_idx >= args.debug_max_batches:
                break

        save_checkpoint(out_dir / "last.ckpt", model, optimizer, epoch, dataset.scene_to_idx, args)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.ckpt", model, optimizer, epoch, dataset.scene_to_idx, args)
        print(json.dumps({"epoch": epoch, "loss": running / max(1, batch_idx), "ckpt": str(out_dir / "last.ckpt")}))


def generate_from_plan(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, scene_to_idx, train_args = load_generator_checkpoint(args.ckpt, device)
    image_size = int(args.image_size or train_args.get("image_size", 128))
    dataset = PlanGenerationDataset(args.plan_csv, scene_to_idx=scene_to_idx, image_size=image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    plan = dataset.df.copy()
    statuses = ["planned"] * len(plan)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="generator-sample"):
            source = batch["image"].to(device, non_blocking=True)
            scene_idx = batch["scene_idx"].to(device, non_blocking=True)
            score = batch["score"].to(device, non_blocking=True)
            indices = batch["index"].cpu().numpy().tolist()
            noise = torch.randn_like(source)
            x = (1 - args.source_strength) * noise + args.source_strength * source
            x = x.clamp(-1, 1)
            for _ in range(max(1, args.refine_steps)):
                pred = model(x, scene_idx, score)
                x = (args.refine_blend * pred + (1 - args.refine_blend) * x).clamp(-1, 1)
            images = tensor_to_save_range(x)
            for local_i, plan_i in enumerate(indices):
                output_path = Path(plan.iloc[plan_i]["output_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and not args.overwrite:
                    statuses[plan_i] = "exists"
                    continue
                to_pil_image(images[local_i].cpu()).save(output_path, quality=95)
                statuses[plan_i] = "generated"

    plan["status"] = statuses
    output_plan = args.output_plan_csv or args.plan_csv
    plan.to_csv(output_plan, index=False)
    print(f"updated_plan={output_plan}")
    print(f"generated={sum(s == 'generated' for s in statuses)}")


def parse_args():
    parser = argparse.ArgumentParser("AVA self-trained ELTA 2.0 image generator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--train_meta", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_train_meta.csv")
    p.add_argument("--image_dir", default="/root/autodl-tmp/ELTA/AVA/AVA_dataset/image")
    p.add_argument("--output_dir", default="/root/autodl-tmp/ELTA/elta2_ava_generator")
    p.add_argument("--gpu_id", default="1")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=0.25)
    p.add_argument("--tv_weight", type=float, default=0.02)
    p.add_argument("--min_noise", type=float, default=0.05)
    p.add_argument("--max_noise", type=float, default=0.35)
    p.add_argument("--drop_prob", type=float, default=0.15)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--debug_max_batches", type=int, default=None)
    p.add_argument("--amp", action="store_true", default=True)
    p.set_defaults(func=train_generator)

    p = sub.add_parser("generate")
    p.add_argument("--plan_csv", default="/root/autodl-tmp/ELTA/elta2_ava_generation/generation_plan.csv")
    p.add_argument("--output_plan_csv", default=None)
    p.add_argument("--ckpt", default="/root/autodl-tmp/ELTA/elta2_ava_generator/last.ckpt")
    p.add_argument("--gpu_id", default="1")
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--source_strength", type=float, default=0.45)
    p.add_argument("--refine_steps", type=int, default=2)
    p.add_argument("--refine_blend", type=float, default=0.85)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=generate_from_plan)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
