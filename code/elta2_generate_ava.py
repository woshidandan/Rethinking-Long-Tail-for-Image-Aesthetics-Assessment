import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFile
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from models.swin_model import Swin


ImageFile.LOAD_TRUNCATED_IMAGES = True
SCORE_COLS = [f"score{i}" for i in range(2, 12)]
PROB_COLS = [f"prob_{i}" for i in range(1, 11)]
LOGIT_COLS = [f"logit_{i}" for i in range(1, 11)]


@dataclass
class GenerationBackendConfig:
    backend: str
    model_id: str | None
    device: str
    torch_dtype: str
    num_inference_steps: int
    guidance_scale: float
    image_guidance_scale: float
    negative_prompt: str


def distribution_to_score_array(dist):
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist / np.clip(dist.sum(axis=1, keepdims=True), 1e-12, None)
    return (dist * np.arange(1, 11, dtype=np.float64)).sum(axis=1)


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def clean_scene_name(scene):
    return str(scene).replace("/", " ").replace("_", " ").strip()


def image_path_for_id(image_dir, image_id):
    try:
        name = f"{int(image_id)}.jpg"
    except Exception:
        name = str(image_id)
    return Path(image_dir) / name


def load_tail_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_generation_plan(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    image_out_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_meta)
    config = load_tail_config(args.tail_config)
    minority_scenes = config["minority_scenes"]
    if args.max_scenes:
        minority_scenes = minority_scenes[: args.max_scenes]
    anchors = parse_float_list(args.anchor_scores)

    if args.source_policy == "label_tail":
        low_cut, high_cut = config["low_cut"], config["high_cut"]
        source_pool = train[(train["score"] <= low_cut) | (train["score"] >= high_cut)].copy()
    elif args.source_policy == "scene_tail":
        source_pool = train[train["category"].isin(minority_scenes)].copy()
    else:
        source_pool = train.copy()
    if source_pool.empty:
        raise ValueError(f"source pool is empty for policy {args.source_policy}")

    rows = []
    job_idx = 0
    for scene in minority_scenes:
        scene_slug = str(scene).replace("/", "__").replace(" ", "_")
        scene_dir = image_out_dir / scene_slug
        scene_dir.mkdir(parents=True, exist_ok=True)
        for anchor in anchors:
            for round_idx in range(args.samples_per_scene_anchor):
                source = source_pool.sample(n=1, random_state=args.seed + job_idx).iloc[0]
                source_path = image_path_for_id(args.image_dir, source["image_id"])
                prompt = args.prompt_template.format(scene=clean_scene_name(scene), score=anchor)
                out_name = f"elta2_ava_{job_idx:07d}_scene-{scene_slug}_score-{anchor:g}.jpg"
                rows.append({
                    "job_id": job_idx,
                    "source_image_id": source["image_id"],
                    "source_path": str(source_path),
                    "source_score": float(source["score"]),
                    "source_category": source.get("category", "unknown"),
                    "target_scene": scene,
                    "anchor_score": float(anchor),
                    "prompt": prompt,
                    "seed": args.seed + job_idx,
                    "output_path": str(scene_dir / out_name),
                    "status": "planned",
                })
                job_idx += 1
    plan = pd.DataFrame(rows)
    plan_path = out_dir / "generation_plan.csv"
    plan.to_csv(plan_path, index=False)
    print(f"plan={plan_path}")
    print(f"jobs={len(plan)}, scenes={len(minority_scenes)}, anchors={anchors}")


class ImageGenerationBackend:
    def __init__(self, config: GenerationBackendConfig):
        self.config = config
        self.pipe = None
        if config.backend in {"instruct_pix2pix", "sd3_ultraedit"}:
            self._load_pipeline()

    def _dtype(self):
        if self.config.torch_dtype == "float32":
            return torch.float32
        if self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def _load_pipeline(self):
        if not self.config.model_id:
            raise ValueError(f"--model_id is required for backend {self.config.backend}")
        if self.config.backend == "instruct_pix2pix":
            from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=self._dtype(),
                safety_checker=None,
                use_safetensors=True,
            ).to(self.config.device)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif self.config.backend == "sd3_ultraedit":
            from diffusers import StableDiffusion3InstructPix2PixPipeline

            self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=self._dtype(),
            ).to(self.config.device)

    def generate(self, source_path, prompt, seed, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.backend == "copy":
            img = Image.open(source_path).convert("RGB").resize((512, 512))
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 512, 38), fill=(0, 0, 0))
            draw.text((8, 10), prompt[:92], fill=(255, 255, 255))
            img.save(output_path, quality=95)
            return

        generator = torch.Generator(device=self.config.device).manual_seed(int(seed))
        if self.config.backend == "instruct_pix2pix":
            image = Image.open(source_path).convert("RGB").resize((512, 512))
            result = self.pipe(
                prompt=prompt,
                image=image,
                negative_prompt=self.config.negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                image_guidance_scale=self.config.image_guidance_scale,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            ).images[0]
        elif self.config.backend == "sd3_ultraedit":
            image = Image.open(source_path).convert("RGB").resize((512, 512))
            mask_img = Image.new("RGB", image.size, (255, 255, 255))
            result = self.pipe(
                prompt=prompt,
                image=image,
                mask_img=mask_img,
                negative_prompt=self.config.negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                image_guidance_scale=self.config.image_guidance_scale,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            ).images[0]
        else:
            raise ValueError(f"Unsupported backend {self.config.backend}")
        result.save(output_path, quality=95)


def generate_from_plan(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0" if torch.cuda.is_available() and args.backend != "copy" else "cpu"
    plan = pd.read_csv(args.plan_csv)
    backend = ImageGenerationBackend(
        GenerationBackendConfig(
            backend=args.backend,
            model_id=args.model_id,
            device=device,
            torch_dtype=args.torch_dtype,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            negative_prompt=args.negative_prompt,
        )
    )
    statuses = []
    generated = 0
    for _, row in tqdm(plan.iterrows(), total=len(plan), desc=f"generate:{args.backend}"):
        output_path = Path(row["output_path"])
        if output_path.exists() and not args.overwrite:
            statuses.append("exists")
            continue
        try:
            backend.generate(row["source_path"], row["prompt"], row["seed"], output_path)
            statuses.append("generated")
            generated += 1
        except Exception as exc:
            statuses.append(f"error:{type(exc).__name__}:{str(exc)[:160]}")
            if not args.keep_going:
                raise
    plan["status"] = statuses
    output_plan = args.output_plan_csv or args.plan_csv
    plan.to_csv(output_plan, index=False)
    print(f"generated={generated}")
    print(f"updated_plan={output_plan}")


class GeneratedImageDataset(Dataset):
    def __init__(self, plan_csv):
        self.df = pd.read_csv(plan_csv)
        self.df = self.df[self.df["output_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = default_loader(row["output_path"])
        return self.transform(image), index


def score_generated_images(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = GeneratedImageDataset(args.plan_csv)
    if len(dataset) == 0:
        raise ValueError("No generated images found in plan")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = Swin(loss_type="emd", pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="score-generated"):
            images = images.to(device, non_blocking=True)
            _, logits, outputs = model(images)
            prob = outputs.detach().cpu().numpy()
            logit_np = logits.detach().cpu().numpy()
            pred_score = distribution_to_score_array(prob)
            for idx, score, p, z in zip(indices.numpy(), pred_score, prob, logit_np):
                meta = dataset.df.iloc[int(idx)].to_dict()
                row = {
                    "job_id": meta["job_id"],
                    "image_id": Path(meta["output_path"]).name,
                    "image_path": meta["output_path"],
                    "category": meta["target_scene"],
                    "anchor_score": float(meta["anchor_score"]),
                    "gt": float(meta["anchor_score"]),
                    "score": float(score),
                    "source_image_id": meta["source_image_id"],
                    "prompt": meta["prompt"],
                }
                row.update({c: float(v) for c, v in zip(PROB_COLS, p)})
                row.update({c: float(v) for c, v in zip(LOGIT_COLS, z)})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"scored={len(out)}")
    print(f"predictions={args.output_csv}")


def select_generated_pseudo(args):
    pred = pd.read_csv(args.predictions_csv)
    logits = pred[LOGIT_COLS].to_numpy(dtype=np.float64)
    scores = pred["score"].to_numpy(dtype=np.float64)
    mean_score = scores.mean()
    beta_values = parse_float_list(args.beta_values)
    threshold_values = parse_float_list(args.threshold_values)

    best = None
    for beta in beta_values:
        tau = np.exp(-beta * np.abs(scores - mean_score))
        prob = softmax(logits / np.clip(tau[:, None], 1e-6, None), axis=1)
        conf = prob.max(axis=1)
        pseudo_score = distribution_to_score_array(prob)
        anchor_error = np.abs(pseudo_score - pred["anchor_score"].to_numpy(dtype=np.float64))
        for threshold in threshold_values:
            mask = (conf >= threshold) & (anchor_error <= args.max_anchor_error)
            if mask.sum() < args.min_samples:
                continue
            mae = float(anchor_error[mask].mean())
            candidate = (mae, -int(mask.sum()), beta, threshold, prob, conf, pseudo_score, mask)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No generated pseudo-labels passed APDS filters")
    mae, neg_count, beta, threshold, prob, conf, pseudo_score, mask = best
    selected = pred.loc[mask, ["image_id", "image_path", "category"]].copy()
    selected["score"] = pseudo_score[mask]
    selected["apds_conf"] = conf[mask]
    selected["anchor_score"] = pred.loc[mask, "anchor_score"].to_numpy()
    selected["is_generated"] = 1
    selected[SCORE_COLS] = prob[mask]
    selected.to_csv(args.output_csv, index=False)
    summary = {"beta": beta, "threshold": threshold, "max_anchor_error": args.max_anchor_error, "count": int(mask.sum()), "anchor_mae": mae}
    Path(args.output_csv).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"pseudo_csv={args.output_csv}")


def parse_args():
    parser = argparse.ArgumentParser("ELTA 2.0 AVA generation framework")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--train_meta", default="/root/autodl-tmp/ELTA/elta2_ava_impl/ava_train_meta.csv")
    p.add_argument("--tail_config", default="/root/autodl-tmp/ELTA/elta2_ava_impl/tail_config.json")
    p.add_argument("--image_dir", default="/root/autodl-tmp/ELTA/AVA/AVA_dataset/image")
    p.add_argument("--output_dir", default="/root/autodl-tmp/ELTA/elta2_ava_generation")
    p.add_argument("--anchor_scores", default="3,5,7")
    p.add_argument("--samples_per_scene_anchor", type=int, default=1)
    p.add_argument("--max_scenes", type=int, default=10)
    p.add_argument("--source_policy", choices=["any", "label_tail", "scene_tail"], default="any")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--prompt_template", default="Change the scene to {scene} with an aesthetic score of {score}.")
    p.set_defaults(func=build_generation_plan)

    p = sub.add_parser("generate")
    p.add_argument("--plan_csv", default="/root/autodl-tmp/ELTA/elta2_ava_generation/generation_plan.csv")
    p.add_argument("--output_plan_csv", default=None)
    p.add_argument("--backend", choices=["copy", "instruct_pix2pix", "sd3_ultraedit"], default="copy")
    p.add_argument("--model_id", default=None)
    p.add_argument("--gpu_id", default="1")
    p.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--image_guidance_scale", type=float, default=1.5)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep_going", action="store_true")
    p.set_defaults(func=generate_from_plan)

    p = sub.add_parser("score")
    p.add_argument("--plan_csv", default="/root/autodl-tmp/ELTA/elta2_ava_generation/generation_plan.csv")
    p.add_argument("--ckpt", default="/root/autodl-tmp/ELTA/Long-Tail-image-aesthetics-and-quality-assessment-main/ELTA 1.0/ava.ckpt.pth")
    p.add_argument("--output_csv", default="/root/autodl-tmp/ELTA/elta2_ava_generation/generated_predictions.csv")
    p.add_argument("--gpu_id", default="1")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.set_defaults(func=score_generated_images)

    p = sub.add_parser("select-pseudo")
    p.add_argument("--predictions_csv", default="/root/autodl-tmp/ELTA/elta2_ava_generation/generated_predictions.csv")
    p.add_argument("--output_csv", default="/root/autodl-tmp/ELTA/elta2_ava_generation/generated_pseudo_selected.csv")
    p.add_argument("--beta_values", default="0.5,1.0,1.25,1.5,2.0")
    p.add_argument("--threshold_values", default="0.50,0.60,0.70,0.75,0.80,0.85")
    p.add_argument("--max_anchor_error", type=float, default=1.5)
    p.add_argument("--min_samples", type=int, default=1)
    p.set_defaults(func=select_generated_pseudo)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
