#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.atas import CLIPVisionWrapper


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]


DEFAULT_PROMPTS = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a clean photo of a {}.",
    "a close-up photo of a {}.",
]


def parse_args():
    parser = argparse.ArgumentParser("VOC2012 Open-Vocabulary Segmentation Evaluation")

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--voc-root", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)

    return parser.parse_args()


def load_cfg(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.voc_root is not None:
        cfg["dataset"]["voc_root"] = args.voc_root

    if args.ckpt is not None:
        cfg["model"]["ckpt"] = args.ckpt

    if args.device is not None:
        cfg["eval"]["device"] = args.device

    if args.image_size is not None:
        cfg["model"]["image_size"] = args.image_size

    return cfg


def build_text_features(model, tokenizer, class_names, prompts, device):
    all_features = []

    model.eval()

    with torch.no_grad():
        for name in class_names:
            texts = [p.format(name) for p in prompts]
            tokens = tokenizer(texts).to(device)

            text_feat = model.encode_text(tokens)
            text_feat = F.normalize(text_feat.float(), dim=-1)

            text_feat = text_feat.mean(dim=0)
            text_feat = F.normalize(text_feat, dim=-1)

            all_features.append(text_feat)

    return torch.stack(all_features, dim=0)


def fast_hist(pred, target, num_classes):
    valid = target != 255

    pred = pred[valid]
    target = target[valid]

    valid = (target >= 0) & (target < num_classes)

    pred = pred[valid]
    target = target[valid]

    hist = np.bincount(
        num_classes * target.astype(np.int64) + pred.astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    return hist


def compute_iou(hist):
    diag = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - diag
    iou = diag / np.maximum(union, 1)
    return iou


def colorize_mask(mask):
    palette = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ],
        dtype=np.uint8,
    )

    return palette[mask]


def main():
    args = parse_args()
    cfg = load_cfg(args)

    voc_root = Path(cfg["dataset"]["voc_root"])
    split = cfg["dataset"].get("split", "val")

    model_name = cfg["model"].get("model_name", "ViT-B-16")
    ckpt = cfg["model"]["ckpt"]
    image_size = int(cfg["model"].get("image_size", 224))

    device_name = cfg["eval"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    save_pred = bool(cfg["eval"].get("save_pred", False))
    out_dir = Path(cfg["eval"].get("out_dir", "outputs/voc2012_ovseg"))
    ignore_bg_miou = bool(cfg["eval"].get("ignore_bg_miou", True))

    prompts = cfg.get("prompts", DEFAULT_PROMPTS)

    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"

    assert jpeg_dir.exists(), f"Missing JPEGImages dir: {jpeg_dir}"
    assert mask_dir.exists(), f"Missing SegmentationClass dir: {mask_dir}"
    assert split_file.exists(), f"Missing split file: {split_file}"
    assert Path(ckpt).exists(), f"Missing checkpoint: {ckpt}"

    ids = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]

    print("===== VOC2012 OVSeg Eval Config =====")
    print(f"VOC root        : {voc_root}")
    print(f"Split           : {split}")
    print(f"Model name      : {model_name}")
    print(f"Checkpoint      : {ckpt}")
    print(f"Image size      : {image_size}")
    print(f"Device          : {device}")
    print(f"Save prediction : {save_pred}")
    print(f"Out dir         : {out_dir}")
    print(f"Ignore bg mIoU  : {ignore_bg_miou}")
    print("=====================================")

    vision_model = CLIPVisionWrapper(
        model_name=model_name,
        pretrained_path=ckpt,
    ).to(device).eval()

    clip_model = vision_model.model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    text_features = build_text_features(
        clip_model,
        tokenizer,
        VOC_CLASSES,
        prompts,
        device,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    if save_pred:
        os.makedirs(out_dir, exist_ok=True)

    hist = np.zeros(
        (len(VOC_CLASSES), len(VOC_CLASSES)),
        dtype=np.float64,
    )

    for img_id in tqdm(ids):
        image_path = jpeg_dir / f"{img_id}.jpg"
        mask_path = mask_dir / f"{img_id}.png"

        image = Image.open(image_path).convert("RGB")
        target = np.array(Image.open(mask_path), dtype=np.uint8)

        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            _, patch = vision_model.forward_features(x)

            patch = F.normalize(patch.float(), dim=-1)

            logits = patch @ text_features.t()

            n_patch = patch.shape[1]
            h = w = int(n_patch ** 0.5)

            assert h * w == n_patch, f"Patch number is not square: {n_patch}"

            logits = logits.permute(0, 2, 1).reshape(
                1,
                len(VOC_CLASSES),
                h,
                w,
            )

            logits = F.interpolate(
                logits,
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )

            pred = logits.argmax(dim=1)[0]
            pred = pred.cpu().numpy().astype(np.uint8)

        hist += fast_hist(pred, target, len(VOC_CLASSES))

        if save_pred:
            color = colorize_mask(pred)
            Image.fromarray(color).save(out_dir / f"{img_id}.png")

    iou = compute_iou(hist)

    if ignore_bg_miou:
        miou = np.nanmean(iou[1:])
    else:
        miou = np.nanmean(iou)

    print("\n===== VOC2012 OVSeg Result =====")

    for name, score in zip(VOC_CLASSES, iou):
        print(f"{name:15s}: {score * 100:.2f}")

    print("--------------------------------")
    print(f"mIoU: {miou * 100:.2f}")

    if ignore_bg_miou:
        print("Note: background excluded from mIoU.")
    else:
        print("Note: background included in mIoU.")


if __name__ == "__main__":
    main()