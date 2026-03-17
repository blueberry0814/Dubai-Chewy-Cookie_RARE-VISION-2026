"""
Step 1: Extract DINOv2 features for all videos.

Usage:
  python extract_features.py --labels_dir ./data/galar_dataset/Labels \
                             --frames_root ./data/galar_dataset \
                             --output_dir  ./features \
                             --model dinov2_vits14

Output (per video):
  features/{video_id}_features.npy  : [N, D] float32  DINOv2 features
  features/{video_id}_frames.npy    : [N] int64        actual frame numbers (from CSV 'frame' column)

Notes:
  - Run this for both training and test data before training/inference
  - GPU recommended; falls back to CPU automatically
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def get_transform(image_size: int = 336) -> transforms.Compose:
    """DINOv2 recommended preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def find_frame_path(frames_root: str, video_id: str, frame_num: int) -> str | None:
    """
    Locate a frame image file, supporting two directory layouts:

      Layout A) frames_root/{video_id}/frame_{frame_num:06d}.PNG
      Layout B) frames_root/Galar_Frames_*/recording_{video_id}/frame_{frame_num:06d}.PNG
    """
    # Layout A (simple local structure)
    path_a = os.path.join(frames_root, str(video_id), f"frame_{frame_num:06d}.PNG")
    if os.path.exists(path_a):
        return path_a

    # Layout B (full Galar server structure)
    for frames_dir in sorted(Path(frames_root).glob("Galar_Frames_*")):
        path_b = frames_dir / f"recording_{video_id}" / f"frame_{frame_num:06d}.PNG"
        if path_b.exists():
            return str(path_b)

    return None


def extract_video_features(
    video_id: str,
    label_path: str,
    frames_root: str,
    output_dir: str,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 64,
    image_size: int = 336,
):
    """Extract and save DINOv2 features for a single video."""
    feat_out  = os.path.join(output_dir, f"{video_id}_features.npy")
    frame_out = os.path.join(output_dir, f"{video_id}_frames.npy")

    if os.path.exists(feat_out) and os.path.exists(frame_out):
        print(f"[SKIP] {video_id}: already extracted")
        return

    df = pd.read_csv(label_path)
    df = df.sort_values("frame").reset_index(drop=True)
    frame_nums = df["frame"].values.astype(np.int64)

    all_features = []
    valid_frame_nums = []
    batch_imgs = []
    batch_fnums = []

    for frame_num in tqdm(frame_nums, desc=f"  [{video_id}]", leave=False):
        img_path = find_frame_path(frames_root, video_id, frame_num)
        if img_path is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(transform(img))
            batch_fnums.append(frame_num)
        except Exception as e:
            print(f"  Warning: failed to load {img_path} ({e})")
            continue

        if len(batch_imgs) == batch_size:
            feats = _extract_batch(model, batch_imgs, device)
            all_features.append(feats)
            valid_frame_nums.extend(batch_fnums)
            batch_imgs.clear()
            batch_fnums.clear()

    # Process remaining batch
    if batch_imgs:
        feats = _extract_batch(model, batch_imgs, device)
        all_features.append(feats)
        valid_frame_nums.extend(batch_fnums)

    if not all_features:
        print(f"  [WARNING] {video_id}: no valid frames found, skipping")
        return

    features_arr = np.vstack(all_features).astype(np.float32)  # [N, D]
    frame_arr    = np.array(valid_frame_nums, dtype=np.int64)   # [N]

    np.save(feat_out,  features_arr)
    np.save(frame_out, frame_arr)
    print(f"  Saved: {video_id}  frames={len(frame_arr)}  shape={features_arr.shape}")


@torch.no_grad()
def _extract_batch(model: nn.Module, imgs: list, device: torch.device) -> np.ndarray:
    tensor = torch.stack(imgs).to(device)   # [B, C, H, W]
    feats  = model(tensor)                  # [B, D]
    return feats.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features")
    parser.add_argument("--labels_dir",  type=str, default="./data/galar_dataset/Labels",
                        help="Directory containing label CSV files ({video_id}.csv)")
    parser.add_argument("--frames_root", type=str, default="./data/galar_dataset",
                        help="Root directory containing video frame images")
    parser.add_argument("--output_dir",  type=str, default="./features")
    parser.add_argument("--model",       type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14"],
                        help="dinov2_vits14 (384-dim) or dinov2_vitb14 (768-dim)")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--image_size",  type=int, default=336)
    parser.add_argument("--video_ids",   type=str, default=None,
                        help="Comma-separated list of video IDs to process (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading DINOv2 model: {args.model} ...")
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model.eval().to(device)
    print(f"Feature dimension: {model.embed_dim}")

    transform = get_transform(args.image_size)

    labels_path = Path(args.labels_dir)
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",")]
    else:
        video_ids = sorted([p.stem for p in labels_path.glob("*.csv")])
    print(f"Processing {len(video_ids)} videos")

    for vid_id in video_ids:
        label_path = labels_path / f"{vid_id}.csv"
        if not label_path.exists():
            print(f"[SKIP] {vid_id}: label file not found")
            continue
        extract_video_features(
            video_id=vid_id,
            label_path=str(label_path),
            frames_root=args.frames_root,
            output_dir=args.output_dir,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )

    print("\nFeature extraction complete!")


if __name__ == "__main__":
    main()
