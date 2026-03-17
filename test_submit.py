"""
ICPR 2026 RARE-VISION Challenge -- Test Submission Script

End-to-end pipeline in a single script:
  1. Read frame list from submission CSV (frame_file column)
  2. Extract DINOv2 features from each frame image
  3. Run sliding window inference
  4. Fill binary predictions into the submission CSV in the required column order

Usage:
  python test_submit.py \
      --frames_root  ./Testdata_ICPR_2026_RARE_Challenge \
      --submit_dir   ./ukdd_navi_00051_00068_00076 \
      --checkpoint   ./checkpoints/best_model.pth \
      --config       ./configs/config.yaml \
      --output_dir   ./results/submission

Output:
  results/submission/ukdd_navi_XXXXX.csv  -- submission CSV (semicolon-delimited)
  results/submission/probs/               -- probability CSVs (for debugging)
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from scipy.signal import medfilt
from tqdm import tqdm

from models.model import GalarModel, ALL_LABELS
from utils.viterbi import viterbi_anatomy


# ── Submission CSV column order (22 columns) ─────────────────────────────────
SUBMIT_COLUMNS = [
    "frame_file",
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus",
    "ampulla of vater",         # not in model -> always 0
    "ileocecal valve",
    "IBD",                      # not in model -> always 0
    "active bleeding", "angiectasia", "blood",
    "cancer",                   # not in model -> always 0
    "erosion", "erythema",
    "foreign body",             # not in model -> always 0
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]

# Model output indices (based on ALL_LABELS):
# anatomy: mouth(0) esophagus(1) stomach(2) small intestine(3) colon(4)
#          z-line(5) pylorus(6) ileocecal valve(7)
# pathology: active bleeding(8) angiectasia(9) blood(10) erosion(11) erythema(12)
#            hematin(13) lymphangioectasis(14) polyp(15) ulcer(16)
#
# Submission column -> model index mapping (None = not in model, always 0)
COL_TO_MODEL_IDX = {
    "mouth": 0, "esophagus": 1, "stomach": 2, "small intestine": 3,
    "colon": 4, "z-line": 5, "pylorus": 6, "ileocecal valve": 7,
    "ampulla of vater": None,
    "IBD": None,
    "active bleeding": 8, "angiectasia": 9, "blood": 10,
    "cancer": None,
    "erosion": 11, "erythema": 12,
    "foreign body": None,
    "hematin": 13, "lymphangioectasis": 14, "polyp": 15, "ulcer": 16,
}


# ── DINOv2 preprocessing ──────────────────────────────────────────────────────
def get_transform(image_size: int = 336) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Feature extraction ────────────────────────────────────────────────────────
@torch.no_grad()
def _extract_batch(dinov2: nn.Module, imgs: list, device: torch.device) -> np.ndarray:
    tensor = torch.stack(imgs).to(device)   # [B, C, H, W]
    feats  = dinov2(tensor)                  # [B, D]
    return feats.cpu().numpy()


def extract_features_from_framelist(
    frame_files: list,          # e.g. ['frame_0000049.png', ...]
    video_dir: str,             # directory containing frame images
    dinov2: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> tuple:
    """
    Returns:
        features  : np.ndarray [N, D]
        valid_files: list[str]  filenames that were successfully loaded
    """
    all_features = []
    valid_files  = []
    batch_imgs   = []
    batch_names  = []

    for fname in tqdm(frame_files, desc="  extracting features", leave=False):
        img_path = os.path.join(video_dir, fname)
        if not os.path.exists(img_path):
            # Try alternative extensions
            for ext in [".PNG", ".jpg", ".JPG"]:
                alt = img_path.rsplit(".", 1)[0] + ext
                if os.path.exists(alt):
                    img_path = alt
                    break
            else:
                print(f"  [WARN] image not found: {img_path}")
                continue

        try:
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(transform(img))
            batch_names.append(fname)
        except Exception as e:
            print(f"  [WARN] failed to load {img_path}: {e}")
            continue

        if len(batch_imgs) == batch_size:
            feats = _extract_batch(dinov2, batch_imgs, device)
            all_features.append(feats)
            valid_files.extend(batch_names)
            batch_imgs.clear()
            batch_names.clear()

    if batch_imgs:
        feats = _extract_batch(dinov2, batch_imgs, device)
        all_features.append(feats)
        valid_files.extend(batch_names)

    if not all_features:
        return np.zeros((0, 384), dtype=np.float32), []

    return np.vstack(all_features).astype(np.float32), valid_files


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_video(
    model: GalarModel,
    features: np.ndarray,   # [N, D]
    window_size: int,
    stride: int,
    device: torch.device,
    threshold: float = 0.3,
    smooth_window: int = 5,
) -> tuple:
    """
    Returns:
        pred_binary : [N, 17] int32
        pred_probs  : [N, 17] float32
    """
    N, D = features.shape
    pred_sum   = np.zeros((N, 17), dtype=np.float64)
    pred_count = np.zeros(N, dtype=np.int32)

    windows = []
    for start in range(0, max(1, N - window_size + 1), stride):
        windows.append((start, min(start + window_size, N)))
    if N > window_size and (N - window_size) % stride != 0:
        windows.append((N - window_size, N))
    if not windows:
        windows = [(0, N)]

    for start, end in tqdm(windows, desc="  inference", leave=False):
        chunk    = features[start:end].astype(np.float32)
        T_actual = len(chunk)

        if T_actual < window_size:
            pad   = np.zeros((window_size - T_actual, D), dtype=np.float32)
            chunk = np.vstack([chunk, pad])

        feat_t = torch.from_numpy(chunk).unsqueeze(0).to(device)
        vp_r   = torch.tensor([start / max(N - 1, 1)], dtype=torch.float32).to(device)

        a_logits, p_logits = model(
            features=feat_t,
            raw_features=feat_t,
            video_pos_ratio=vp_r,
        )
        preds = torch.cat([
            torch.sigmoid(a_logits),
            torch.sigmoid(p_logits),
        ], dim=-1).squeeze(0).cpu().numpy()  # [T, 17]

        pred_sum[start:start + T_actual]   += preds[:T_actual]
        pred_count[start:start + T_actual] += 1

    count_safe = np.maximum(pred_count, 1)[:, np.newaxis]
    pred_probs = (pred_sum / count_safe).astype(np.float32)

    if smooth_window > 1:
        for c in range(17):
            pred_probs[:, c] = medfilt(pred_probs[:, c], kernel_size=smooth_window)

    # Viterbi anatomy smoothing
    pred_probs[:, :8] = viterbi_anatomy(pred_probs[:, :8])

    pred_binary = (pred_probs >= threshold).astype(np.int32)
    return pred_binary, pred_probs


# ── Official JSON format conversion ──────────────────────────────────────────
def _to_events_official(
    frame_nums: np.ndarray,   # [N] int64 -- actual frame numbers
    pred_binary: np.ndarray,  # [N, 17] int32
    video_id: str,
) -> dict:
    """
    Official sample.json format:
      "label" is a list (not a single string)
      A new event is created whenever the set of active labels changes
      Intervals with no active labels (all predictions = 0) produce no event
    """
    events = []
    N = len(frame_nums)
    if N == 0:
        return {"video_id": video_id, "events": []}

    def active_set(row_idx):
        return tuple(sorted([
            ALL_LABELS[c] for c in range(17) if pred_binary[row_idx, c] == 1
        ]))

    current = active_set(0)
    start_f = int(frame_nums[0])

    for i in range(1, N):
        nxt = active_set(i)
        if nxt != current:
            if current:  # skip empty label sets
                events.append({
                    "start": start_f,
                    "end":   int(frame_nums[i - 1]),
                    "label": list(current),
                })
            current = nxt
            start_f = int(frame_nums[i])

    # Handle last segment
    if current:
        events.append({
            "start": start_f,
            "end":   int(frame_nums[-1]),
            "label": list(current),
        })

    return {"video_id": video_id, "events": events}


# ── Save submission CSV ───────────────────────────────────────────────────────
def fill_submission_csv(
    submit_csv_path: str,
    frame_files: list,          # inferred frame filenames (order preserved)
    pred_binary: np.ndarray,    # [N, 17]
    output_path: str,
):
    """
    Fill in predictions while preserving the original submission CSV frame_file order.
    Frames that were not inferred (missing images) are filled with 0.
    """
    df_orig = pd.read_csv(submit_csv_path, sep=";", dtype=str)

    # Map frame_file -> prediction row index
    fname_to_idx = {fname: i for i, fname in enumerate(frame_files)}

    label_cols = [c for c in SUBMIT_COLUMNS if c != "frame_file"]

    for col in label_cols:
        model_idx = COL_TO_MODEL_IDX.get(col, None)
        values = []
        for _, row in df_orig.iterrows():
            fname = row["frame_file"]
            if model_idx is None or fname not in fname_to_idx:
                values.append(0)
            else:
                row_idx = fname_to_idx[fname]
                values.append(int(pred_binary[row_idx, model_idx]))
        df_orig[col] = values

    # Save with semicolon delimiter
    df_orig.to_csv(output_path, sep=";", index=False)
    print(f"  Saved: {output_path}")


# ── Save probability CSV (debug) ──────────────────────────────────────────────
def save_prob_csv(
    video_id: str,
    frame_files: list,
    pred_probs: np.ndarray,
    output_dir: str,
):
    from models.model import ALL_LABELS
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for i, fname in enumerate(frame_files):
        row = {"frame_file": fname}
        for j, lbl in enumerate(ALL_LABELS):
            row[f"prob_{lbl}"] = float(pred_probs[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"{video_id}_probs.csv"), index=False
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ICPR 2026 RARE test submission script")
    parser.add_argument("--frames_root", type=str,
                        default="./Testdata_ICPR_2026_RARE_Challenge",
                        help="Root directory for frame images ({video_id}/frame_XXXXXXX.png)")
    parser.add_argument("--submit_dir",  type=str,
                        default="./ukdd_navi_00051_00068_00076",
                        help="Directory containing submission CSVs (semicolon-delimited, with frame_file column)")
    parser.add_argument("--checkpoint",  type=str,
                        default="./checkpoints/best_model.pth")
    parser.add_argument("--config",      type=str,
                        default="./configs/config.yaml")
    parser.add_argument("--output_dir",  type=str,
                        default="./results/submission")
    parser.add_argument("--dinov2_model", type=str, default=None,
                        help="dinov2_vits14 / dinov2_vitb14 (default: read from config)")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Binarization threshold (default: config inference.threshold)")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Batch size for DINOv2 feature extraction")
    parser.add_argument("--video_ids",   type=str,   default=None,
                        help="Comma-separated video IDs to process (default: all CSVs in submit_dir)")
    parser.add_argument("--feat_cache_dir", type=str, default=None,
                        help="Directory for cached feature .npy files (skips re-extraction if present)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    threshold     = args.threshold    or config["inference"]["threshold"]
    smooth_window = config["inference"]["smooth_window"]
    window_size   = config["data"]["window_size"]
    infer_stride  = config["data"]["infer_stride"]
    image_size    = config["data"]["image_size"]
    dinov2_name   = args.dinov2_model or config["model"]["dinov2_model"]

    os.makedirs(args.output_dir, exist_ok=True)
    prob_dir = os.path.join(args.output_dir, "probs")
    os.makedirs(prob_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}  Window: {window_size}  Stride: {infer_stride}")

    # ── Load DINOv2 ───────────────────────────────────────────────────────────
    print(f"\nLoading DINOv2: {dinov2_name} ...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", dinov2_name)
    dinov2.eval().to(device)
    transform = get_transform(image_size)

    # ── Load classification model ─────────────────────────────────────────────
    print(f"Loading model: {args.checkpoint} ...")
    galar_model = GalarModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    galar_model.load_state_dict(ckpt["model_state_dict"])
    galar_model.anatomy_proto_initialized = ckpt.get("anatomy_proto_initialized", False)
    galar_model.proto_initialized         = ckpt.get("proto_initialized", False)
    galar_model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  val_mAP={ckpt.get('val_map', 0):.4f}")

    # ── Determine video list ──────────────────────────────────────────────────
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",")]
    else:
        video_ids = sorted([p.stem for p in Path(args.submit_dir).glob("*.csv")])
    print(f"\nVideos to process: {video_ids}\n")

    # ── Per-video processing ──────────────────────────────────────────────────
    all_video_preds = []

    for vid_id in video_ids:
        print(f"{'='*60}")
        print(f"[{vid_id}]")

        submit_csv = os.path.join(args.submit_dir, f"{vid_id}.csv")
        if not os.path.exists(submit_csv):
            print(f"  [SKIP] submission CSV not found: {submit_csv}")
            continue

        # Read frame file list
        df_sub = pd.read_csv(submit_csv, sep=";", dtype=str)
        frame_files = df_sub["frame_file"].tolist()
        print(f"  Total frames: {len(frame_files)}")

        # Extract features (use cache if available)
        if args.feat_cache_dir:
            feat_path  = os.path.join(args.feat_cache_dir, f"{vid_id}_features.npy")
            fname_path = os.path.join(args.feat_cache_dir, f"{vid_id}_framenames.txt")
            if os.path.exists(feat_path) and os.path.exists(fname_path):
                print(f"  Loading cached features: {feat_path}")
                features    = np.load(feat_path).astype(np.float32)
                valid_files = open(fname_path).read().splitlines()
            else:
                video_dir   = os.path.join(args.frames_root, vid_id)
                features, valid_files = extract_features_from_framelist(
                    frame_files, video_dir, dinov2, transform, device, args.batch_size
                )
                os.makedirs(args.feat_cache_dir, exist_ok=True)
                np.save(feat_path, features)
                with open(fname_path, "w") as fh:
                    fh.write("\n".join(valid_files))
                print(f"  Cached features saved: {feat_path}")
        else:
            video_dir   = os.path.join(args.frames_root, vid_id)
            features, valid_files = extract_features_from_framelist(
                frame_files, video_dir, dinov2, transform, device, args.batch_size
            )

        if len(features) == 0:
            print(f"  [SKIP] no valid frames found")
            continue

        print(f"  Feature shape: {features.shape}  Valid frames: {len(valid_files)}")

        # Run inference
        pred_binary, pred_probs = infer_video(
            model=galar_model,
            features=features,
            window_size=window_size,
            stride=infer_stride,
            device=device,
            threshold=threshold,
            smooth_window=smooth_window,
        )

        # Print active class summary
        from models.model import ALL_LABELS
        active = [ALL_LABELS[i] for i in range(17) if pred_binary[:, i].any()]
        print(f"  Active classes: {active}")

        # Save probability CSV (debug)
        save_prob_csv(vid_id, valid_files, pred_probs, prob_dir)

        # Fill and save submission CSV
        out_csv = os.path.join(args.output_dir, f"{vid_id}.csv")
        fill_submission_csv(submit_csv, valid_files, pred_binary, out_csv)

        # Extract integer frame numbers for JSON (frame_0000049.png -> 49)
        frame_nums_int = np.array([
            int(f.replace("frame_", "").replace(".png", "").replace(".PNG", ""))
            for f in valid_files
        ], dtype=np.int64)
        all_video_preds.append({
            "video_id":   vid_id,
            "frame_nums": frame_nums_int,
            "pred_binary": pred_binary,
        })

    # ── Generate full-video JSON (for evaluator upload) ───────────────────────
    if all_video_preds:
        import json
        json_path = os.path.join(args.output_dir, "predictions.json")
        result = {"videos": []}
        for vp in all_video_preds:
            vid_events = _to_events_official(
                frame_nums=vp["frame_nums"],
                pred_binary=vp["pred_binary"],
                video_id=vp["video_id"],
            )
            result["videos"].append(vid_events)
        with open(json_path, "w") as fj:
            json.dump(result, fj, indent=2)
        print(f"JSON saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"Done! Submission files: {args.output_dir}")
    print("* Upload predictions.json to the evaluator website.")


if __name__ == "__main__":
    main()
