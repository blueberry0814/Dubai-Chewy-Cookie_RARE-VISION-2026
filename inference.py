"""
Step 3: Run inference on test videos -> generate competition submission JSON

Usage:
  # 1) Extract features for test videos (using extract_features.py)
  python extract_features.py --labels_dir ./test_data/Labels \
                             --frames_root ./test_data \
                             --output_dir ./features/test

  # 2) Run inference and generate JSON
  python inference.py --config configs/config.yaml \
                      --test_features_dir ./features/test \
                      --test_labels_dir ./test_data/Labels \
                      --checkpoint ./checkpoints/best_model.pth \
                      --output_dir ./results

  # For test data without labels, use --test_video_ids instead of --test_labels_dir
  python inference.py --config configs/config.yaml \
                      --test_features_dir ./features/test \
                      --test_video_ids VID_001,VID_002,VID_003 \
                      --checkpoint ./checkpoints/best_model.pth

Output:
  results/predictions.json   competition submission JSON (temporal events format)
  results/frame_preds/       per-video frame prediction CSVs (for debugging)
"""

import os
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.signal import medfilt
from tqdm import tqdm

from models.model import GalarModel, ALL_LABELS
from utils.make_json import predictions_to_events, build_json_from_predictions
from utils.viterbi import viterbi_anatomy


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> GalarModel:
    model = GalarModel(config).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    # Restore prototype initialization flags (not stored in state_dict)
    model.anatomy_proto_initialized = ckpt.get("anatomy_proto_initialized", False)
    model.proto_initialized         = ckpt.get("proto_initialized", False)
    model.eval()
    print(f"Checkpoint loaded: {checkpoint_path}  (epoch={ckpt.get('epoch', '?')}, val_mAP={ckpt.get('val_map', 0):.4f})")
    print(f"  anatomy_proto_initialized={model.anatomy_proto_initialized}, proto_initialized={model.proto_initialized}")
    return model


@torch.no_grad()
def infer_video(
    model: GalarModel,
    features: np.ndarray,    # [N, D]
    frame_nums: np.ndarray,  # [N]
    window_size: int,
    stride: int,
    device: torch.device,
    threshold: float = 0.5,
    smooth_window: int = 5,
) -> np.ndarray:
    """
    Sliding window inference for a single video.

    Returns:
        pred_binary: [N, 17] binary predictions
    """
    N, D = features.shape
    # Accumulate predictions and counts across overlapping windows (averaged)
    pred_sum   = np.zeros((N, 17), dtype=np.float64)
    pred_count = np.zeros(N, dtype=np.int32)

    windows = []
    for start in range(0, max(1, N - window_size + 1), stride):
        windows.append((start, min(start + window_size, N)))
    if N > window_size and (N - window_size) % stride != 0:
        windows.append((N - window_size, N))
    if not windows:
        windows = [(0, N)]

    for start, end in tqdm(windows, desc="  sliding window", leave=False):
        chunk = features[start:end].astype(np.float32)
        T_actual = len(chunk)

        # Zero-pad if shorter than window_size
        if T_actual < window_size:
            pad = np.zeros((window_size - T_actual, D), dtype=np.float32)
            chunk = np.vstack([chunk, pad])

        feat_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)  # [1, T, D]
        vp_ratio    = torch.tensor([start / max(N - 1, 1)], dtype=torch.float32).to(device)
        anatomy_logits, pathology_logits = model(
            features=feat_tensor,
            raw_features=feat_tensor,
            video_pos_ratio=vp_ratio,
        )
        preds = torch.cat([
            torch.sigmoid(anatomy_logits),
            torch.sigmoid(pathology_logits),
        ], dim=-1).squeeze(0).cpu().numpy()  # [T, 17]

        # Accumulate only the valid (non-padded) frame range
        pred_sum[start:start + T_actual]   += preds[:T_actual]
        pred_count[start:start + T_actual] += 1

    # Average over overlapping windows
    count_safe = np.maximum(pred_count, 1)[:, np.newaxis]
    pred_probs = pred_sum / count_safe  # [N, 17]

    # Median filter smoothing (removes noisy predictions)
    if smooth_window > 1:
        for cls_idx in range(17):
            pred_probs[:, cls_idx] = medfilt(pred_probs[:, cls_idx], kernel_size=smooth_window)

    # Viterbi anatomy smoothing: enforces biological GI traversal order
    pred_probs[:, :8] = viterbi_anatomy(pred_probs[:, :8])

    pred_binary = (pred_probs >= threshold).astype(np.int32)
    return pred_binary, pred_probs


def save_frame_preds_csv(
    video_id: str,
    frame_nums: np.ndarray,
    pred_probs: np.ndarray,
    pred_binary: np.ndarray,
    output_dir: str,
):
    """Save per-frame prediction probabilities and binary results as CSV (for debugging)."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for i, fn in enumerate(frame_nums):
        row = {"video_id": video_id, "frame": int(fn)}
        for j, lbl in enumerate(ALL_LABELS):
            row[f"prob_{lbl}"] = float(pred_probs[i, j])
            row[f"pred_{lbl}"] = int(pred_binary[i, j])
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"{video_id}_predictions.csv")
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint",        type=str, default=None,
                        help="Checkpoint path (if None, uses default from config)")
    parser.add_argument("--test_features_dir", type=str, default=None,
                        help="Directory of extracted test features (.npy files)")
    parser.add_argument("--test_labels_dir",   type=str, default=None,
                        help="Test label CSV directory (if absent, use --test_video_ids)")
    parser.add_argument("--test_video_ids",    type=str, default=None,
                        help="Comma-separated list of test video IDs (used when no labels available)")
    parser.add_argument("--output_dir",        type=str, default=None)
    parser.add_argument("--threshold",         type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Argument priority: CLI > config
    checkpoint    = args.checkpoint    or config["inference"]["checkpoint"]
    output_dir    = args.output_dir    or config["inference"]["output_dir"]
    threshold     = args.threshold     or config["inference"]["threshold"]
    smooth_window = config["inference"]["smooth_window"]
    window_size   = config["data"]["window_size"]
    infer_stride  = config["data"]["infer_stride"]
    features_dir  = args.test_features_dir or config["data"]["features_dir"]

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(config, checkpoint, device)

    # Determine list of test videos
    if args.test_video_ids:
        video_ids = [v.strip() for v in args.test_video_ids.split(",")]
    elif args.test_labels_dir:
        video_ids = sorted([p.stem for p in Path(args.test_labels_dir).glob("*.csv")])
    else:
        # Fall back to all *_features.npy files in features_dir
        video_ids = sorted([
            p.name.replace("_features.npy", "")
            for p in Path(features_dir).glob("*_features.npy")
        ])
    print(f"Videos to infer: {video_ids}")

    # ── Per-video inference ───────────────────────────────────────────────────
    video_predictions = []

    for vid_id in video_ids:
        feat_path  = os.path.join(features_dir, f"{vid_id}_features.npy")
        frame_path = os.path.join(features_dir, f"{vid_id}_frames.npy")

        if not os.path.exists(feat_path):
            print(f"[SKIP] {vid_id}: feature file not found ({feat_path})")
            continue

        features   = np.load(feat_path).astype(np.float32)  # [N, D]
        frame_nums = np.load(frame_path).astype(np.int64)   # [N]
        print(f"\n[{vid_id}] frames={len(features)}  shape={features.shape}")

        pred_binary, pred_probs = infer_video(
            model=model,
            features=features,
            frame_nums=frame_nums,
            window_size=window_size,
            stride=infer_stride,
            device=device,
            threshold=threshold,
            smooth_window=smooth_window,
        )

        # Save per-frame debug CSV
        save_frame_preds_csv(
            video_id=vid_id,
            frame_nums=frame_nums,
            pred_probs=pred_probs,
            pred_binary=pred_binary,
            output_dir=os.path.join(output_dir, "frame_preds"),
        )

        video_predictions.append({
            "video_id":   vid_id,
            "frame_nums": frame_nums,
            "pred_binary": pred_binary,
        })

        # Print active class summary
        print(f"  Active classes: {[ALL_LABELS[i] for i in range(17) if pred_binary[:, i].any()]}")

    # ── Generate JSON ─────────────────────────────────────────────────────────
    output_json = os.path.join(output_dir, "predictions.json")
    build_json_from_predictions(video_predictions, output_json)

    print(f"\nInference complete! Output: {output_json}")
    print("Next step: upload predictions.json to the Sanity Checker website")


if __name__ == "__main__":
    main()
