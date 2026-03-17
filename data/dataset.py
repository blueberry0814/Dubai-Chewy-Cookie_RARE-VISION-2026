"""
Galar sliding-window dataset using pre-extracted DINOv2 features.

Pipeline:
  1. Run extract_features.py to save per-video features as .npy files
  2. This Dataset loads .npy + CSV and returns windows of frames

Expected file structure:
  features_dir/{video_id}_features.npy  : [N, feat_dim] float32
  features_dir/{video_id}_frames.npy    : [N] int64  (actual frame numbers)
  labels_dir/{video_id}.csv             : columns include 'frame' and one column per label
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve"
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer"
]
ALL_LABELS = ANATOMY_LABELS + PATHOLOGY_LABELS  # 17 classes, fixed order


class GalarWindowDataset(Dataset):
    """
    Sliding-window Dataset.
    Each item = (features, labels, frame_nums) for a contiguous window of window_size frames.
    The last window is zero-padded if shorter than window_size.
    """

    def __init__(
        self,
        video_ids: list,
        features_dir: str,
        labels_dir: str,
        window_size: int = 512,
        stride: int = 256,
    ):
        self.features_dir = features_dir
        self.labels_dir   = labels_dir
        self.window_size  = window_size

        self.windows = []  # list of (video_id, start, end)
        # Cache features and labels in memory to minimize repeated I/O
        self.feat_cache  = {}
        self.frame_cache = {}
        self.label_cache = {}

        for vid_id in video_ids:
            feat_path  = os.path.join(features_dir, f"{vid_id}_features.npy")
            frame_path = os.path.join(features_dir, f"{vid_id}_frames.npy")
            label_path = os.path.join(labels_dir, f"{vid_id}.csv")

            if not all(os.path.exists(p) for p in [feat_path, frame_path, label_path]):
                print(f"[SKIP] {vid_id}: feature file or label not found")
                continue

            feats     = np.load(feat_path).astype(np.float32)   # [N, D]
            frame_arr = np.load(frame_path).astype(np.int64)    # [N]
            df        = pd.read_csv(label_path)
            df        = df.sort_values("frame").reset_index(drop=True)

            # Extract 17 label columns (missing columns filled with 0)
            label_arr = np.zeros((len(df), 17), dtype=np.float32)
            for i, col in enumerate(ALL_LABELS):
                if col in df.columns:
                    label_arr[:, i] = df[col].values.astype(np.float32)

            # Defensive alignment in case feature count and CSV row count differ
            n = min(len(feats), len(df))
            self.feat_cache[vid_id]  = feats[:n]
            self.frame_cache[vid_id] = frame_arr[:n]
            self.label_cache[vid_id] = label_arr[:n]

            # Generate windows
            for start in range(0, max(1, n - window_size + 1), stride):
                self.windows.append((vid_id, start, min(start + window_size, n)))
            # Add last window if stride doesn't divide evenly
            if n > window_size and (n - window_size) % stride != 0:
                self.windows.append((vid_id, n - window_size, n))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        vid_id, start, end = self.windows[idx]

        feats     = self.feat_cache[vid_id][start:end]   # [T', D]
        frame_arr = self.frame_cache[vid_id][start:end]  # [T']
        labels    = self.label_cache[vid_id][start:end]  # [T', 17]
        T_actual  = len(feats)

        # Zero-pad if window is shorter than window_size
        if T_actual < self.window_size:
            pad = self.window_size - T_actual
            feats     = np.vstack([feats,     np.zeros((pad, feats.shape[1]), dtype=np.float32)])
            frame_arr = np.concatenate([frame_arr, np.full(pad, -1, dtype=np.int64)])
            labels    = np.vstack([labels,    np.zeros((pad, 17),             dtype=np.float32)])

        total_rows = len(self.feat_cache[vid_id])
        video_pos_ratio = float(start) / max(total_rows - 1, 1)  # 0.0~1.0

        return {
            "features":        torch.from_numpy(feats),      # [T, D]
            "labels":          torch.from_numpy(labels),     # [T, 17]
            "frame_nums":      torch.from_numpy(frame_arr),  # [T]
            "valid_len":       T_actual,
            "video_id":        vid_id,
            "video_pos_ratio": torch.tensor(video_pos_ratio, dtype=torch.float32),
        }


def compute_pos_weights(dataset: GalarWindowDataset) -> torch.Tensor:
    """
    Compute inverse positive frequency per class across all training data.
    Used as pos_weight argument for BCE loss.
    """
    all_labels = np.vstack([
        dataset.label_cache[vid]
        for vid in dataset.label_cache
    ])  # [total_frames, 17]

    pos = all_labels.mean(axis=0).clip(1e-4, 1 - 1e-4)  # [17]
    weights = (1 - pos) / pos
    return torch.from_numpy(weights.astype(np.float32))


def make_weighted_sampler(dataset: GalarWindowDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that oversamples windows containing rare pathology frames.
    Window weights are proportional to the number of positive pathology frames.
    """
    pathology_idx = list(range(8, 17))  # pathology class indices in ALL_LABELS

    weights = []
    for vid_id, start, end in dataset.windows:
        window_labels = dataset.label_cache[vid_id][start:end]  # [T, 17]
        rare_count = window_labels[:, pathology_idx].sum()
        weights.append(1.0 + rare_count * 2.0)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)


def stratified_video_split(
    video_ids: list,
    labels_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> "tuple[list, list]":
    """
    Split videos into train/val such that rare pathology classes appear in both splits.

    Computes each video's set of pathology classes, then assigns rare-class videos
    to the validation set first to ensure coverage.
    """
    np.random.seed(seed)

    # Collect pathology classes present in each video
    video_pathology = {}
    for vid_id in video_ids:
        label_path = os.path.join(labels_dir, f"{vid_id}.csv")
        if not os.path.exists(label_path):
            video_pathology[vid_id] = set()
            continue
        df = pd.read_csv(label_path)
        present = set()
        for col in PATHOLOGY_LABELS:
            if col in df.columns and df[col].sum() > 0:
                present.add(col)
        video_pathology[vid_id] = present

    # Assign rare-class videos to val first to guarantee at least one video per rare class
    val_set, train_set = [], []
    covered_in_val = set()

    # Sort by number of pathology classes (fewest first = rarest first)
    sorted_vids = sorted(video_ids, key=lambda v: len(video_pathology[v]))
    np.random.shuffle(sorted_vids)

    n_val = max(1, int(len(video_ids) * val_ratio))

    for vid in sorted_vids:
        path_set = video_pathology[vid]
        if len(val_set) < n_val and not path_set.issubset(covered_in_val):
            val_set.append(vid)
            covered_in_val |= path_set
        elif len(val_set) < n_val:
            val_set.append(vid)
        else:
            train_set.append(vid)

    # Remaining videos go to train
    for vid in sorted_vids:
        if vid not in val_set and vid not in train_set:
            train_set.append(vid)

    return train_set, val_set
