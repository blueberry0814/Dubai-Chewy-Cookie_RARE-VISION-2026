"""
Step 2: Train the model.

Usage:
  python train.py --config configs/config.yaml

Training pipeline:
  1. Load pre-extracted DINOv2 features from features/
  2. Stratified video-level train/val split
  3. Oversample rare pathology windows via WeightedRandomSampler
  4. Multi-task loss: anatomy BCE + pathology ASL + prototype auxiliary losses
  5. Validate each epoch and save best model
"""

import os
import sys
import yaml
import argparse
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from data.dataset import GalarWindowDataset, compute_pos_weights, make_weighted_sampler, stratified_video_split
from models.model import GalarModel
from utils.losses import FocalLoss, ClassWeightedBCE, AsymmetricLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve"
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer"
]


def compute_frame_map(all_preds: np.ndarray, all_labels: np.ndarray, verbose: bool = False) -> dict:
    """
    Compute frame-level mAP as a proxy metric during training.
    Note: official evaluation uses temporal mAP, but frame-level mAP is useful for monitoring.

    Returns: {"anatomy_mAP": float, "pathology_mAP": float, "overall_mAP": float,
              "per_class_ap": list}
    """
    n_anatomy, n_pathology = 8, 9
    ap_list = []

    for cls_idx in range(n_anatomy + n_pathology):
        y_true = all_labels[:, cls_idx]
        y_pred = all_preds[:, cls_idx]
        if y_true.sum() == 0:
            ap_list.append(None)
            continue
        try:
            ap = average_precision_score(y_true, y_pred)
            ap_list.append(ap)
        except Exception:
            ap_list.append(None)

    valid_ap = [x for x in ap_list if x is not None]
    anatomy_ap = [x for x in ap_list[:n_anatomy] if x is not None]
    pathology_ap = [x for x in ap_list[n_anatomy:] if x is not None]

    if verbose:
        print("  [Anatomy per-class AP]")
        for i, name in enumerate(ANATOMY_LABELS):
            val = ap_list[i]
            print(f"    {name:20s}: {val:.4f}" if val is not None else f"    {name:20s}: N/A")
        print("  [Pathology per-class AP]")
        for i, name in enumerate(PATHOLOGY_LABELS):
            val = ap_list[n_anatomy + i]
            print(f"    {name:20s}: {val:.4f}" if val is not None else f"    {name:20s}: N/A")

    return {
        "anatomy_mAP":   float(np.mean(anatomy_ap))   if anatomy_ap   else 0.0,
        "pathology_mAP": float(np.mean(pathology_ap)) if pathology_ap else 0.0,
        "overall_mAP":   float(np.mean(valid_ap))      if valid_ap     else 0.0,
        "per_class_ap":  ap_list,
    }


def train_one_epoch(
    model: GalarModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    anatomy_loss_fn,
    pathology_loss_fn,
    proto_loss_fn,
    config: dict,
    device: torch.device,
    epoch: int,
    log_writer,
):
    model.train()
    total_loss = 0.0
    steps = 0
    log_interval = config["training"]["log_interval"]
    proto_momentum         = config["training"]["proto_momentum"]
    anatomy_proto_momentum = config["training"].get("anatomy_proto_momentum", 0.99)
    a_w       = config["training"]["anatomy_loss_weight"]
    p_w       = config["training"]["pathology_loss_weight"]
    proto_w   = config["training"]["proto_loss_weight"]
    cluster_w = config["training"].get("anatomy_cluster_loss_weight", 0.0)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        features        = batch["features"].to(device)         # [B, T, D]
        labels          = batch["labels"].to(device)           # [B, T, 17]
        valid_len       = batch["valid_len"]
        video_pos_ratio = batch["video_pos_ratio"].to(device)  # [B]

        anatomy_labels   = labels[:, :, :8]   # [B, T, 8]
        pathology_labels = labels[:, :, 8:]   # [B, T, 9]

        # ── Random Temporal Masking (train only) ─────────────────────────────
        mask_ratio = config["training"].get("mask_ratio", 0.0)
        if mask_ratio > 0:
            B_s, T_s, D_s = features.shape
            tmask = (torch.rand(B_s, T_s, device=device) < mask_ratio).unsqueeze(-1)
            noise = torch.randn(B_s, T_s, D_s, device=device) * 0.1
            features_input = torch.where(tmask.expand_as(features), noise, features)
        else:
            features_input = features

        # ── Feature Noise (train only) ────────────────────────────────────────
        # Small perturbation to improve generalization (std=0.01 is mild relative to DINOv2 norms)
        feat_noise_std = config["training"].get("feat_noise_std", 0.0)
        if feat_noise_std > 0:
            features_input = features_input + torch.randn_like(features_input) * feat_noise_std

        # ── Sequence Mixup (train only) ───────────────────────────────────────
        # Mixes two sequences with Beta(alpha, alpha) coefficient
        # Applies to both features and labels simultaneously (soft labels)
        mixup_alpha = config["training"].get("mixup_alpha", 0.0)
        mixup_prob  = config["training"].get("mixup_prob",  0.0)
        if mixup_alpha > 0 and random.random() < mixup_prob:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(features_input.size(0), device=device)
            features_input   = lam * features_input   + (1 - lam) * features_input[idx]
            anatomy_labels   = lam * anatomy_labels   + (1 - lam) * anatomy_labels[idx]
            pathology_labels = lam * pathology_labels + (1 - lam) * pathology_labels[idx]
            labels = torch.cat([anatomy_labels, pathology_labels], dim=-1)

        anatomy_logits, pathology_logits, x_a = model(
            features=features_input,
            raw_features=features,        # unmasked raw features for prototype gate and residual
            video_pos_ratio=video_pos_ratio,
            return_features=True,
        )

        # ── Loss computation ──────────────────────────────────────────────────
        # Mask out padded frames (frames beyond valid_len)
        B, T, _ = anatomy_logits.shape
        mask = torch.zeros(B, T, device=device)
        for i, vl in enumerate(valid_len):
            mask[i, :vl] = 1.0

        # Anatomy loss (BCE with class weights)
        # + transition boundary weighting: extra weight at class transition points (±2 frames)
        # Focuses on brief transitional classes like pylorus, z-line, ileocecal valve
        a_loss_raw = anatomy_loss_fn(anatomy_logits, anatomy_labels)   # [B, T, 8]

        anat_diff = (anatomy_labels[:, 1:] - anatomy_labels[:, :-1]).abs().any(dim=-1).float()  # [B, T-1]
        boundary_w = torch.zeros(B, T, device=device)
        boundary_w[:, 1:]  += anat_diff
        boundary_w[:, :-1] += anat_diff
        boundary_w = F.max_pool1d(
            boundary_w.unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)                                                             # [B, T]
        transition_boost = config["training"].get("transition_loss_boost", 3.0)
        boundary_w = 1.0 + transition_boost * boundary_w

        a_loss = (a_loss_raw * mask.unsqueeze(-1) * boundary_w.unsqueeze(-1)).sum() \
                 / (mask.sum() * 8 + 1e-8)

        # Pathology loss (ASL: Asymmetric Sigmoid Loss)
        p_loss = pathology_loss_fn(pathology_logits, pathology_labels)
        p_loss = (p_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * 9 + 1e-8)

        # Prototype auxiliary loss: maximize cosine similarity between active pathology frames
        # and their corresponding prototypes
        proto_loss = torch.tensor(0.0, device=device)
        if proto_w > 0:
            norm_feat  = torch.nn.functional.normalize(features, dim=-1)
            norm_proto = torch.nn.functional.normalize(model.pathology_prototypes, dim=-1)
            sim = norm_feat @ norm_proto.T  # [B, T, 9]
            active_mask = (pathology_labels > 0.5).float()
            if active_mask.sum() > 0:
                proto_loss = ((1.0 - sim) * active_mask).sum() / (active_mask.sum() + 1e-8)

        # ── Anatomy Temporal Smoothness Loss ─────────────────────────────────
        # Smooth loss is only applied to stable classes (stomach, colon, etc.)
        # Transitional classes (z-line, pylorus, ileocecal) are excluded (weight=0)
        # because they appear only briefly — smoothing would suppress them
        smooth_w = config["training"].get("smooth_loss_weight", 0.0)
        smooth_loss = torch.tensor(0.0, device=device)
        if smooth_w > 0:
            anatomy_probs = torch.sigmoid(anatomy_logits)          # [B, T, 8]
            pair_mask = mask[:, :-1] * mask[:, 1:]
            smooth_class_w = torch.tensor(
                [1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],        # z-line/pylorus/ileocecal = 0
                device=device
            )
            smooth_loss = (
                (anatomy_probs[:, 1:] - anatomy_probs[:, :-1]).abs()
                * pair_mask.unsqueeze(-1)
                * smooth_class_w.view(1, 1, 8)
            ).sum() / (pair_mask.sum() * smooth_class_w.sum() + 1e-8)

        # ── Anatomy Clustering Loss ───────────────────────────────────────────
        # Maximize cosine similarity between x_a and the corresponding anatomy prototype
        # Makes x_a space anatomy-discriminative, improving GCN adjacency quality
        anatomy_cluster_loss = torch.tensor(0.0, device=device)
        if cluster_w > 0 and model.anatomy_proto_initialized:
            norm_x_a_c   = F.normalize(x_a, dim=-1)
            norm_proto_c = F.normalize(model.anatomy_prototypes.detach(), dim=-1)
            sim_c        = norm_x_a_c @ norm_proto_c.T                         # [B, T, 8]
            # Apply only to healthy frames (pathology frames are naturally pushed away)
            no_path_mask = (pathology_labels.sum(dim=-1) == 0).float()
            has_label    = (anatomy_labels.sum(dim=-1) > 0).float() * mask * no_path_mask
            target_sim_c = (sim_c * anatomy_labels).sum(dim=-1)
            anatomy_cluster_loss = (
                (1.0 - target_sim_c) * has_label
            ).sum() / (has_label.sum() + 1e-8)

        loss = (a_w * a_loss + p_w * p_loss + proto_w * proto_loss
                + smooth_w * smooth_loss + cluster_w * anatomy_cluster_loss)

        # ── Backpropagation ───────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        optimizer.step()

        # Update EMA prototypes after gradient step
        model.update_prototypes(
            raw_features=features.detach(),
            pathology_labels=pathology_labels.detach(),
            momentum=proto_momentum,
        )
        model.update_anatomy_prototypes(
            x_a=x_a.detach(),
            anatomy_labels=anatomy_labels.detach(),
            pathology_labels=pathology_labels.detach(),
            momentum=anatomy_proto_momentum,
        )

        total_loss += loss.item()
        steps += 1

        if steps % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "a": f"{a_loss.item():.4f}", "p": f"{p_loss.item():.4f}", "cl": f"{anatomy_cluster_loss.item():.3f}", "gn": f"{grad_norm:.1f}"})
            log_writer.writerow([epoch, steps, f"{loss.item():.6f}", f"{a_loss.item():.6f}", f"{p_loss.item():.6f}", f"{proto_loss.item():.6f}", f"{smooth_loss.item():.6f}", f"{grad_norm:.4f}"])

    return total_loss / max(steps, 1)


@torch.no_grad()
def validate(
    model: GalarModel,
    loader: DataLoader,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  [val]", leave=False):
        features        = batch["features"].to(device)
        labels          = batch["labels"].cpu().numpy()
        valid_len       = batch["valid_len"]
        video_pos_ratio = batch["video_pos_ratio"].to(device)

        anatomy_logits, pathology_logits = model(
            features=features,
            raw_features=features,
            video_pos_ratio=video_pos_ratio,
        )

        preds = torch.cat([
            torch.sigmoid(anatomy_logits),
            torch.sigmoid(pathology_logits),
        ], dim=-1).cpu().numpy()  # [B, T, 17]

        for i, vl in enumerate(valid_len):
            all_preds.append(preds[i, :vl])
            all_labels.append(labels[i, :vl])

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    return compute_frame_map(all_preds, all_labels, verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Video list and split ──────────────────────────────────────────────────
    labels_dir   = config["data"]["labels_dir"]
    features_dir = config["data"]["features_dir"]

    all_video_ids = sorted([p.stem for p in Path(labels_dir).glob("*.csv")])
    val_ratio = config["training"]["val_ratio"]
    if val_ratio > 0:
        train_ids, val_ids = stratified_video_split(
            video_ids=all_video_ids,
            labels_dir=labels_dir,
            val_ratio=val_ratio,
            seed=config["training"]["seed"],
        )
    else:
        # val_ratio=0: use all data for training (for final submission after hyperparameter tuning)
        train_ids, val_ids = all_video_ids, []
    print(f"Train videos: {len(train_ids)}  Val videos: {len(val_ids)}")

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    train_ds = GalarWindowDataset(
        video_ids=train_ids,
        features_dir=features_dir,
        labels_dir=labels_dir,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
    )
    val_ds = GalarWindowDataset(
        video_ids=val_ids,
        features_dir=features_dir,
        labels_dir=labels_dir,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
    )
    print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")

    sampler = make_weighted_sampler(train_ds)  # oversample rare pathology windows
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    num_epochs = config["training"]["num_epochs"]

    # ── Model & loss functions ────────────────────────────────────────────────
    model = GalarModel(config).to(device)

    # Warm-start: load from a previous checkpoint (new layers remain randomly initialized)
    warmstart_ckpt = config["training"].get("warmstart_checkpoint", None)
    if warmstart_ckpt and os.path.exists(warmstart_ckpt):
        ws = torch.load(warmstart_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(ws["model_state_dict"], strict=False)
        model.anatomy_proto_initialized = ws.get("anatomy_proto_initialized", False)
        model.proto_initialized         = ws.get("proto_initialized", False)
        print(f"Warm-start from: {warmstart_ckpt}")
        print(f"  Missing (new): {missing}")
        print(f"  Unexpected (removed): {unexpected}")

    pos_weights = compute_pos_weights(train_ds).to(device)  # [17]

    # Extra boost for rare anatomy classes (z-line, pylorus, ileocecal valve, mouth)
    # Index: mouth=0, esophagus=1, stomach=2, SI=3, colon=4, z-line=5, pylorus=6, ileocecal=7
    anatomy_rare_boost = torch.ones(8, device=device)
    anatomy_rare_boost[0] = config["training"].get("mouth_boost",      2.0)
    anatomy_rare_boost[5] = config["training"].get("zline_boost",      4.0)
    anatomy_rare_boost[6] = config["training"].get("pylorus_boost",    4.0)
    anatomy_rare_boost[7] = config["training"].get("ileocecal_boost",  4.0)

    anatomy_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights[:8] * anatomy_rare_boost,
        reduction="none"
    )
    # ASL: more aggressively down-weights easy negatives than Focal Loss
    # gamma_neg=4 for strong negative focusing, gamma_pos=1 to preserve positive gradients
    pathology_loss_fn = AsymmetricLoss(
        gamma_pos=config["training"].get("asl_gamma_pos", 1.0),
        gamma_neg=config["training"].get("asl_gamma_neg", 4.0),
        clip=config["training"].get("asl_clip", 0.05),
    )
    proto_loss_fn = None  # computed inline in train_one_epoch

    # ── Per-branch learning rates ─────────────────────────────────────────────
    # Anatomy converges earlier (~epoch 70-80); pathology continues improving until epoch 200
    # anatomy branch: 0.5x LR for stable maintenance
    # pathology branch: 2.0x LR for continued learning
    # shared encoder: base LR
    base_lr = config["training"]["lr"]
    _ANATOMY_MODULES  = {"gc_sim1", "gc_sim2", "gc_dis1", "gc_dis2",
                         "anatomy_gcn_proj", "boundary_proj", "anatomy_head",
                         "anatomy_prototypes", "mamba_fwd", "mamba_bwd"}
    _PATHOLOGY_MODULES = {"raw_feat_proj", "gc_p", "pathology_head",
                          "pathology_prototypes", "proto_weight"}

    anatomy_params, pathology_params, shared_params = [], [], []
    for name, param in model.named_parameters():
        module = name.split(".")[0]
        if module in _ANATOMY_MODULES:
            anatomy_params.append(param)
        elif module in _PATHOLOGY_MODULES:
            pathology_params.append(param)
        else:
            shared_params.append(param)

    lr_anatomy_mul   = config["training"].get("lr_anatomy_mul",   0.5)
    lr_pathology_mul = config["training"].get("lr_pathology_mul", 2.0)
    optimizer = torch.optim.AdamW([
        {"params": shared_params,    "lr": base_lr},
        {"params": anatomy_params,   "lr": base_lr * lr_anatomy_mul},
        {"params": pathology_params, "lr": base_lr * lr_pathology_mul},
    ], weight_decay=config["training"]["weight_decay"])
    print(f"[LR] shared={base_lr:.1e}  anatomy={base_lr*0.5:.1e}  pathology={base_lr*2.0:.1e}")

    warmup_epochs = config["training"].get("warmup_epochs", 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs - warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    log_path = "./logs/train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "total_loss", "anatomy_loss", "pathology_loss", "proto_loss", "smooth_loss"])

    val_log_path = "./logs/val_log.csv"
    val_log_file = open(val_log_path, "w", newline="")
    val_log_writer = csv.writer(val_log_file)
    val_log_writer.writerow(["epoch", "anatomy_mAP", "pathology_mAP", "overall_mAP", "lr"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_map = 0.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer,
            anatomy_loss_fn, pathology_loss_fn, proto_loss_fn,
            config, device, epoch, log_writer
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if val_ids:
            # Normal mode: select best model based on validation mAP
            verbose_epoch = (epoch % 10 == 0)
            metrics = validate(model, val_loader, device, verbose=verbose_epoch)
            overall_map = metrics["overall_mAP"]
            print(
                f"Epoch {epoch:3d}/{num_epochs} | loss={avg_loss:.4f} | lr={current_lr:.2e} | "
                f"val anatomy={metrics['anatomy_mAP']:.4f} "
                f"pathology={metrics['pathology_mAP']:.4f} "
                f"overall={overall_map:.4f}"
            )
            val_log_writer.writerow([epoch, metrics["anatomy_mAP"], metrics["pathology_mAP"], overall_map, current_lr])
            val_log_file.flush()
            log_file.flush()

            if overall_map > best_map:
                best_map   = overall_map
                best_epoch = epoch
                ckpt_path  = os.path.join(config["training"]["save_dir"], "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": overall_map,
                    "config": config,
                    "anatomy_proto_initialized": model.anatomy_proto_initialized,
                    "proto_initialized": model.proto_initialized,
                }, ckpt_path)
                print(f"  -> Best model saved (epoch={epoch}, overall_mAP={overall_map:.4f})")
        else:
            # Full-data mode: save every epoch; last epoch = best_model.pth
            print(f"Epoch {epoch:3d}/{num_epochs} | loss={avg_loss:.4f} | lr={current_lr:.2e} | (full-data mode, no val)")
            log_file.flush()
            ckpt_path = os.path.join(config["training"]["save_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_map": 0.0,
                "config": config,
                "anatomy_proto_initialized": model.anatomy_proto_initialized,
                "proto_initialized": model.proto_initialized,
            }, ckpt_path)

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(config["training"]["save_dir"], f"epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)

    log_file.close()
    val_log_file.close()
    print(f"\nTraining complete! Best epoch={best_epoch}, best overall mAP={best_map:.4f}")


if __name__ == "__main__":
    main()
