"""
Loss function implementations.
- FocalLoss: for rare multi-label pathology classes — focuses on hard examples
- AsymmetricLoss (ASL): optimized for multi-label with extreme class imbalance
- ClassWeightedBCE: BCE with inverse-frequency class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al., 2017).
    L = -alpha * (1 - p)^gamma * log(p)

    When gamma > 0, reduces the loss contribution of easy examples (high p),
    forcing the model to focus on rare, hard classes.
    """

    def __init__(self, gamma: float = 3.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [*, n_class]  (raw logits, before sigmoid)
        targets : [*, n_class]  (0 or 1)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., 2021) — optimized for rare multi-label classes.

    Applies different focusing parameters to positives and negatives:
      - gamma_pos: positive focusing (typically 0~2, kept low to preserve gradients)
      - gamma_neg: negative focusing (typically 2~4, high to suppress easy negatives)
      - clip: probability floor for negatives (suppresses false negatives)

    More effective than Focal Loss for multi-label settings where negatives dominate:
      - Strongly down-weights easy negative samples
      - Mildly focuses positives (gamma_pos=1) to preserve gradient signal
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        # Apply clip to negative probabilities to suppress easy negatives
        p_neg = (p + self.clip).clamp(max=1.0)

        loss_pos = targets       * (1 - p)     ** self.gamma_pos * torch.log(p.clamp(min=1e-8))
        loss_neg = (1 - targets) * p_neg       ** self.gamma_neg * torch.log((1 - p_neg).clamp(min=1e-8))
        loss = -(loss_pos + loss_neg)
        return loss  # reduction applied externally in train.py after mask


class ClassWeightedBCE(nn.Module):
    """
    BCE with per-class weights based on inverse positive frequency.
    pos_weight is precomputed from training data statistics.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight  # [n_class]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
