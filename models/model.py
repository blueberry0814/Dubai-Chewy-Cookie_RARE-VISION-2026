"""
GALAR-TemporalNet: Multi-label temporal VCE classification model

Architecture:
  DINOv2 features (pre-extracted) -> Linear Projection + Positional Embed
  -> Windowed Self-Attention (Local Temporal)  [VadCLIP Local Module]
  -> GCN (Similarity Adj + Distance Adj)       [VadCLIP Global Module]
  -> [Anatomy Head: 8 sigmoid]
     [Pathology Head: 9 sigmoid x (1 + proto_weight x Prototype Gate)]

Prototype Gate:
  Computes per-class cosine similarity between raw DINOv2 features and
  class prototypes, converted via sigmoid -> used as a multiplicative gate
  for pathology predictions (auxiliary branch).
  proto_weight=0 has no effect; automatically tuned during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import GraphConvolution, DistanceAdj
from mamba_ssm import Mamba


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve"
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer"
]
ALL_LABELS = ANATOMY_LABELS + PATHOLOGY_LABELS  # 17 classes, fixed order


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class WindowedAttentionBlock(nn.Module):
    """
    Windowed Self-Attention + FFN.
    Attention between frames outside the window is blocked with -inf masking.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)
        # Register window mask as buffer (automatically moved to device)
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        mask = self.attn_mask if self.attn_mask is not None else None
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class GalarModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        feat_dim   = config["model"]["feat_dim"]
        hidden_dim = config["model"]["hidden_dim"]
        n_heads    = config["model"]["n_heads"]
        n_layers   = config["model"]["n_layers"]
        window     = config["model"]["attn_window"]
        max_len    = config["model"]["max_seq_len"]

        n_anatomy   = len(ANATOMY_LABELS)   # 8
        n_pathology = len(PATHOLOGY_LABELS) # 9
        self.sim_threshold = config["model"].get("sim_threshold", 0.5)

        # ── 1. Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_embed  = nn.Embedding(max_len, hidden_dim)

        # Motion feature: feat[t] - feat[t-1] (velocity in DINOv2 space)
        # Spikes strongly at transition frames (z-line, pylorus, ileocecal) -> boundary detector
        # Nearly zero in stable regions (e.g., inside stomach) -> no noise
        self.motion_proj = nn.Linear(feat_dim, hidden_dim)

        # ── 2. Local Temporal: Windowed Self-Attention ───────────────────────
        dropout = config["model"].get("dropout", 0.1)
        attn_mask = self._build_window_mask(max_len, window)
        self.local_blocks = nn.ModuleList([
            WindowedAttentionBlock(hidden_dim, n_heads, attn_mask, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── 3. Global Temporal: GCN ──────────────────────────────────────────
        gcn_dim = hidden_dim // 2
        self.dist_adj = DistanceAdj()
        self.gelu = QuickGELU()

        # Anatomy branch: DUAL GCN (both similarity and distance adjacency)
        self.gc_sim1 = GraphConvolution(hidden_dim, gcn_dim, residual=True)
        self.gc_sim2 = GraphConvolution(gcn_dim,    gcn_dim, residual=True)
        self.gc_dis1 = GraphConvolution(hidden_dim, gcn_dim, residual=True)
        self.gc_dis2 = GraphConvolution(gcn_dim,    gcn_dim, residual=True)
        self.anatomy_gcn_proj = nn.Linear(hidden_dim, hidden_dim)  # concat(256+256)->512

        # Pathology branch: x_a.detach() + raw residual -> sim_GCN (1 layer)
        # detach(): prevents smooth_loss gradients from flowing into pathology branch
        # gc_p: single GCN based on visual similarity -> re-clustering within short temporal range
        self.raw_feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.gc_p = GraphConvolution(hidden_dim, hidden_dim, residual=True)

        # Video-level GPS: encodes where in the full video this window is (added after attention)
        self.video_pos_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            QuickGELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        dropout = config["model"].get("dropout", 0.1)

        # ── 4. Classification Heads ──────────────────────────────────────────

        # Bi-Mamba: forward/backward SSMs encode "what was seen so far and what comes next"
        # More selective memory than prefix_mean/suffix_mean running averages
        # d_state=16: SSM state size (memory capacity), d_conv=4: local convolution, expand=1: saves parameters
        self.mamba_fwd = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=1)
        self.mamba_bwd = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=1)
        self.boundary_proj = nn.Linear(hidden_dim * 3, hidden_dim)  # [x_a, h_fwd, h_bwd] -> hidden

        self.anatomy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            QuickGELU(),
            nn.Linear(hidden_dim // 2, n_anatomy),
        )
        self.pathology_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            QuickGELU(),
            nn.Linear(hidden_dim // 2, n_pathology),
        )

        # ── 5. Anatomy Prototype (hidden space, EMA) ─────────────────────────
        # Maintains per-class centroids in x_a space
        # Pathology input = x_a - anatomy_prototype_assigned (residual: deviation from normal)
        self.register_buffer(
            "anatomy_prototypes",
            torch.zeros(n_anatomy, hidden_dim)
        )
        self.anatomy_proto_initialized = False

        # ── 6. Pathology Prototype Gate (raw DINOv2 space, EMA) ─────────────
        self.register_buffer(
            "pathology_prototypes",
            torch.zeros(n_pathology, feat_dim)
        )
        # proto_weight: starts at 0 -> no effect early in training, gradually activated
        self.proto_weight = nn.Parameter(torch.tensor(0.0))
        self.proto_initialized = False

    # ── Utility methods ───────────────────────────────────────────────────────

    def _build_window_mask(self, max_len: int, window: int) -> torch.Tensor:
        """Build additive mask: -inf outside window, 0 inside."""
        mask = torch.full((max_len, max_len), float("-inf"))
        for start in range(0, max_len, window):
            end = min(start + window, max_len)
            mask[start:end, start:end] = 0.0
        return mask

    def _similarity_adj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity-based adjacency matrix.
        Values below threshold are zeroed; self-loops added; row-sum normalized.
        (Using normalize instead of softmax to avoid uniform-distribution issue on zero rows.)
        x: [B, T, D]
        returns: [B, T, T]
        """
        x_norm = F.normalize(x, dim=-1)
        sim = x_norm @ x_norm.transpose(-1, -2)  # [B, T, T]
        sim = F.threshold(sim, self.sim_threshold, 0.0)
        # Self-loop: isolated frames (no connections after thresholding) retain their own signal
        T = sim.shape[1]
        eye = torch.eye(T, device=sim.device).unsqueeze(0)  # [1, T, T]
        sim = sim + eye
        # Row-sum normalize (D^{-1} A)
        row_sum = sim.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sim = sim / row_sum
        return sim

    # ── Core Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,
        raw_features: torch.Tensor = None,
        valid_len: torch.Tensor = None,
        video_pos_ratio: torch.Tensor = None,
        return_features: bool = False,
        return_all: bool = False,
    ):
        """
        Args:
            features        : [B, T, feat_dim]
            raw_features    : [B, T, feat_dim]  original features without masking
            valid_len       : [B] actual number of frames (currently unused)
            video_pos_ratio : [B] position of this window in the full video (0.0~1.0)
            return_features : returns (anatomy_logits, pathology_logits, x_a)
            return_all      : returns (anatomy_logits, pathology_logits, x_a, x_residual)

        Returns:
            anatomy_logits  : [B, T, 8]
            pathology_logits: [B, T, 9]
        """
        B, T, _ = features.shape
        device = features.device

        # 1. Project + local positional embedding
        x = self.input_proj(features)                                    # [B, T, hidden]
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) # [B, T]
        x = x + self.pos_embed(pos)

        # Motion: feat[t] - feat[t-1] (zero-padded at t=0)
        # Spikes at transition frames -> critical for detecting short transition classes
        # Padding boundary bug prevention: motion at padding frames (all-zero) is masked to 0
        # (padding boundary: motion[valid_len] = 0 - real_feat -> large spurious spike)
        motion = torch.zeros_like(features)
        motion[:, 1:] = features[:, 1:] - features[:, :-1]
        is_padding = (features.abs().sum(dim=-1) == 0)          # [B, T]
        motion = motion * (~is_padding).float().unsqueeze(-1)    # zero out motion at padding positions
        x = x + self.motion_proj(motion)

        # 2. Windowed Self-Attention (Local, shared encoder)
        for block in self.local_blocks:
            x = block(x)

        # 2.5. Video GPS -- added after attention (before adjacency computation to avoid contamination)
        if video_pos_ratio is not None:
            vp = video_pos_ratio.float().view(B, 1, 1)   # [B, 1, 1]
            vp_embed = self.video_pos_proj(vp)            # [B, 1, hidden]
            x = x + vp_embed                             # broadcast along T dimension

        # 3. Anatomy Branch: DUAL GCN (visual similarity + temporal distance)
        sim_adj = self._similarity_adj(x)                               # [B, T, T]
        dis_adj = self.dist_adj(B, T, device)                           # [B, T, T]

        x_sim = self.gelu(self.gc_sim1(x, sim_adj))
        x_sim = self.gelu(self.gc_sim2(x_sim, sim_adj))
        x_dis = self.gelu(self.gc_dis1(x, dis_adj))
        x_dis = self.gelu(self.gc_dis2(x_dis, dis_adj))

        x_a = self.anatomy_gcn_proj(torch.cat([x_sim, x_dis], dim=-1)) # [B, T, hidden]

        # 4. Bi-Mamba Boundary Context
        # h_fwd[t]: state selectively memorizing frames 0..t (forward SSM)
        # h_bwd[t]: state selectively memorizing frames t..T (backward SSM, flip trick)
        # Unlike running averages, Mamba can forget irrelevant info and focus on boundary signals
        # -> especially beneficial for short transition classes like pylorus and ileocecal valve
        h_fwd = self.mamba_fwd(x_a)                      # [B, T, H] forward hidden states
        h_bwd = self.mamba_bwd(x_a.flip(1)).flip(1)      # [B, T, H] backward hidden states (flip-process-flip)
        x_a_ctx = self.boundary_proj(torch.cat([x_a, h_fwd, h_bwd], dim=-1))  # [B, T, H]

        # 4. Anatomy Head (boundary-aware input)
        anatomy_logits = self.anatomy_head(x_a_ctx)                      # [B, T, 8]

        # 5. Anatomy-Relative Residual (relative to healthy frames)
        # anatomy_prototypes = EMA centroids of pathology-free (healthy) frames
        # cluster_loss is also applied only on healthy frames -> pathology frames naturally deviate
        # residual = "how different is this frame from the healthy organ centroid" = pathology signal
        if self.anatomy_proto_initialized:
            norm_proto_a   = F.normalize(self.anatomy_prototypes, dim=-1)  # [8, hidden]
            norm_x_a_      = F.normalize(x_a, dim=-1)                      # [B, T, hidden]
            anat_sim_      = norm_x_a_ @ norm_proto_a.T                    # [B, T, 8]
            anat_weights_  = F.softmax(anat_sim_, dim=-1)                  # [B, T, 8]
            proto_assigned = anat_weights_ @ self.anatomy_prototypes        # [B, T, hidden]
            x_residual     = x_a - proto_assigned                          # [B, T, hidden]
        else:
            x_residual = x_a  # fallback when not yet initialized

        if raw_features is not None:
            x_p_base = x_residual + self.raw_feat_proj(raw_features)    # [B, T, hidden]
        else:
            x_p_base = x_residual

        path_adj = self._similarity_adj(x_p_base)                       # [B, T, T]
        x_p = self.gelu(self.gc_p(x_p_base, path_adj))                 # [B, T, hidden]

        # Frame-level skip connection: direct path bypassing the GCN
        # Issue: GCN can dilute hematin by mixing it with blood neighbors -> AP instability
        # Fix: add x_p_base (pre-GCN self signal) directly to preserve per-frame identity
        # GCN adds context; the original signal survives via the skip connection
        x_p = x_p + x_p_base

        pathology_logits = self.pathology_head(x_p)                     # [B, T, 9]

        # 6. Pathology Prototype Gate
        if raw_features is not None:
            proto_gate = self._prototype_gate(raw_features)             # [B, T, 9]
            proto_w    = torch.sigmoid(self.proto_weight)
            pathology_logits = pathology_logits * (1.0 + proto_w * proto_gate)

        if return_all:
            return anatomy_logits, pathology_logits, x_a, x_residual
        if return_features:
            return anatomy_logits, pathology_logits, x_a
        return anatomy_logits, pathology_logits

    def _prototype_gate(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between raw DINOv2 features and each pathology prototype.
        -> sigmoid -> [B, T, 9] gate values (0~1)

        Has no effect on output when proto_weight=0.
        """
        norm_feat  = F.normalize(raw_features, dim=-1)          # [B, T, D]
        norm_proto = F.normalize(self.pathology_prototypes, dim=-1)  # [9, D]
        sim = norm_feat @ norm_proto.T                           # [B, T, 9]
        return torch.sigmoid(sim)

    @torch.no_grad()
    def update_prototypes(
        self,
        raw_features: torch.Tensor,
        pathology_labels: torch.Tensor,
        momentum: float = 0.99,
    ):
        """
        EMA update of per-class pathology prototype centroids.
        Called every batch during the training loop.

        Args:
            raw_features     : [B, T, feat_dim]
            pathology_labels : [B, T, 9]  (0 or 1)
            momentum         : EMA coefficient (closer to 1 = slower update)
        """
        for cls_idx in range(len(PATHOLOGY_LABELS)):
            mask = pathology_labels[:, :, cls_idx] == 1  # [B, T] bool
            if not mask.any():
                continue
            cls_feats   = raw_features[mask]               # [N, D]
            new_proto   = cls_feats.mean(dim=0)            # [D]

            if not self.proto_initialized:
                self.pathology_prototypes[cls_idx] = new_proto
            else:
                self.pathology_prototypes[cls_idx] = (
                    momentum * self.pathology_prototypes[cls_idx]
                    + (1 - momentum) * new_proto
                )
        self.proto_initialized = True

    @torch.no_grad()
    def update_anatomy_prototypes(
        self,
        x_a: torch.Tensor,
        anatomy_labels: torch.Tensor,
        pathology_labels: torch.Tensor = None,
        momentum: float = 0.99,
    ):
        """
        EMA update of per-class anatomy prototype centroids (in hidden space).
        Uses only pathology-free (healthy) frames -> prototype = "healthy organ centroid"

        Args:
            x_a              : [B, T, hidden_dim]  anatomy GCN output (detached)
            anatomy_labels   : [B, T, 8]  (0 or 1)
            pathology_labels : [B, T, 9]  pathology labels (used to filter healthy frames)
            momentum         : EMA coefficient
        """
        # Mask for frames with no pathology labels
        if pathology_labels is not None:
            healthy_mask = pathology_labels.sum(dim=-1) == 0  # [B, T]
        else:
            healthy_mask = torch.ones(x_a.shape[:2], dtype=torch.bool, device=x_a.device)

        for cls_idx in range(len(ANATOMY_LABELS)):
            anat_mask = anatomy_labels[:, :, cls_idx] == 1    # [B, T]
            mask      = anat_mask & healthy_mask               # only healthy frames of this organ
            if not mask.any():
                continue
            cls_feats = x_a[mask]              # [N, hidden_dim]
            new_proto = cls_feats.mean(dim=0)  # [hidden_dim]

            if not self.anatomy_proto_initialized:
                self.anatomy_prototypes[cls_idx] = new_proto
            else:
                self.anatomy_prototypes[cls_idx] = (
                    momentum * self.anatomy_prototypes[cls_idx]
                    + (1 - momentum) * new_proto
                )
        self.anatomy_proto_initialized = True
