"""
GCN layer implementations — based on VadCLIP, device-agnostic version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform


class GraphConvolution(nn.Module):
    """
    Standard GCN layer (Kipf & Welling, 2017).
    Includes residual connection: if in_features != out_features, uses a Linear projection.
    """

    def __init__(self, in_features: int, out_features: int, residual: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if residual:
            if in_features == out_features:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, T, in_features]
        adj : [B, T, T]  normalized adjacency matrix
        """
        support = x @ self.weight          # [B, T, out_features]
        output = adj @ support             # [B, T, out_features]
        if self.residual is not None:
            output = output + self.residual(x)
        return output


class DistanceAdj(nn.Module):
    """
    Temporal distance-based adjacency matrix.
    Closer frames get higher weights: exp(-|i-j| / sigma).
    """

    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns: [B, T, T]
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # [T, T]
        adj = torch.exp(-dist / (self.sigma.abs() + 1e-6))                 # [T, T]
        # Row-normalize (row sum = 1)
        adj = F.softmax(adj, dim=-1)
        return adj.unsqueeze(0).expand(batch_size, -1, -1)                  # [B, T, T]
