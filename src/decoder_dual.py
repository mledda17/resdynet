from __future__ import annotations

import torch
import torch.nn as nn

from .config import ResDyNetConfig
from .encoder_dual import MLP


class DecoderDual(nn.Module):
    """
    Input:
        x   : (B, n_x)
        u_k : (B, n_u)
    Output:
        gamma_hat: (B, (m+1) * n_y)

    Convention:
        gamma_k = [y_k, y_{k-1}, ..., y_{k-m}]
        so if m=0 -> only current prediction.

    State-space interpretation:
        y_k = g(x_k, u_k)
    """

    def __init__(self, cfg: ResDyNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        in_features = cfg.n_x + cfg.n_u
        out_features = cfg.n_gamma

        self.linear_branch = nn.Linear(in_features, out_features)
        self.nonlinear_branch = MLP(
            in_features=in_features,
            hidden_sizes=cfg.decoder_hidden,
            out_features=out_features,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )

    def forward(self, x: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        module_device = self.linear_branch.weight.device
        if x.device != module_device:
            raise RuntimeError(f"x is on {x.device}, but decoder is on {module_device}")
        if u_k.device != module_device:
            raise RuntimeError(f"u_k is on {u_k.device}, but decoder is on {module_device}")

        z = torch.cat([x, u_k], dim=-1)
        return self.linear_branch(z) + self.nonlinear_branch(z)
