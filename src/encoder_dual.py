from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .config import ResDyNetConfig


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: List[int],
        out_features: int,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderDual(nn.Module):
    """
    Input:
        y_hist: (B, n_a, n_y)
        u_hist: (B, n_b, n_u)
    Output:
        x0: (B, n_x)
    """

    def __init__(self, cfg: ResDyNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        in_features = cfg.n_a * cfg.n_y + cfg.n_b * cfg.n_u

        self.linear_branch = nn.Linear(in_features, cfg.n_x)
        self.nonlinear_branch = MLP(
            in_features=in_features,
            hidden_sizes=cfg.encoder_hidden,
            out_features=cfg.n_x,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )

    def forward(self, y_hist: torch.Tensor, u_hist: torch.Tensor) -> torch.Tensor:
        module_device = self.linear_branch.weight.device
        if y_hist.device != module_device:
            raise RuntimeError(f"y_hist is on {y_hist.device}, but encoder is on {module_device}")
        if u_hist.device != module_device:
            raise RuntimeError(f"u_hist is on {u_hist.device}, but encoder is on {module_device}")

        y_flat = y_hist.reshape(y_hist.shape[0], -1)
        u_flat = u_hist.reshape(u_hist.shape[0], -1)
        z = torch.cat([y_flat, u_flat], dim=-1)
        return self.linear_branch(z) + self.nonlinear_branch(z)
