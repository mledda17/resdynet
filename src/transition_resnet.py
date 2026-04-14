from __future__ import annotations

import torch
import torch.nn as nn

from .config import ResDyNetConfig
from .residual_block import ResidualBlock


class TransitionResNet(nn.Module):
    """
    x_{k+1} = A x_k + B u_k + Xi(x_k, u_k)
    """

    def __init__(self, cfg: ResDyNetConfig) -> None:
        super().__init__()
        self.A_layer = nn.Linear(cfg.n_x, cfg.n_x, bias=cfg.use_bias_A)
        self.B_layer = nn.Linear(cfg.n_u, cfg.n_x, bias=cfg.use_bias_B)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    n_x=cfg.n_x,
                    n_u=cfg.n_u,
                    hidden_size=cfg.transition_hidden,
                    activation=cfg.activation,
                )
                for _ in range(cfg.transition_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        module_device = self.A_layer.weight.device
        if x.device != module_device:
            raise RuntimeError(f"x is on {x.device}, but transition is on {module_device}")
        if u.device != module_device:
            raise RuntimeError(f"u is on {u.device}, but transition is on {module_device}")

        linear_part = self.A_layer(x) + self.B_layer(u)

        z = x
        for block in self.blocks:
            z = z + block(z, u)

        residual_part = z - x
        return linear_part + residual_part
