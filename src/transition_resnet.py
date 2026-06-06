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
        self.cfg = cfg
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

    def reset_parameters_stable(self, state_decay: float = 0.95) -> None:
        with torch.no_grad():
            self.A_layer.weight.zero_()
            diag_len = min(self.A_layer.weight.shape)
            self.A_layer.weight[:diag_len, :diag_len].fill_diagonal_(state_decay)
            if self.A_layer.bias is not None:
                self.A_layer.bias.zero_()

            nn.init.xavier_uniform_(self.B_layer.weight, gain=0.1)
            if self.B_layer.bias is not None:
                self.B_layer.bias.zero_()

            for block in self.blocks:
                block.reset_output_to_zero()

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
