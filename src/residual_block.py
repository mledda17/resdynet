from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    xi_theta(x, u) in R^{n_x}
    """

    def __init__(self, n_x: int, n_u: int, hidden_size: int, activation: str = "relu") -> None:
        super().__init__()

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.fc1 = nn.Linear(n_x + n_u, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_x)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        module_device = self.fc1.weight.device
        if x.device != module_device:
            raise RuntimeError(f"x is on {x.device}, but residual block is on {module_device}")
        if u.device != module_device:
            raise RuntimeError(f"u is on {u.device}, but residual block is on {module_device}")

        xu = torch.cat([x, u], dim=-1)
        return self.fc2(self.activation(self.fc1(xu)))
