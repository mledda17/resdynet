from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .config import ResDyNetConfig
from .decoder_dual import DecoderDual
from .encoder_dual import EncoderDual
from .transition_resnet import TransitionResNet


class AutoencoderResNetModel(nn.Module):
    """
    Forward interface kept simple and close to the reference code:
        model(y_hist, u_hist, u_seq)

    Inputs:
        y_hist: (B, n_a, n_y)
        u_hist: (B, n_b, n_u)
        u_seq : (B, H, n_u)

    Outputs:
        dict with:
            Gamma_hat: (B, H, (m+1)*n_y)
            Y_hat    : (B, H, n_y)

    Interpretation:
        encoder   -> initializes x_k from past y/u histories
        decoder   -> predicts y_k from (x_k, u_k)
        transition-> propagates x_k to x_{k+1} using the same u_k
    """

    def __init__(self, cfg: ResDyNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = EncoderDual(cfg)
        self.transition = TransitionResNet(cfg)
        self.decoder = DecoderDual(cfg)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        y_hist: torch.Tensor,
        u_hist: torch.Tensor,
        u_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if y_hist.ndim != 3:
            raise ValueError(f"y_hist must have shape (B,n_a,n_y), got {tuple(y_hist.shape)}")
        if u_hist.ndim != 3:
            raise ValueError(f"u_hist must have shape (B,n_b,n_u), got {tuple(u_hist.shape)}")
        if u_seq.ndim != 3:
            raise ValueError(f"u_seq must have shape (B,H,n_u), got {tuple(u_seq.shape)}")

        model_device = self._parameter_device()
        if y_hist.device != model_device:
            raise RuntimeError(f"y_hist is on {y_hist.device}, but model is on {model_device}")
        if u_hist.device != model_device:
            raise RuntimeError(f"u_hist is on {u_hist.device}, but model is on {model_device}")
        if u_seq.device != model_device:
            raise RuntimeError(f"u_seq is on {u_seq.device}, but model is on {model_device}")

        B, H, _ = u_seq.shape
        x = self.encoder(y_hist, u_hist)

        gamma_preds: List[torch.Tensor] = []
        y_preds: List[torch.Tensor] = []

        for h in range(H):
            u_h = u_seq[:, h, :]                       # (B, n_u)
            gamma_h = self.decoder(x, u_h)            # (B, (m+1)*n_y)
            y_h = gamma_h[:, : self.cfg.n_y]          # current prediction only

            gamma_preds.append(gamma_h)
            y_preds.append(y_h)

            x = self.transition(x, u_h)

        Gamma_hat = torch.stack(gamma_preds, dim=1)   # (B,H,(m+1)*n_y)
        Y_hat = torch.stack(y_preds, dim=1)           # (B,H,n_y)

        return {
            "Gamma_hat": Gamma_hat,
            "Y_hat": Y_hat,
        }
