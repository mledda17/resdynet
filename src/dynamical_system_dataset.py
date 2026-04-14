from __future__ import annotations

from typing import Dict

import numpy as np
import torch as _torch
from torch.utils.data import Dataset

from .config import ResDyNetConfig


def to_numpy_2d(x, name: str) -> np.ndarray:
    if isinstance(x, _torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"{name} must have shape (T,) or (T,d), got {x.shape}")
    return x


def to_torch_2d(x, dtype: _torch.dtype = _torch.float32) -> _torch.Tensor:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return _torch.as_tensor(x, dtype=dtype)


def _coerce_scalar_index(idx: object, name: str) -> int:
    if isinstance(idx, _torch.Tensor):
        if idx.ndim == 0 or idx.numel() == 1:
            return int(idx.reshape(()).item())
        raise ValueError(f"{name} must be scalar, got tensor with shape {tuple(idx.shape)}")
    if isinstance(idx, np.ndarray):
        if idx.ndim == 0 or idx.size == 1:
            return int(idx.reshape(()).item())
        raise ValueError(f"{name} must be scalar, got array with shape {idx.shape}")
    return int(idx)


def _is_batched_index(idx: object) -> bool:
    if isinstance(idx, _torch.Tensor):
        return idx.ndim > 0 and idx.numel() > 1
    if isinstance(idx, np.ndarray):
        return idx.ndim > 0 and idx.size > 1
    return isinstance(idx, (list, tuple))


def build_gamma_window(y: _torch.Tensor, k: int, m: int) -> _torch.Tensor:
    """
    gamma_k = [y_k, y_{k-1}, ..., y_{k-m}]
    Output shape: ((m+1)*n_y,)
    """
    k = _coerce_scalar_index(k, "k")
    m = _coerce_scalar_index(m, "m")
    if k - m < 0:
        raise IndexError("Not enough history to build gamma window")
    # Prefer slice-based indexing here; it is equivalent to stacking y[k-j]
    # but avoids PyTorch scalar indexing paths that can fail in some builds.
    window = y[k - m : k + 1].flip(0)
    return window.reshape(-1)


class DynamicalSystemDataset(Dataset):
    """
    Returns:
        y_hist     : (n_a, n_y)
        u_hist     : (n_b, n_u)
        u_seq      : (H, n_u)
        y_true     : (H, n_y)
        gamma_true : (H, (m+1)*n_y)

    Convention:
        y_hist = [y_{k-n_a}, ..., y_{k-1}]
        u_hist = [u_{k-n_b}, ..., u_{k-1}]
    """

    def __init__(
        self,
        u: _torch.Tensor,
        y: _torch.Tensor,
        cfg: ResDyNetConfig,
        dtype: _torch.dtype = _torch.float32,
    ) -> None:
        super().__init__()

        if u.ndim != 2:
            raise ValueError(f"u must have shape (T,n_u), got {tuple(u.shape)}")
        if y.ndim != 2:
            raise ValueError(f"y must have shape (T,n_y), got {tuple(y.shape)}")
        if u.shape[0] != y.shape[0]:
            raise ValueError("u and y must have same number of time steps")

        self.u = u.to(dtype)
        self.y = y.to(dtype)
        self.cfg = cfg

        self.T = int(u.shape[0])
        if not (0 <= cfg.m <= cfg.n_a):
            raise ValueError(f"m must satisfy 0 <= m <= n_a, got m={cfg.m}, n_a={cfg.n_a}")

        self.k_min = max(cfg.n_a, cfg.n_b, cfg.m)
        self.k_max = self.T - cfg.horizon

        if self.k_max < self.k_min:
            raise ValueError(
                f"Sequence too short. T={self.T}, k_min={self.k_min}, k_max={self.k_max}"
            )

        self.valid_indices = list(range(self.k_min, self.k_max + 1))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, _torch.Tensor]:
        if _is_batched_index(idx):
            items = [self.__getitem__(single_idx) for single_idx in idx]
            return {key: _torch.stack([item[key] for item in items], dim=0) for key in items[0]}

        idx = _coerce_scalar_index(idx, "idx")
        k = self.valid_indices[idx]
        cfg = self.cfg

        y_hist = self.y[k - cfg.n_a : k]             # oldest -> newest
        u_hist = self.u[k - cfg.n_b : k]             # oldest -> newest
        u_seq = self.u[k : k + cfg.horizon]
        y_true = self.y[k : k + cfg.horizon]

        gamma_true = _torch.stack(
            [build_gamma_window(self.y, k + h, cfg.m) for h in range(cfg.horizon)],
            dim=0,
        )

        return {
            "y_hist": y_hist,
            "u_hist": u_hist,
            "u_seq": u_seq,
            "y_true": y_true,
            "gamma_true": gamma_true,
            "start_index": _torch.tensor(k, dtype=_torch.long),
        }
