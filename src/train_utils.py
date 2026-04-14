# from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .autoencoder_resnet_model import AutoencoderResNetModel
from .config import ResDyNetConfig
from .dynamical_system_dataset import to_numpy_2d, to_torch_2d

import os


def _state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cpu_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state_dict[key] = value.detach().cpu()
        elif isinstance(value, dict):
            cpu_state_dict[key] = _state_dict_to_cpu(value)
        elif isinstance(value, list):
            cpu_state_dict[key] = [
                item.detach().cpu() if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict


def save_checkpoint_safe(
    model,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    checkpoint = {
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "optimizer_state_dict": (
            _state_dict_to_cpu(optimizer.state_dict()) if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            _state_dict_to_cpu(scheduler.state_dict()) if scheduler is not None else None
        ),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint_state(checkpoint_path: str | Path, map_location: torch.device | str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint

    # Backward compatibility with legacy checkpoints storing only model.state_dict().
    return {
        "model_state_dict": checkpoint,
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "epoch": None,
        "best_val_loss": None,
    }


def compute_nrmse_percent(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")
    rmse = torch.sqrt(torch.mean((y_true - y_pred).pow(2), dim=0))
    denom = torch.std(y_true, dim=0, unbiased=False).clamp_min(1e-12)
    return 100.0 * rmse / denom


def loss_multistep(
    pred_dict: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    gamma_true: torch.Tensor,
    gamma_weights: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """
    Flexible loss:
        - if m == 0: compare only current prediction Y_hat with y_true
        - if m > 0 : compare full decoder window Gamma_hat with gamma_true
    """
    H = gamma_weights.numel()
    weights = gamma_weights.view(1, H, 1).to(y_true.device)
    weights = weights / weights.mean()

    if m == 0:
        se = (pred_dict["Y_hat"] - y_true).pow(2)           # (B,H,n_y)
    else:
        se = (pred_dict["Gamma_hat"] - gamma_true).pow(2)   # (B,H,(m+1)*n_y)

    return (weights * se).mean()


def train_model_multistep(
    model: AutoencoderResNetModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    gamma,
    m: int,
    num_epochs: int = 1500,
    patience: int = 10_000,
    checkpoint_path: str = "checkpoints/best_resdynet.pth",
    tail_start: int = 900,
    clip_grad_norm: float | None = 1.0,
):
    model = model.to(device)
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    stop_epoch = None

    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=device)
    train_losses = []
    val_losses = []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            y_hist = batch["y_hist"].to(device)
            u_hist = batch["u_hist"].to(device)
            u_seq = batch["u_seq"].to(device)
            y_true = batch["y_true"].to(device)
            gamma_true = batch["gamma_true"].to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(y_hist, u_hist, u_seq)

            loss = loss_multistep(
                pred_dict=pred,
                y_true=y_true,
                gamma_true=gamma_true,
                gamma_weights=gamma,
                m=m,
            )

            
            loss.backward()

            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            
            optimizer.step()

            train_loss_sum += loss.detach().item()


        avg_train_loss = train_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                y_hist = batch["y_hist"].to(device)
                u_hist = batch["u_hist"].to(device)
                u_seq = batch["u_seq"].to(device)
                y_true = batch["y_true"].to(device)
                gamma_true = batch["gamma_true"].to(device)

                pred = model(y_hist, u_hist, u_seq)
                loss = loss_multistep(
                    pred_dict=pred,
                    y_true=y_true,
                    gamma_true=gamma_true,
                    gamma_weights=gamma,
                    m=m,
                )
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        val_losses.append(avg_val_loss)

        if scheduler is not None and epoch >= tail_start:
            scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val_loss + 1e-12 < best_val:
            best_val = avg_val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint_safe(
                model,
                checkpoint_path,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val,
            )
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:04d} | "
            f"Train {avg_train_loss:.8f} | "
            f"Val {avg_val_loss:.8f} | "
            f"LR {current_lr:.3e}"
        )

        if no_improve >= patience:
            stop_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    if stop_epoch is None:
        stop_epoch = num_epochs

    checkpoint = load_checkpoint_state(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    model.to(device)

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
    }


@torch.no_grad()
def rollout_on_loader(
    model: AutoencoderResNetModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model = model.to(device)
    model.eval()

    gamma_hat_all = []
    y_hat_all = []
    gamma_true_all = []
    y_true_all = []

    for batch in loader:
        y_hist = batch["y_hist"].to(device)
        u_hist = batch["u_hist"].to(device)
        u_seq = batch["u_seq"].to(device)

        pred = model(y_hist, u_hist, u_seq)

        gamma_hat_all.append(pred["Gamma_hat"].cpu())
        y_hat_all.append(pred["Y_hat"].cpu())
        gamma_true_all.append(batch["gamma_true"].cpu())
        y_true_all.append(batch["y_true"].cpu())

    return {
        "Gamma_hat_all": torch.cat(gamma_hat_all, dim=0),
        "Y_hat_all": torch.cat(y_hat_all, dim=0),
        "Gamma_true_all": torch.cat(gamma_true_all, dim=0),
        "Y_true_all": torch.cat(y_true_all, dim=0),
    }


@torch.no_grad()
def evaluate_chunked_test_sequence(
    model: AutoencoderResNetModel,
    u: torch.Tensor,
    y: torch.Tensor,
    cfg: ResDyNetConfig,
    device: torch.device,
    y_scaler: Optional[StandardScaler] = None,
) -> Dict[str, torch.Tensor]:
    model = model.to(device)
    model.eval()

    k0 = max(cfg.n_a, cfg.n_b, cfg.m)
    T = int(u.shape[0])
    if T <= k0:
        raise ValueError("Sequence too short for chunked evaluation")

    y_pred_chunks = []
    y_true_chunks = []

    start = k0
    while start < T:
        chunk_len = min(cfg.horizon, T - start)

        y_hist = y[start - cfg.n_a : start].unsqueeze(0).to(device)
        u_hist = u[start - cfg.n_b : start].unsqueeze(0).to(device)
        u_seq = u[start : start + chunk_len].unsqueeze(0).to(device)

        pred = model(y_hist, u_hist, u_seq)
        y_pred_norm_chunk = pred["Y_hat"].squeeze(0).cpu()
        y_true_norm_chunk = y[start : start + chunk_len].cpu()

        y_pred_chunks.append(y_pred_norm_chunk)
        y_true_chunks.append(y_true_norm_chunk)

        start += chunk_len

    y_pred_norm = torch.cat(y_pred_chunks, dim=0)
    y_true_norm = torch.cat(y_true_chunks, dim=0)

    if y_scaler is not None:
        y_true_np = y_scaler.inverse_transform(to_numpy_2d(y_true_norm, "y_true_norm"))
        y_pred_np = y_scaler.inverse_transform(to_numpy_2d(y_pred_norm, "y_pred_norm"))
        y_true_real = to_torch_2d(y_true_np, dtype=y_true_norm.dtype)
        y_pred_real = to_torch_2d(y_pred_np, dtype=y_pred_norm.dtype)
    else:
        y_true_real = y_true_norm
        y_pred_real = y_pred_norm

    rmse = torch.sqrt(torch.mean((y_true_real - y_pred_real).pow(2), dim=0))
    nrmse_pct = compute_nrmse_percent(y_true_real, y_pred_real)

    return {
        "y_true_norm": y_true_norm,
        "y_pred_norm": y_pred_norm,
        "y_true": y_true_real,
        "y_pred": y_pred_real,
        "rmse": rmse,
        "nrmse_pct": nrmse_pct,
    }


def plot_chunked_test_prediction(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    save_path: str = "outputs/chunked_test_prediction.png",
    title: str = "Chunked test prediction",
) -> None:
    y_true = to_torch_2d(y_true)
    y_pred = to_torch_2d(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")

    n_samples, n_y = y_true.shape
    t = np.arange(n_samples)

    fig, axes = plt.subplots(n_y, 1, figsize=(12, 3.5 * n_y), squeeze=False)
    axes = axes[:, 0]

    for j in range(n_y):
        axes[j].plot(t, y_true[:, j].cpu().numpy(), label="y", linewidth=1.5)
        axes[j].plot(t, y_pred[:, j].cpu().numpy(), label=r"$\hat{y}$", linewidth=1.5)
        axes[j].set_xlabel("k")
        axes[j].set_ylabel(f"Output {j}")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend()

    fig.suptitle(title)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
