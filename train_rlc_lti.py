from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.signal import chirp
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.autoencoder_resnet_model import AutoencoderResNetModel
from src.config import ResDyNetConfig
from src.dynamical_system_dataset import DynamicalSystemDataset, to_numpy_2d, to_torch_2d
from src.train_utils import (
    rollout_on_loader,
    train_model_multistep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResDyNet on the second-order linear RLC dataset and fit an affine true-state/latent-state map."
    )
    parser.add_argument("--dataset-path", default="dataset/rlc_lti_dataset.mat")
    parser.add_argument("--checkpoint-path", default="checkpoints/rlc_lti_best.pth")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--n-x", type=int, default=2)
    parser.add_argument("--n-a", type=int, default=20)
    parser.add_argument("--n-b", type=int, default=20)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)

    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--transition-hidden", type=int, default=128)
    parser.add_argument("--transition-blocks", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="tanh")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=50)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--tail-start", type=int, default=20)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-warm", type=int, default=800)
    parser.add_argument("--n-free", type=int, default=1200)
    parser.add_argument("--input-amplitude", type=float, default=2.0)
    parser.add_argument("--warm-frequency", type=float, default=2.0)
    parser.add_argument("--chirp-f-min", type=float, default=0.05)
    parser.add_argument("--chirp-f-max", type=float, default=5.0)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def _require_field(obj, field_name: str):
    if not hasattr(obj, field_name):
        raise KeyError(f"Missing field {field_name!r} in MATLAB struct.")
    return getattr(obj, field_name)


def load_rlc_lti_data(path: str | Path, dtype: torch.dtype = torch.float32) -> dict[str, object]:
    raw = loadmat(Path(path), squeeze_me=True, struct_as_record=False)
    if "dataset" not in raw:
        raise KeyError("Expected top-level key 'dataset' in MAT file.")

    dataset = raw["dataset"]
    splits: dict[str, object] = {}
    raw_shapes: dict[str, tuple[int, ...]] = {}

    for split_name in ("train", "val", "test"):
        split = _require_field(dataset, split_name)
        u = to_torch_2d(_require_field(split, "u"), dtype=dtype)
        y = to_torch_2d(_require_field(split, "y"), dtype=dtype)
        x = to_torch_2d(_require_field(split, "x"), dtype=dtype)

        splits[f"u_{split_name}"] = u
        splits[f"y_{split_name}"] = y
        splits[f"x_{split_name}"] = x
        raw_shapes[f"u_{split_name}"] = tuple(u.shape)
        raw_shapes[f"y_{split_name}"] = tuple(y.shape)
        raw_shapes[f"x_{split_name}"] = tuple(x.shape)

    u_scaler = StandardScaler()
    y_scaler = StandardScaler()

    u_train_scaled = u_scaler.fit_transform(to_numpy_2d(splits["u_train"], "u_train"))
    y_train_scaled = y_scaler.fit_transform(to_numpy_2d(splits["y_train"], "y_train"))
    u_val_scaled = u_scaler.transform(to_numpy_2d(splits["u_val"], "u_val"))
    y_val_scaled = y_scaler.transform(to_numpy_2d(splits["y_val"], "y_val"))
    u_test_scaled = u_scaler.transform(to_numpy_2d(splits["u_test"], "u_test"))
    y_test_scaled = y_scaler.transform(to_numpy_2d(splits["y_test"], "y_test"))

    A = np.asarray(_require_field(dataset, "A"), dtype=np.float64)
    B = np.asarray(_require_field(dataset, "B"), dtype=np.float64).reshape(-1, 1)
    C = np.asarray(_require_field(dataset, "Cmat"), dtype=np.float64).reshape(1, -1)
    D = float(np.asarray(_require_field(dataset, "D"), dtype=np.float64).reshape(()))
    Ts = float(np.asarray(_require_field(dataset, "Ts"), dtype=np.float64).reshape(()))

    return {
        "u_train": to_torch_2d(u_train_scaled, dtype=dtype),
        "y_train": to_torch_2d(y_train_scaled, dtype=dtype),
        "u_val": to_torch_2d(u_val_scaled, dtype=dtype),
        "y_val": to_torch_2d(y_val_scaled, dtype=dtype),
        "u_test": to_torch_2d(u_test_scaled, dtype=dtype),
        "y_test": to_torch_2d(y_test_scaled, dtype=dtype),
        "x_train": splits["x_train"],
        "x_val": splits["x_val"],
        "x_test": splits["x_test"],
        "u_scaler": u_scaler,
        "y_scaler": y_scaler,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "Ts": Ts,
        "raw_shapes": raw_shapes,
    }


def build_cfg(args: argparse.Namespace, n_u: int, n_y: int) -> ResDyNetConfig:
    if args.n_x != 2:
        raise ValueError(
            f"This experiment requires latent dimension n_x = 2 to match the true system state. Received n_x={args.n_x}."
        )
    return ResDyNetConfig(
        n_u=n_u,
        n_y=n_y,
        n_x=args.n_x,
        n_a=args.n_a,
        n_b=args.n_b,
        m=args.m,
        horizon=args.horizon,
        encoder_hidden=args.encoder_hidden,
        transition_hidden=args.transition_hidden,
        transition_blocks=args.transition_blocks,
        decoder_hidden=args.decoder_hidden,
        activation=args.activation,
    )


def build_loader(dataset: DynamicalSystemDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


@torch.no_grad()
def latent_rollout_from_sequence(
    model: AutoencoderResNetModel,
    u: torch.Tensor,
    y: torch.Tensor,
    cfg: ResDyNetConfig,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    model = model.to(device)
    model.eval()

    k0 = max(cfg.n_a, cfg.n_b, cfg.m)
    T = int(u.shape[0])
    if T <= k0:
        raise ValueError(f"Sequence too short for latent rollout: T={T}, k0={k0}")

    y_hist = y[k0 - cfg.n_a : k0].unsqueeze(0).to(device)
    u_hist = u[k0 - cfg.n_b : k0].unsqueeze(0).to(device)

    x_k = model.encoder(y_hist, u_hist)
    latent_states = [x_k.squeeze(0).cpu()]

    for k in range(k0, T - 1):
        u_k = u[k].unsqueeze(0).to(device)
        x_k = model.transition(x_k, u_k)
        latent_states.append(x_k.squeeze(0).cpu())

    return torch.stack(latent_states, dim=0), k0


def solve_affine_map_closed_form(
    x_true: np.ndarray,
    x_latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if x_true.ndim != 2 or x_latent.ndim != 2:
        raise ValueError("x_true and x_latent must both be 2D arrays.")
    if x_true.shape != x_latent.shape:
        raise ValueError(f"Shape mismatch: {x_true.shape} vs {x_latent.shape}")
    x_tilde = np.concatenate(
        [x_true, np.ones((x_true.shape[0], 1), dtype=x_true.dtype)],
        axis=1,
    )
    theta, *_ = np.linalg.lstsq(x_tilde, x_latent, rcond=None)
    p = theta[:-1, :].T
    c = theta[-1, :].reshape(-1, 1)
    return p, c


def affine_predict(x_true: np.ndarray, p: np.ndarray, c: np.ndarray) -> np.ndarray:
    return x_true @ p.T + c.reshape(1, -1)


def fit_affine_similarity_centered(
    x_true: np.ndarray,
    x_latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if x_true.ndim != 2 or x_latent.ndim != 2:
        raise ValueError("x_true and x_latent must both be 2D arrays.")
    if x_true.shape != x_latent.shape:
        raise ValueError(f"Shape mismatch: {x_true.shape} vs {x_latent.shape}")

    x_true_t = x_true.T
    x_latent_t = x_latent.T
    x_mean = np.mean(x_true_t, axis=1, keepdims=True)
    xhat_mean = np.mean(x_latent_t, axis=1, keepdims=True)

    x_centered = x_true_t - x_mean
    xhat_centered = x_latent_t - xhat_mean

    p = xhat_centered @ np.linalg.pinv(x_centered)
    c = xhat_mean - p @ x_mean
    return p, c


def inverse_affine_predict(x_latent: np.ndarray, p: np.ndarray, c: np.ndarray) -> np.ndarray:
    p_inv = np.linalg.pinv(p)
    return (p_inv @ (x_latent.T - c)).T


def true_system_step(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: float,
    x: np.ndarray,
    u_scalar: float,
) -> tuple[np.ndarray, np.ndarray]:
    u_vec = np.array([[float(u_scalar)]], dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    y = (C.astype(np.float32) @ x + np.float32(D) * u_vec).astype(np.float32)
    x_next = (A.astype(np.float32) @ x + B.astype(np.float32) @ u_vec).astype(np.float32)
    return y, x_next


def update_hist_1d(hist: torch.Tensor, new_val: float) -> torch.Tensor:
    new_entry = torch.tensor([[new_val]], device=hist.device, dtype=hist.dtype)
    return torch.cat([hist[:, 1:, :], new_entry.unsqueeze(0)], dim=1)


def make_input_sin_then_zero(N: int, Ts: float, k_switch: int, A_in: float, f_warm: float) -> np.ndarray:
    t = np.arange(N, dtype=np.float32) * np.float32(Ts)
    u = np.zeros((N, 1), dtype=np.float32)
    u[:k_switch, 0] = (A_in * np.sin(2 * np.pi * f_warm * t[:k_switch])).astype(np.float32)
    u[k_switch:, 0] = 0.0
    return u


def make_input_sin_then_chirp(
    N: int,
    Ts: float,
    k_switch: int,
    A_in: float,
    f_warm: float,
    f_min: float = 0.05,
    f_max: float = 5.0,
) -> np.ndarray:
    t = np.arange(N, dtype=np.float32) * np.float32(Ts)
    u = np.zeros((N, 1), dtype=np.float32)
    u[:k_switch, 0] = (A_in * np.sin(2 * np.pi * f_warm * t[:k_switch])).astype(np.float32)

    tau = t[k_switch:] - t[k_switch]
    T_chirp = float((N - k_switch) * Ts)
    f0 = float(f_min)
    f1 = float((f_max - f_min) / (2 * max(T_chirp, 1e-9)))
    phi0 = float(2 * np.pi * f_warm * t[k_switch])
    u[k_switch:, 0] = (A_in * np.sin(phi0 + 2 * np.pi * (f1 * tau**2 + f0 * tau))).astype(np.float32)
    return u


def simulate_linear_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: float,
    u: np.ndarray,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    n_steps = u.shape[0]
    n_x = A.shape[0]

    x = np.zeros((n_steps, n_x), dtype=np.float64)
    y = np.zeros((n_steps, 1), dtype=np.float64)

    x_k = np.zeros((n_x, 1), dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(n_x, 1)
    for k in range(n_steps):
        x[k] = x_k[:, 0]
        y[k] = (C @ x_k + D * u[k : k + 1])[0, 0]
        x_k = A @ x_k + B @ u[k : k + 1]

    return x, y


def build_chirp_then_zero_input(
    chirp_samples: int,
    free_decay_samples: int,
    Ts: float,
    f0_hz: float,
    f1_hz: float,
    amplitude: float,
) -> np.ndarray:
    t_chirp = np.arange(chirp_samples, dtype=np.float64) * Ts
    if chirp_samples <= 1:
        chirp_signal = np.zeros((chirp_samples,), dtype=np.float64)
    else:
        chirp_signal = amplitude * chirp(
            t_chirp,
            f0=f0_hz,
            t1=t_chirp[-1],
            f1=f1_hz,
            method="linear",
        )
    zero_tail = np.zeros((free_decay_samples,), dtype=np.float64)
    return np.concatenate([chirp_signal, zero_tail], axis=0)


def fit_and_report_affine_map(
    *,
    label: str,
    x_true: torch.Tensor,
    x_latent: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    x_true_np = x_true.detach().cpu().numpy()
    x_latent_np = x_latent.detach().cpu().numpy()
    p_closed, c_closed = solve_affine_map_closed_form(x_true_np, x_latent_np)
    p, c = fit_affine_similarity_centered(x_true_np, x_latent_np)
    residual = x_latent_np - affine_predict(x_true_np, p, c)

    print(f"\nAffine map fitted on {label}:", flush=True)
    print("Closed-form P =", flush=True)
    print(p_closed, flush=True)
    print("Closed-form c =", flush=True)
    print(c_closed, flush=True)
    print("P =", flush=True)
    print(p, flush=True)
    print("c =", flush=True)
    print(c, flush=True)
    print(f"Residual Frobenius norm ({label}): {np.linalg.norm(residual, ord='fro'):.8e}", flush=True)
    return p, c


def analyze_similarity_and_plots(
    prefix: str,
    out_dir: str | Path,
    na: int,
    nb: int,
    X: np.ndarray,
    X_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p, c = fit_affine_similarity_centered(X.T, X_hat.T)
    ones = np.ones((1, X.shape[1]), dtype=X.dtype)
    p_inv = np.linalg.pinv(p)
    X_hat_phys = p_inv @ (X_hat - c @ ones)

    k0 = max(na, nb)
    k_axis = np.arange(X.shape[1]) + k0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(X[0, :], X[1, :], label="True state $x_k$", linewidth=2)
    ax.plot(X_hat_phys[0, :], X_hat_phys[1, :], "--", label="Mapped latent $P^{-1}(\\hat x_k-c)$", linewidth=2)
    ax.scatter(X[0, 0], X[1, 0], s=60, label="Initial")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{prefix} - 2D trajectory")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_traj_2d.png", dpi=200)
    plt.show()
    plt.close(fig)

    err_k = np.linalg.norm(X - X_hat_phys, axis=0)
    plt.figure(figsize=(9, 4))
    plt.plot(k_axis, err_k, linewidth=1.5)
    plt.grid(True)
    plt.xlabel(r"time index $k$")
    plt.ylabel(r"$\|x_k - P^{-1}(\hat x_k - c)\|_2$")
    plt.title(f"{prefix} - pointwise state reconstruction error")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_state_recon_error.png", dpi=200)
    plt.show()
    plt.close()

    R = X_hat - (p @ X + c @ ones)
    err_sq = np.sum(R**2, axis=0)
    plt.figure(figsize=(9, 4))
    plt.plot(k_axis, err_sq, linewidth=1.5)
    plt.grid(True)
    plt.xlabel(r"time index $k$")
    plt.ylabel(r"$\|\hat x_k - (P x_k + c)\|_2^2$")
    plt.title(f"{prefix} - pointwise affine LS residual")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_affine_ls_residual.png", dpi=200)
    plt.show()
    plt.close()

    return p, c


def plot_outputs(prefix: str, out_dir: str | Path, t: np.ndarray, y_real: np.ndarray, y_hat: np.ndarray, k_switch: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_real[:, 0], label="y_real(k)")
    plt.plot(t, y_hat[:, 0], "--", label="y_hat(k)")
    plt.axvline(t[k_switch], linestyle=":", linewidth=2, label=f"switch k={k_switch}")
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("y")
    plt.title(prefix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_y_plot.png", dpi=200)
    plt.show()
    plt.close()


@torch.no_grad()
def rollout_encoder_decoder_teacher_forcing(
    model: AutoencoderResNetModel,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: float,
    u_raw: np.ndarray,
    u_scaler: StandardScaler,
    y_scaler: StandardScaler,
    cfg: ResDyNetConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    N = u_raw.shape[0]
    n = A.shape[0]

    zero_u_norm = float(u_scaler.transform(np.array([[0.0]], dtype=np.float32))[0, 0])
    zero_y_norm = float(y_scaler.transform(np.array([[0.0]], dtype=np.float32))[0, 0])

    x_real = np.zeros((n, 1), dtype=np.float32)
    y_hist = torch.full((1, cfg.n_a, cfg.n_y), zero_y_norm, dtype=torch.float32, device=device)
    u_hist = torch.full((1, cfg.n_b, cfg.n_u), zero_u_norm, dtype=torch.float32, device=device)

    y_real = np.zeros((N, 1), dtype=np.float32)
    x_real_hist = np.zeros((N, n), dtype=np.float32)
    y_hat = np.zeros((N, 1), dtype=np.float32)
    x_lat_hist = np.zeros((N, n), dtype=np.float32)

    for i in range(N):
        u_i_raw = float(u_raw[i, 0])
        u_i_norm = float(u_scaler.transform(np.array([[u_i_raw]], dtype=np.float32))[0, 0])

        x_real_hist[i, :] = x_real.reshape(-1)
        y_i_raw, x_next = true_system_step(A, B, C, D, x_real, u_i_raw)
        y_i_scalar = float(y_i_raw.item())
        y_i_norm = float(y_scaler.transform(np.array([[y_i_scalar]], dtype=np.float32))[0, 0])

        y_real[i, 0] = y_i_scalar
        x_real = x_next

        y_hist = update_hist_1d(y_hist, y_i_norm)
        u_hist = update_hist_1d(u_hist, u_i_norm)

        x_lat = model.encoder(y_hist, u_hist)
        x_lat_hist[i, :] = x_lat.squeeze(0).detach().cpu().numpy()

        u_dec = torch.tensor([[u_i_norm]], dtype=torch.float32, device=device)
        y_hat_norm = model.decoder(x_lat, u_dec)[:, : cfg.n_y].detach().cpu().numpy()
        y_hat[i, :] = y_scaler.inverse_transform(y_hat_norm).reshape(-1)

    return y_real, x_real_hist, y_hat, x_lat_hist


@torch.no_grad()
def rollout_resnet_decoder(
    model: AutoencoderResNetModel,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: float,
    u_raw: np.ndarray,
    u_scaler: StandardScaler,
    y_scaler: StandardScaler,
    cfg: ResDyNetConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    N = u_raw.shape[0]
    n = A.shape[0]

    zero_u_norm = float(u_scaler.transform(np.array([[0.0]], dtype=np.float32))[0, 0])
    zero_y_norm = float(y_scaler.transform(np.array([[0.0]], dtype=np.float32))[0, 0])

    x_real = np.zeros((n, 1), dtype=np.float32)
    y_hist = torch.full((1, cfg.n_a, cfg.n_y), zero_y_norm, dtype=torch.float32, device=device)
    u_hist = torch.full((1, cfg.n_b, cfg.n_u), zero_u_norm, dtype=torch.float32, device=device)

    y_real = np.zeros((N, 1), dtype=np.float32)
    x_real_hist = np.zeros((N, n), dtype=np.float32)
    y_hat = np.zeros((N, 1), dtype=np.float32)
    x_lat_hist = np.zeros((N, n), dtype=np.float32)

    x_lat = model.encoder(y_hist, u_hist)

    for i in range(N):
        u_i_raw = float(u_raw[i, 0])
        u_i_norm = float(u_scaler.transform(np.array([[u_i_raw]], dtype=np.float32))[0, 0])

        x_real_hist[i, :] = x_real.reshape(-1)
        y_i_raw, x_next = true_system_step(A, B, C, D, x_real, u_i_raw)
        y_i_scalar = float(y_i_raw.item())
        y_i_norm = float(y_scaler.transform(np.array([[y_i_scalar]], dtype=np.float32))[0, 0])

        y_real[i, 0] = y_i_scalar
        x_real = x_next

        y_hist = update_hist_1d(y_hist, y_i_norm)
        u_hist = update_hist_1d(u_hist, u_i_norm)

        u_k = torch.tensor([[u_i_norm]], dtype=torch.float32, device=device)
        x_lat = model.transition(x_lat, u_k)
        x_lat_hist[i, :] = x_lat.squeeze(0).detach().cpu().numpy()

        y_hat_norm = model.decoder(x_lat, u_k)[:, : cfg.n_y].detach().cpu().numpy()
        y_hat[i, :] = y_scaler.inverse_transform(y_hat_norm).reshape(-1)

    return y_real, x_real_hist, y_hat, x_lat_hist


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    print(f"Using device: {device}", flush=True)

    data = load_rlc_lti_data(args.dataset_path, dtype=torch.float32)
    print("Loaded RLC dataset:", flush=True)
    for key, shape in data["raw_shapes"].items():
        print(f"  {key}: {shape}", flush=True)

    n_u = int(data["u_train"].shape[1])
    n_y = int(data["y_train"].shape[1])
    cfg = build_cfg(args, n_u=n_u, n_y=n_y)

    train_ds = DynamicalSystemDataset(data["u_train"], data["y_train"], cfg)
    val_ds = DynamicalSystemDataset(data["u_val"], data["y_val"], cfg)
    test_ds = DynamicalSystemDataset(data["u_test"], data["y_test"], cfg)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = build_loader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}", flush=True)
    print(f"Val samples:   {len(val_ds)}", flush=True)
    print(f"Test samples:  {len(test_ds)}", flush=True)

    model = AutoencoderResNetModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    gamma = torch.ones(cfg.horizon, dtype=torch.float32, device=device)

    history = train_model_multistep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gamma=gamma,
        m=cfg.m,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        tail_start=args.tail_start,
        clip_grad_norm=args.clip_grad_norm,
    )

    print("\nTraining summary:", flush=True)
    print(f"  Best epoch:     {history['best_epoch']}", flush=True)
    print(f"  Best val loss:  {history['best_val_loss']:.8f}", flush=True)
    print(f"  Stop epoch:     {history['stop_epoch']}", flush=True)
    print(f"  Checkpoint:     {args.checkpoint_path}", flush=True)

    test_rollout = rollout_on_loader(model=model, loader=test_loader, device=device)
    test_window_rmse = torch.sqrt(
        torch.mean((test_rollout["Y_hat_all"] - test_rollout["Y_true_all"]).pow(2))
    )
    print("\nWindowed test metric:", flush=True)
    print(f"  Normalized-space RMSE: {float(test_window_rmse):.8f}", flush=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path("result_rlc") / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    N = args.n_warm + args.n_free
    t = np.arange(N) * float(data["Ts"])
    k_switch = args.n_warm
    inputs = {
        "sin_then_zero": make_input_sin_then_zero(
            N=N,
            Ts=float(data["Ts"]),
            k_switch=k_switch,
            A_in=args.input_amplitude,
            f_warm=args.warm_frequency,
        ),
        "sin_then_chirp": make_input_sin_then_chirp(
            N=N,
            Ts=float(data["Ts"]),
            k_switch=k_switch,
            A_in=args.input_amplitude,
            f_warm=args.warm_frequency,
            f_min=args.chirp_f_min,
            f_max=args.chirp_f_max,
        ),
    }

    k0 = max(cfg.n_a, cfg.n_b)
    for input_name, u_raw in inputs.items():
        exp_dir = out_root / input_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        y_real_ed, x_real_ed, y_hat_ed, x_lat_ed = rollout_encoder_decoder_teacher_forcing(
            model=model,
            A=data["A"],
            B=data["B"],
            C=data["C"],
            D=float(data["D"]),
            u_raw=u_raw,
            u_scaler=data["u_scaler"],
            y_scaler=data["y_scaler"],
            cfg=cfg,
            device=device,
        )
        plot_outputs(
            prefix=f"{input_name}_ENCDEC",
            out_dir=exp_dir,
            t=t,
            y_real=y_real_ed,
            y_hat=y_hat_ed,
            k_switch=k_switch,
        )
        X = x_real_ed[k0:, :].T.astype(np.float32)
        X_hat = x_lat_ed[k0:, :].T.astype(np.float32)
        analyze_similarity_and_plots(
            prefix=f"{input_name}_ENCDEC",
            out_dir=exp_dir,
            na=cfg.n_a,
            nb=cfg.n_b,
            X=X,
            X_hat=X_hat,
        )

        y_real_rn, x_real_rn, y_hat_rn, x_lat_rn = rollout_resnet_decoder(
            model=model,
            A=data["A"],
            B=data["B"],
            C=data["C"],
            D=float(data["D"]),
            u_raw=u_raw,
            u_scaler=data["u_scaler"],
            y_scaler=data["y_scaler"],
            cfg=cfg,
            device=device,
        )
        plot_outputs(
            prefix=f"{input_name}_RESNETDEC",
            out_dir=exp_dir,
            t=t,
            y_real=y_real_rn,
            y_hat=y_hat_rn,
            k_switch=k_switch,
        )
        X_r = x_real_rn[k0 + 1 :, :].T.astype(np.float32)
        Xhat_r = x_lat_rn[k0:-1, :].T.astype(np.float32)
        analyze_similarity_and_plots(
            prefix=f"{input_name}_RESNETDEC",
            out_dir=exp_dir,
            na=cfg.n_a,
            nb=cfg.n_b,
            X=X_r,
            X_hat=Xhat_r,
        )

    print(f"\nAll results saved under: {out_root}", flush=True)


if __name__ == "__main__":
    main()
