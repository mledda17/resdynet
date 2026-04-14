from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.autoencoder_resnet_model import AutoencoderResNetModel
from src.config import ResDyNetConfig
from src.dynamical_system_dataset import DynamicalSystemDataset, to_numpy_2d, to_torch_2d
from src.train_utils import evaluate_chunked_test_sequence, loss_multistep


@dataclass
class EpochMetric:
    seed: int
    n_x: int
    epoch: int
    elapsed_sec: float
    train_loss: float
    val_loss: float
    test_nrmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study the effect of latent state dimension n_x on ResDyNet using the Duffing dataset."
    )
    parser.add_argument("--dataset-path", default="dataset/dataset_duffing.mat")
    parser.add_argument("--output-dir", default="outputs/duffing_nx_sensitivity")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    parser.add_argument("--nx-values", type=int, nargs="+", default=[2, 8, 14])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=1.0)

    parser.add_argument("--n-a", type=int, default=20)
    parser.add_argument("--n-b", type=int, default=20)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)

    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--transition-hidden", type=int, default=256)
    parser.add_argument("--transition-blocks", type=int, default=3)
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="tanh")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--tail-start", type=int, default=20)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_duffing_data(path: str | Path, dtype: torch.dtype = torch.float32) -> dict[str, object]:
    raw = loadmat(Path(path))
    required_keys = ["u_train", "y_train", "u_val", "y_val", "u_test", "y_test"]
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise KeyError(f"Missing keys in Duffing dataset: {missing}")

    u_train = to_torch_2d(raw["u_train"], dtype=dtype)
    y_train = to_torch_2d(raw["y_train"], dtype=dtype)
    u_val = to_torch_2d(raw["u_val"], dtype=dtype)
    y_val = to_torch_2d(raw["y_val"], dtype=dtype)
    u_test = to_torch_2d(raw["u_test"], dtype=dtype)
    y_test = to_torch_2d(raw["y_test"], dtype=dtype)

    u_scaler = StandardScaler()
    y_scaler = StandardScaler()

    u_train_scaled = u_scaler.fit_transform(to_numpy_2d(u_train, "u_train"))
    y_train_scaled = y_scaler.fit_transform(to_numpy_2d(y_train, "y_train"))
    u_val_scaled = u_scaler.transform(to_numpy_2d(u_val, "u_val"))
    y_val_scaled = y_scaler.transform(to_numpy_2d(y_val, "y_val"))
    u_test_scaled = u_scaler.transform(to_numpy_2d(u_test, "u_test"))
    y_test_scaled = y_scaler.transform(to_numpy_2d(y_test, "y_test"))

    ts = raw.get("Ts")

    return {
        "u_train": to_torch_2d(u_train_scaled, dtype=dtype),
        "y_train": to_torch_2d(y_train_scaled, dtype=dtype),
        "u_val": to_torch_2d(u_val_scaled, dtype=dtype),
        "y_val": to_torch_2d(y_val_scaled, dtype=dtype),
        "u_test": to_torch_2d(u_test_scaled, dtype=dtype),
        "y_test": to_torch_2d(y_test_scaled, dtype=dtype),
        "u_scaler": u_scaler,
        "y_scaler": y_scaler,
        "Ts": None if ts is None else float(np.asarray(ts).reshape(-1)[0]),
        "raw_shapes": {
            "u_train": tuple(raw["u_train"].shape),
            "y_train": tuple(raw["y_train"].shape),
            "u_val": tuple(raw["u_val"].shape),
            "y_val": tuple(raw["y_val"].shape),
            "u_test": tuple(raw["u_test"].shape),
            "y_test": tuple(raw["y_test"].shape),
        },
    }


def truncate_pair(
    u: torch.Tensor,
    y: torch.Tensor,
    fraction: float,
    min_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        return u, y

    target_len = max(min_steps, int(round(u.shape[0] * fraction)))
    target_len = min(target_len, u.shape[0])
    return u[:target_len], y[:target_len]


def build_cfg(args: argparse.Namespace, n_x: int) -> ResDyNetConfig:
    return ResDyNetConfig(
        n_u=1,
        n_y=1,
        n_x=n_x,
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


def build_loaders(
    data: dict[str, object],
    cfg: ResDyNetConfig,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    min_steps = max(cfg.n_a, cfg.n_b, cfg.m) + cfg.horizon + 1
    u_train, y_train = truncate_pair(data["u_train"], data["y_train"], data["train_fraction"], min_steps)
    u_val, y_val = truncate_pair(data["u_val"], data["y_val"], data["val_fraction"], min_steps)

    train_ds = DynamicalSystemDataset(u_train, y_train, cfg)
    val_ds = DynamicalSystemDataset(u_val, y_val, cfg)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate_loader_loss(
    model: AutoencoderResNetModel,
    loader: DataLoader,
    device: torch.device,
    gamma: torch.Tensor,
    m: int,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            y_hist = batch["y_hist"].to(device, non_blocking=True)
            u_hist = batch["u_hist"].to(device, non_blocking=True)
            u_seq = batch["u_seq"].to(device, non_blocking=True)
            y_true = batch["y_true"].to(device, non_blocking=True)
            gamma_true = batch["gamma_true"].to(device, non_blocking=True)

            pred = model(y_hist, u_hist, u_seq)
            loss = loss_multistep(
                pred_dict=pred,
                y_true=y_true,
                gamma_true=gamma_true,
                gamma_weights=gamma,
                m=m,
            )
            total_loss += loss.item()
    return total_loss / len(loader)


def train_single_run(
    *,
    args: argparse.Namespace,
    data: dict[str, object],
    device: torch.device,
    n_x: int,
    seed: int,
    checkpoint_dir: Path,
) -> list[EpochMetric]:
    set_seed(seed)
    cfg = build_cfg(args, n_x=n_x)
    train_loader, val_loader = build_loaders(data, cfg, args.batch_size)
    min_steps = max(cfg.n_a, cfg.n_b, cfg.m) + cfg.horizon + 1
    u_test, y_test = truncate_pair(data["u_test"], data["y_test"], data["test_fraction"], min_steps)

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

    checkpoint_path = checkpoint_dir / f"duffing_nx{n_x}_seed{seed}.pth"
    best_val = float("inf")
    history: list[EpochMetric] = []
    start_time = time.perf_counter()

    initial_test_eval = evaluate_chunked_test_sequence(
        model=model,
        u=u_test,
        y=y_test,
        cfg=cfg,
        device=device,
        y_scaler=data["y_scaler"],
    )
    initial_test_nrmse = float((initial_test_eval["nrmse_pct"].mean().item()) / 100.0)
    initial_val_loss = evaluate_loader_loss(model, val_loader, device, gamma, cfg.m)
    history.append(
        EpochMetric(
            seed=seed,
            n_x=n_x,
            epoch=0,
            elapsed_sec=0.0,
            train_loss=float("nan"),
            val_loss=float(initial_val_loss),
            test_nrmse=float(initial_test_nrmse),
        )
    )
    print(
        f"n_x={n_x:02d} seed={seed} epoch=000/{args.epochs} "
        f"| train=nan val={initial_val_loss:.7f} "
        f"| test_nrmse={initial_test_nrmse:.6f} | t=0.0s | lr={optimizer.param_groups[0]['lr']:.3e}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            y_hist = batch["y_hist"].to(device, non_blocking=True)
            u_hist = batch["u_hist"].to(device, non_blocking=True)
            u_seq = batch["u_seq"].to(device, non_blocking=True)
            y_true = batch["y_true"].to(device, non_blocking=True)
            gamma_true = batch["gamma_true"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(y_hist, u_hist, u_seq)
            loss = loss_multistep(
                pred_dict=pred,
                y_true=y_true,
                gamma_true=gamma_true,
                gamma_weights=gamma,
                m=cfg.m,
            )
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            train_loss_sum += loss.detach().item()

        train_loss = train_loss_sum / len(train_loader)
        val_loss = evaluate_loader_loss(model, val_loader, device, gamma, cfg.m)
        if epoch >= args.tail_start:
            scheduler.step(val_loss)

        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        test_eval = evaluate_chunked_test_sequence(
            model=model,
            u=u_test,
            y=y_test,
            cfg=cfg,
            device=device,
            y_scaler=data["y_scaler"],
        )
        test_nrmse = float((test_eval["nrmse_pct"].mean().item()) / 100.0)
        elapsed_sec = time.perf_counter() - start_time

        metric = EpochMetric(
            seed=seed,
            n_x=n_x,
            epoch=epoch,
            elapsed_sec=float(elapsed_sec),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            test_nrmse=test_nrmse,
        )
        history.append(metric)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"n_x={n_x:02d} seed={seed} epoch={epoch:03d}/{args.epochs} "
            f"| train={train_loss:.7f} val={val_loss:.7f} "
            f"| test_nrmse={test_nrmse:.6f} | t={elapsed_sec:.1f}s | lr={current_lr:.3e}"
        )

    return history


def save_metrics_csv(path: Path, rows: list[EpochMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "n_x", "epoch", "elapsed_sec", "train_loss", "val_loss", "test_nrmse"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row.seed,
                    "n_x": row.n_x,
                    "epoch": row.epoch,
                    "elapsed_sec": row.elapsed_sec,
                    "train_loss": row.train_loss,
                    "val_loss": row.val_loss,
                    "test_nrmse": row.test_nrmse,
                }
            )


def save_summary(path: Path, rows: list[EpochMetric], data: dict[str, object]) -> None:
    grouped: dict[int, list[EpochMetric]] = {}
    for row in rows:
        grouped.setdefault(row.n_x, []).append(row)

    lines = [
        "Duffing latent-dimension sensitivity study",
        "",
        f"Dataset path: {Path(data['dataset_path'])}",
        f"Raw shapes: {data['raw_shapes']}",
        "",
    ]
    for n_x in sorted(grouped):
        by_epoch = sorted(grouped[n_x], key=lambda item: (item.epoch, item.seed))
        last_epoch = max(item.epoch for item in by_epoch)
        finals = [item.test_nrmse for item in by_epoch if item.epoch == last_epoch]
        lines.append(
            f"n_x={n_x}: final test NRMSE mean={np.mean(finals):.6f} std={np.std(finals):.6f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plot_curves(
    rows: list[EpochMetric],
    nx_values: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    curves_by_nx: dict[int, dict[str, np.ndarray]] = {}
    for n_x in nx_values:
        subset = [row for row in rows if row.n_x == n_x]
        seeds = sorted({row.seed for row in subset})

        time_curves = []
        nrmse_curves = []
        for seed in seeds:
            seed_rows = sorted(
                (row for row in subset if row.seed == seed),
                key=lambda item: item.elapsed_sec,
            )
            time_curves.append(np.asarray([row.elapsed_sec for row in seed_rows], dtype=float))
            nrmse_curves.append(np.asarray([row.test_nrmse for row in seed_rows], dtype=float))

        max_common_time = max(times[-1] for times in time_curves)
        grid_size = max(300, max(len(times) for times in time_curves))
        time_grid = np.linspace(0.0, max_common_time, num=grid_size)

        interpolated = []
        for times, values in zip(time_curves, nrmse_curves):
            step_interp = interp1d(
                times,
                values,
                kind="previous",
                bounds_error=False,
                fill_value=(values[0], values[-1]),
            )
            interpolated.append(step_interp(time_grid))
        curves_np = np.asarray(interpolated, dtype=float)

        mean_curve = curves_np.mean(axis=0)
        std_curve = curves_np.std(axis=0)
        lower_curve = np.maximum(mean_curve - std_curve, 1e-12)
        upper_curve = mean_curve + std_curve

        curves_by_nx[n_x] = {
            "time_grid": time_grid,
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "lower_curve": lower_curve,
            "upper_curve": upper_curve,
        }

    return curves_by_nx


def save_plot_curves_txt(path: Path, curves_by_nx: dict[int, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for n_x in sorted(curves_by_nx):
        curves = curves_by_nx[n_x]
        lines.append(f"# n_x = {n_x}")
        lines.append("time_sec mean_nrmse std_nrmse lower_band upper_band")
        for t, mean, std, lower, upper in zip(
            curves["time_grid"],
            curves["mean_curve"],
            curves["std_curve"],
            curves["lower_curve"],
            curves["upper_curve"],
        ):
            lines.append(f"{t:.10f} {mean:.10f} {std:.10f} {lower:.10f} {upper:.10f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_nx_histories(
    path: Path,
    curves_by_nx: dict[int, dict[str, np.ndarray]],
    nx_values: list[int],
) -> None:
    fig, axes = plt.subplots(len(nx_values), 1, figsize=(7, 8), sharex=True, dpi=150)
    if len(nx_values) == 1:
        axes = [axes]
    cmap = matplotlib.colormaps["tab10"]

    for ax, n_x in zip(axes, nx_values):
        curves = curves_by_nx[n_x]
        time_grid = curves["time_grid"]
        mean_curve = curves["mean_curve"]
        lower_curve = curves["lower_curve"]
        upper_curve = curves["upper_curve"]
        i = nx_values.index(n_x)
        color_pos = i / max(len(nx_values) - 1, 1) / (10 / 3)
        color = cmap(color_pos)

        ax.fill_between(
            time_grid,
            lower_curve,
            upper_curve,
            color=color,
            alpha=0.2,
            label=None,
        )
        ax.semilogy(
            time_grid,
            mean_curve,
            "-",
            color=color,
            linewidth=1.2,
            label=rf"$n_x={n_x}$",
        )
        ax.set_xlim(left=0.0, right=float(time_grid[-1]))
        ax.set_ylim(1e-4, 1e1)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

        if i < len(nx_values) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Optimization time (seconds)")
            ax.tick_params(axis="x", labelsize=9)

        if i == len(nx_values) // 2:
            ax.set_ylabel("Test error (NRMSE)")

        xticks = ax.get_xticks()
        ax.set_xticks(np.unique(np.concatenate(([0.0], xticks))))

    plt.tight_layout(pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data = load_duffing_data(args.dataset_path, dtype=torch.float32)
    data["dataset_path"] = str(Path(args.dataset_path).resolve())
    data["train_fraction"] = args.train_fraction
    data["val_fraction"] = args.val_fraction
    data["test_fraction"] = args.test_fraction
    print(f"Using device: {device}")
    print(f"Sampling time Ts: {data['Ts']}")
    print(f"Dataset shapes found in MAT file: {data['raw_shapes']}")

    all_rows: list[EpochMetric] = []
    for n_x in args.nx_values:
        for seed in args.seeds:
            print(f"Starting run for n_x={n_x}, seed={seed}")
            all_rows.extend(
                train_single_run(
                    args=args,
                    data=data,
                    device=device,
                    n_x=n_x,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                )
            )

    save_metrics_csv(output_dir / "metrics.csv", all_rows)
    save_summary(output_dir / "summary.txt", all_rows, data)
    curves_by_nx = build_plot_curves(all_rows, args.nx_values)
    save_plot_curves_txt(output_dir / "duffing_nx_test_nrmse_curves.txt", curves_by_nx)
    plot_nx_histories(output_dir / "duffing_nx_test_nrmse.png", curves_by_nx, args.nx_values)

    print(f"Saved metrics to {output_dir / 'metrics.csv'}")
    print(f"Saved plot curves to {output_dir / 'duffing_nx_test_nrmse_curves.txt'}")
    print(f"Saved figure to {output_dir / 'duffing_nx_test_nrmse.png'}")
    print(f"Saved summary to {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
