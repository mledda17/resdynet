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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.autoencoder_resnet_model import AutoencoderResNetModel
from src.config import ResDyNetConfig
from src.dynamical_system_dataset import DynamicalSystemDataset, to_numpy_2d, to_torch_2d
from src.train_utils import evaluate_chunked_test_sequence, loss_multistep


@dataclass
class StageMetric:
    residual_blocks: int
    horizon: int
    epoch: int
    elapsed_sec: float
    train_loss: float
    val_loss: float
    test_nrmse_pct: float
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep residual blocks L and horizon H on the Duffing benchmark. "
            "By default each (L, H) pair is trained independently to keep the grid comparison clean."
        )
    )
    parser.add_argument("--dataset-path", default="dataset/dataset_duffing.mat")
    parser.add_argument("--output-dir", default="outputs/duffing_lh_sensitivity")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    parser.add_argument("--l-values", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--h-values", type=int, nargs="+", default=[1, 10, 20, 40])

    parser.add_argument("--n-x", type=int, default=8)
    parser.add_argument("--n-a", type=int, default=10)
    parser.add_argument("--n-b", type=int, default=10)
    parser.add_argument("--m", type=int, default=0)

    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--transition-hidden", type=int, default=256)
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="tanh")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--tail-start", type=int, default=20)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warm-start-across-h",
        action="store_true",
        help=(
            "Reuse the model trained at the previous horizon for the next horizon with the same L. "
            "Disabled by default so each heatmap cell is an independent run."
        ),
    )
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

    return {
        "u_train": to_torch_2d(u_train_scaled, dtype=dtype),
        "y_train": to_torch_2d(y_train_scaled, dtype=dtype),
        "u_val": to_torch_2d(u_val_scaled, dtype=dtype),
        "y_val": to_torch_2d(y_val_scaled, dtype=dtype),
        "u_test": to_torch_2d(u_test_scaled, dtype=dtype),
        "y_test": to_torch_2d(y_test_scaled, dtype=dtype),
        "u_scaler": u_scaler,
        "y_scaler": y_scaler,
        "raw_shapes": {
            "u_train": tuple(raw["u_train"].shape),
            "y_train": tuple(raw["y_train"].shape),
            "u_val": tuple(raw["u_val"].shape),
            "y_val": tuple(raw["y_val"].shape),
            "u_test": tuple(raw["u_test"].shape),
            "y_test": tuple(raw["y_test"].shape),
        },
    }


def build_cfg(args: argparse.Namespace, residual_blocks: int, horizon: int) -> ResDyNetConfig:
    return ResDyNetConfig(
        n_u=1,
        n_y=1,
        n_x=args.n_x,
        n_a=args.n_a,
        n_b=args.n_b,
        m=args.m,
        horizon=horizon,
        encoder_hidden=args.encoder_hidden,
        transition_hidden=args.transition_hidden,
        transition_blocks=residual_blocks,
        decoder_hidden=args.decoder_hidden,
        activation=args.activation,
    )


def build_loaders(
    data: dict[str, object],
    cfg: ResDyNetConfig,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = DynamicalSystemDataset(data["u_train"], data["y_train"], cfg)
    val_ds = DynamicalSystemDataset(data["u_val"], data["y_val"], cfg)
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


def train_stage(
    *,
    args: argparse.Namespace,
    data: dict[str, object],
    device: torch.device,
    residual_blocks: int,
    horizon: int,
    checkpoint_dir: Path,
    model: AutoencoderResNetModel | None = None,
) -> tuple[list[StageMetric], AutoencoderResNetModel, Path]:
    cfg = build_cfg(args, residual_blocks=residual_blocks, horizon=horizon)
    train_loader, val_loader = build_loaders(data, cfg, args.batch_size)

    if model is None:
        model = AutoencoderResNetModel(cfg).to(device)
    else:
        model.cfg = cfg
        model.to(device)

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

    checkpoint_path = checkpoint_dir / f"duffing_L{residual_blocks}_H{horizon}.pth"
    best_val = float("inf")
    stage_history: list[StageMetric] = []
    start_time = time.perf_counter()

    initial_val_loss = evaluate_loader_loss(model, val_loader, device, gamma, cfg.m)
    initial_test_eval = evaluate_chunked_test_sequence(
        model=model,
        u=data["u_test"],
        y=data["y_test"],
        cfg=cfg,
        device=device,
        y_scaler=data["y_scaler"],
    )
    initial_test_nrmse_pct = float(initial_test_eval["nrmse_pct"].mean().item())
    stage_history.append(
        StageMetric(
            residual_blocks=residual_blocks,
            horizon=horizon,
            epoch=0,
            elapsed_sec=0.0,
            train_loss=float("nan"),
            val_loss=float(initial_val_loss),
            test_nrmse_pct=initial_test_nrmse_pct,
            checkpoint_path=str(checkpoint_path),
        )
    )
    print(
        f"L={residual_blocks} H={horizon} epoch=000/{args.epochs} "
        f"| train=nan val={initial_val_loss:.7f} | test_nrmse={initial_test_nrmse_pct:.4f}% "
        f"| t=0.0s | lr={optimizer.param_groups[0]['lr']:.3e}",
        flush=True,
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
            u=data["u_test"],
            y=data["y_test"],
            cfg=cfg,
            device=device,
            y_scaler=data["y_scaler"],
        )
        test_nrmse_pct = float(test_eval["nrmse_pct"].mean().item())
        elapsed_sec = time.perf_counter() - start_time

        stage_history.append(
            StageMetric(
                residual_blocks=residual_blocks,
                horizon=horizon,
                epoch=epoch,
                elapsed_sec=float(elapsed_sec),
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                test_nrmse_pct=test_nrmse_pct,
                checkpoint_path=str(checkpoint_path),
            )
        )
        print(
            f"L={residual_blocks} H={horizon} epoch={epoch:03d}/{args.epochs} "
            f"| train={train_loss:.7f} val={val_loss:.7f} | test_nrmse={test_nrmse_pct:.4f}% "
            f"| t={elapsed_sec:.1f}s | lr={optimizer.param_groups[0]['lr']:.3e}",
            flush=True,
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.cfg = cfg
    return stage_history, model, checkpoint_path


def save_metrics_csv(path: Path, rows: list[StageMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "residual_blocks",
                "horizon",
                "epoch",
                "elapsed_sec",
                "train_loss",
                "val_loss",
                "test_nrmse_pct",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "residual_blocks": row.residual_blocks,
                    "horizon": row.horizon,
                    "epoch": row.epoch,
                    "elapsed_sec": row.elapsed_sec,
                    "train_loss": row.train_loss,
                    "val_loss": row.val_loss,
                    "test_nrmse_pct": row.test_nrmse_pct,
                    "checkpoint_path": row.checkpoint_path,
                }
            )


def save_stage_summary(path: Path, final_rows: list[StageMetric], args: argparse.Namespace, data: dict[str, object]) -> None:
    lines = [
        "Duffing L-H sensitivity study",
        "",
        f"Dataset path: {Path(args.dataset_path)}",
        f"Raw shapes: {data['raw_shapes']}",
        f"n_x={args.n_x}, n_a={args.n_a}, n_b={args.n_b}, m={args.m}",
        f"encoder_hidden={args.encoder_hidden}, decoder_hidden={args.decoder_hidden}",
        f"transition_hidden={args.transition_hidden}, activation={args.activation}",
        f"L values={args.l_values}",
        f"H values={args.h_values}",
        f"warm_start_across_h={args.warm_start_across_h}",
        "",
        "Final stage results:",
    ]
    for row in sorted(final_rows, key=lambda item: (item.residual_blocks, item.horizon)):
        lines.append(
            f"L={row.residual_blocks}, H={row.horizon}: test NRMSE={row.test_nrmse_pct:.4f}% "
            f"(checkpoint={row.checkpoint_path})"
        )

    lines.extend(
        [
            "",
            "Per-horizon trend across L:",
        ]
    )
    for horizon in args.h_values:
        horizon_rows = sorted(
            [row for row in final_rows if row.horizon == horizon],
            key=lambda item: item.residual_blocks,
        )
        trend = ", ".join(
            f"L={row.residual_blocks}: {row.test_nrmse_pct:.4f}%"
            for row in horizon_rows
        )
        lines.append(f"H={horizon} -> {trend}")

    lines.extend(
        [
            "",
            "Per-block trend across H:",
        ]
    )
    for residual_blocks in args.l_values:
        block_rows = sorted(
            [row for row in final_rows if row.residual_blocks == residual_blocks],
            key=lambda item: item.horizon,
        )
        trend = ", ".join(
            f"H={row.horizon}: {row.test_nrmse_pct:.4f}%"
            for row in block_rows
        )
        lines.append(f"L={residual_blocks} -> {trend}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(path: Path, final_rows: list[StageMetric], l_values: list[int], h_values: list[int]) -> None:
    z = np.full((len(h_values), len(l_values)), np.nan, dtype=float)
    for row in final_rows:
        j = h_values.index(row.horizon)
        i = l_values.index(row.residual_blocks)
        z[j, i] = row.test_nrmse_pct

    x, y = np.meshgrid(np.asarray(l_values, dtype=float), np.asarray(h_values, dtype=float))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    contour = ax.contourf(x, y, z, levels=12, cmap="viridis")
    ax.scatter(x.flatten(), y.flatten(), c=z.flatten(), cmap="viridis", edgecolors="white", s=90)

    for row in final_rows:
        ax.text(
            row.residual_blocks,
            row.horizon,
            f"{row.test_nrmse_pct:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Residual blocks L")
    ax.set_ylabel("Horizon H")
    ax.set_title("Duffing test NRMSE [%] over L and H")
    ax.set_xticks(l_values)
    ax.set_yticks(h_values)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Test NRMSE [%]")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data = load_duffing_data(args.dataset_path)

    all_rows: list[StageMetric] = []
    final_rows: list[StageMetric] = []

    for residual_blocks in args.l_values:
        print(f"Starting sweep for L={residual_blocks}", flush=True)
        model: AutoencoderResNetModel | None = None

        for horizon in args.h_values:
            print(f"  Training stage with L={residual_blocks}, H={horizon}", flush=True)
            if not args.warm_start_across_h:
                model = None
            stage_rows, model, checkpoint_path = train_stage(
                args=args,
                data=data,
                device=device,
                residual_blocks=residual_blocks,
                horizon=horizon,
                checkpoint_dir=checkpoint_dir,
                model=model,
            )
            all_rows.extend(stage_rows)

            final_eval = evaluate_chunked_test_sequence(
                model=model,
                u=data["u_test"],
                y=data["y_test"],
                cfg=model.cfg,
                device=device,
                y_scaler=data["y_scaler"],
            )
            final_rows.append(
                StageMetric(
                    residual_blocks=residual_blocks,
                    horizon=horizon,
                    epoch=args.epochs,
                    elapsed_sec=stage_rows[-1].elapsed_sec,
                    train_loss=stage_rows[-1].train_loss,
                    val_loss=stage_rows[-1].val_loss,
                    test_nrmse_pct=float(final_eval["nrmse_pct"].mean().item()),
                    checkpoint_path=str(checkpoint_path),
                )
            )

    save_metrics_csv(output_dir / "metrics.csv", all_rows)
    save_metrics_csv(output_dir / "final_grid.csv", final_rows)
    save_stage_summary(output_dir / "summary.txt", final_rows, args, data)
    plot_heatmap(output_dir / "duffing_lh_test_nrmse_heatmap.png", final_rows, args.l_values, args.h_values)

    print(f"Saved metrics to {output_dir / 'metrics.csv'}", flush=True)
    print(f"Saved final grid to {output_dir / 'final_grid.csv'}", flush=True)
    print(f"Saved summary to {output_dir / 'summary.txt'}", flush=True)
    print(f"Saved heatmap to {output_dir / 'duffing_lh_test_nrmse_heatmap.png'}", flush=True)


if __name__ == "__main__":
    main()
