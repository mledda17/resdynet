from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from main import prepare_cascaded_tanks_data, prepare_hammerstein_data
from src.autoencoder_resnet_model import AutoencoderResNetModel
from src.config import ResDyNetConfig
from src.dynamical_system_dataset import DynamicalSystemDataset
from src.train_utils import evaluate_chunked_test_sequence, train_model_multistep


@dataclass
class RunResult:
    phase: str
    dataset_fraction: float
    epochs: int
    patience: int
    n_a: int
    n_b: int
    n_x: int
    transition_blocks: int
    best_val_loss: float
    stop_epoch: int
    test_rmse: float
    test_nrmse_pct: float
    duration_sec: float
    checkpoint_path: str


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_dataset(name: str, val_fraction: float, dtype: torch.dtype) -> dict[str, object]:
    if name == "hammerstein":
        return prepare_hammerstein_data(val_fraction=val_fraction, dtype=dtype)
    if name == "cascaded_tanks":
        return prepare_cascaded_tanks_data(val_fraction=val_fraction, dtype=dtype)
    raise ValueError(f"Unsupported dataset: {name}")


def truncate_tensor_pair(
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


def build_loaders(
    data: dict[str, object],
    cfg: ResDyNetConfig,
    batch_size: int,
    dataset_fraction: float,
) -> tuple[DataLoader, DataLoader]:
    min_steps = max(cfg.n_a, cfg.n_b, cfg.m) + cfg.horizon + 1

    u_train, y_train = truncate_tensor_pair(
        data["u_train"],
        data["y_train"],
        dataset_fraction,
        min_steps,
    )
    u_val, y_val = truncate_tensor_pair(
        data["u_val"],
        data["y_val"],
        dataset_fraction,
        min_steps,
    )

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


def make_model_config(
    args: argparse.Namespace,
    n_a: int,
    n_b: int,
    n_x: int,
    transition_blocks: int,
) -> ResDyNetConfig:
    return ResDyNetConfig(
        n_u=1,
        n_y=1,
        n_x=n_x,
        n_a=n_a,
        n_b=n_b,
        m=args.m,
        horizon=args.horizon,
        encoder_hidden=args.encoder_hidden,
        transition_hidden=args.transition_hidden,
        transition_blocks=transition_blocks,
        decoder_hidden=args.decoder_hidden,
        activation=args.activation,
    )


def evaluate_candidate(
    *,
    phase: str,
    args: argparse.Namespace,
    data: dict[str, object],
    device: torch.device,
    dataset_fraction: float,
    epochs: int,
    patience: int,
    n_a: int,
    n_b: int,
    n_x: int,
    transition_blocks: int,
    checkpoint_dir: Path,
) -> RunResult:
    cfg = make_model_config(args, n_a=n_a, n_b=n_b, n_x=n_x, transition_blocks=transition_blocks)
    train_loader, val_loader = build_loaders(data, cfg, args.batch_size, dataset_fraction)

    model = AutoencoderResNetModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    gamma = torch.ones(cfg.horizon, dtype=torch.float32)
    checkpoint_path = checkpoint_dir / (
        f"{phase}_na{n_a}_nb{n_b}_nx{n_x}_tb{transition_blocks}.pth"
    )

    print(
        f"\n[{phase}] fraction={dataset_fraction:.2f} epochs={epochs} "
        f"n_a={n_a} n_b={n_b} n_x={n_x} blocks={transition_blocks}"
    )

    start_time = time.perf_counter()
    history = train_model_multistep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gamma=gamma,
        m=cfg.m,
        num_epochs=epochs,
        patience=patience,
        checkpoint_path=str(checkpoint_path),
        tail_start=args.tail_start,
        clip_grad_norm=args.clip_grad_norm,
    )
    duration_sec = time.perf_counter() - start_time

    best_val_loss = min(history["val_loss"])

    test_eval = evaluate_chunked_test_sequence(
        model=model,
        u=data["u_test"],
        y=data["y_test"],
        cfg=cfg,
        device=device,
        y_scaler=data["y_scaler"],
    )

    return RunResult(
        phase=phase,
        dataset_fraction=dataset_fraction,
        epochs=epochs,
        patience=patience,
        n_a=n_a,
        n_b=n_b,
        n_x=n_x,
        transition_blocks=transition_blocks,
        best_val_loss=float(best_val_loss),
        stop_epoch=int(history["stop_epoch"]),
        test_rmse=float(test_eval["rmse"].mean().item()),
        test_nrmse_pct=float(test_eval["nrmse_pct"].mean().item()),
        duration_sec=duration_sec,
        checkpoint_path=str(checkpoint_path),
    )


def save_results(path: Path, results: Iterable[RunResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_top_results(title: str, results: list[RunResult], top_k: int) -> None:
    print(f"\n{title}")
    ranked = sorted(results, key=lambda item: item.best_val_loss)
    for rank, result in enumerate(ranked[:top_k], start=1):
        print(
            f"{rank:02d}. val={result.best_val_loss:.8f} "
            f"test_nrmse={result.test_nrmse_pct:.4f}% "
            f"n_a={result.n_a} n_b={result.n_b} "
            f"n_x={result.n_x} blocks={result.transition_blocks} "
            f"epochs={result.stop_epoch}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Progressive hyperparameter search for ResDyNet."
    )
    parser.add_argument("--dataset", choices=["hammerstein", "cascaded_tanks"], default="hammerstein")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="tuning_runs")

    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--tail-start", type=int, default=20)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)

    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="tanh")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--transition-hidden", type=int, default=256)

    parser.add_argument("--na-candidates", default="5,10,20")
    parser.add_argument("--nb-candidates", default="5,10,20")
    parser.add_argument("--nx-candidates", default="8,16,32")
    parser.add_argument("--transition-block-candidates", default="1,2,4")

    parser.add_argument("--memory-fraction", type=float, default=0.10)
    parser.add_argument("--memory-epochs", type=int, default=40)
    parser.add_argument("--memory-patience", type=int, default=12)
    parser.add_argument("--memory-top-k", type=int, default=2)

    parser.add_argument("--state-fraction", type=float, default=0.20)
    parser.add_argument("--state-epochs", type=int, default=60)
    parser.add_argument("--state-patience", type=int, default=15)
    parser.add_argument("--state-top-k", type=int, default=2)

    parser.add_argument("--transition-fraction", type=float, default=0.25)
    parser.add_argument("--transition-epochs", type=int, default=80)
    parser.add_argument("--transition-patience", type=int, default=20)
    parser.add_argument("--transition-top-k", type=int, default=2)

    parser.add_argument("--final-fraction", type=float, default=1.0)
    parser.add_argument("--final-epochs", type=int, default=150)
    parser.add_argument("--final-patience", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    data = load_dataset(args.dataset, val_fraction=args.val_fraction, dtype=torch.float32)
    print(f"Device: {device}")
    print(f"Output dir: {output_dir.resolve()}")

    na_candidates = parse_int_list(args.na_candidates)
    nb_candidates = parse_int_list(args.nb_candidates)
    nx_candidates = parse_int_list(args.nx_candidates)
    transition_block_candidates = parse_int_list(args.transition_block_candidates)

    all_results: list[RunResult] = []

    memory_results: list[RunResult] = []
    for n_a, n_b in itertools.product(na_candidates, nb_candidates):
        result = evaluate_candidate(
            phase="memory",
            args=args,
            data=data,
            device=device,
            dataset_fraction=args.memory_fraction,
            epochs=args.memory_epochs,
            patience=args.memory_patience,
            n_a=n_a,
            n_b=n_b,
            n_x=nx_candidates[0],
            transition_blocks=transition_block_candidates[0],
            checkpoint_dir=checkpoint_dir,
        )
        memory_results.append(result)
        all_results.append(result)

    print_top_results("Top memory candidates", memory_results, args.memory_top_k)
    top_memory = sorted(memory_results, key=lambda item: item.best_val_loss)[: args.memory_top_k]

    state_results: list[RunResult] = []
    for base in top_memory:
        for n_x in nx_candidates:
            result = evaluate_candidate(
                phase="state",
                args=args,
                data=data,
                device=device,
                dataset_fraction=args.state_fraction,
                epochs=args.state_epochs,
                patience=args.state_patience,
                n_a=base.n_a,
                n_b=base.n_b,
                n_x=n_x,
                transition_blocks=base.transition_blocks,
                checkpoint_dir=checkpoint_dir,
            )
            state_results.append(result)
            all_results.append(result)

    print_top_results("Top state candidates", state_results, args.state_top_k)
    top_state = sorted(state_results, key=lambda item: item.best_val_loss)[: args.state_top_k]

    transition_results: list[RunResult] = []
    for base in top_state:
        for transition_blocks in transition_block_candidates:
            result = evaluate_candidate(
                phase="transition",
                args=args,
                data=data,
                device=device,
                dataset_fraction=args.transition_fraction,
                epochs=args.transition_epochs,
                patience=args.transition_patience,
                n_a=base.n_a,
                n_b=base.n_b,
                n_x=base.n_x,
                transition_blocks=transition_blocks,
                checkpoint_dir=checkpoint_dir,
            )
            transition_results.append(result)
            all_results.append(result)

    print_top_results("Top transition candidates", transition_results, args.transition_top_k)
    finalists = sorted(transition_results, key=lambda item: item.best_val_loss)[: args.transition_top_k]

    final_results: list[RunResult] = []
    for base in finalists:
        result = evaluate_candidate(
            phase="final",
            args=args,
            data=data,
            device=device,
            dataset_fraction=args.final_fraction,
            epochs=args.final_epochs,
            patience=args.final_patience,
            n_a=base.n_a,
            n_b=base.n_b,
            n_x=base.n_x,
            transition_blocks=base.transition_blocks,
            checkpoint_dir=checkpoint_dir,
        )
        final_results.append(result)
        all_results.append(result)

    print_top_results("Final ranking", final_results, top_k=len(final_results))
    best_result = min(final_results, key=lambda item: item.best_val_loss)

    save_results(output_dir / "results.csv", all_results)
    summary_path = output_dir / "best_config.json"
    summary_path.write_text(json.dumps(asdict(best_result), indent=2), encoding="utf-8")

    print("\nBest configuration selected by validation loss:")
    print(json.dumps(asdict(best_result), indent=2))
    print(f"\nSaved all results to {output_dir / 'results.csv'}")
    print(f"Saved best config to {summary_path}")


if __name__ == "__main__":
    main()
