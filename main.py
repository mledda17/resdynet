from __future__ import annotations
import os

# These must come before torch import
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import nonlinear_benchmarks

from src.autoencoder_resnet_model import AutoencoderResNetModel
from src.config import ResDyNetConfig
from src.dynamical_system_dataset import DynamicalSystemDataset, to_numpy_2d, to_torch_2d
from src.train_utils import (
    evaluate_chunked_test_sequence,
    load_checkpoint_state,
    plot_chunked_test_prediction,
    rollout_on_loader,
    train_model_multistep,
)


def split_train_val(u: torch.Tensor, y: torch.Tensor, val_fraction: float = 0.2):
    T = u.shape[0]
    split_idx = int((1.0 - val_fraction) * T)
    return u[:split_idx], y[:split_idx], u[split_idx:], y[split_idx:]


def prepare_cascaded_tanks_data(val_fraction: float = 0.2, dtype: torch.dtype = torch.float32):
    train_val, test = nonlinear_benchmarks.Cascaded_Tanks()
    print("state_initialization_window_length:", test.state_initialization_window_length)

    train_val_u, train_val_y = train_val
    test_u, test_y = test

    train_val_u = to_torch_2d(train_val_u, dtype=dtype)
    train_val_y = to_torch_2d(train_val_y, dtype=dtype)
    test_u = to_torch_2d(test_u, dtype=dtype)
    test_y = to_torch_2d(test_y, dtype=dtype)

    u_train, y_train, u_val, y_val = split_train_val(train_val_u, train_val_y, val_fraction=val_fraction)

    u_scaler = StandardScaler()
    y_scaler = StandardScaler()

    u_train_scaled = u_scaler.fit_transform(to_numpy_2d(u_train, "u_train"))
    y_train_scaled = y_scaler.fit_transform(to_numpy_2d(y_train, "y_train"))
    u_val_scaled = u_scaler.transform(to_numpy_2d(u_val, "u_val"))
    y_val_scaled = y_scaler.transform(to_numpy_2d(y_val, "y_val"))
    u_test_scaled = u_scaler.transform(to_numpy_2d(test_u, "u_test"))
    y_test_scaled = y_scaler.transform(to_numpy_2d(test_y, "y_test"))

    return {
        "u_train": to_torch_2d(u_train_scaled, dtype=dtype),
        "y_train": to_torch_2d(y_train_scaled, dtype=dtype),
        "u_val": to_torch_2d(u_val_scaled, dtype=dtype),
        "y_val": to_torch_2d(y_val_scaled, dtype=dtype),
        "u_test": to_torch_2d(u_test_scaled, dtype=dtype),
        "y_test": to_torch_2d(y_test_scaled, dtype=dtype),
        "u_scaler": u_scaler,
        "y_scaler": y_scaler,
    }

def prepare_hammerstein_data(val_fraction: float = 0.2, dtype: torch.dtype = torch.float32):
    train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
    print("state_initialization_window_length:", test.state_initialization_window_length)

    train_val_u, train_val_y = train_val
    test_u, test_y = test

    train_val_u = to_torch_2d(train_val_u, dtype=dtype)
    train_val_y = to_torch_2d(train_val_y, dtype=dtype)
    test_u = to_torch_2d(test_u, dtype=dtype)
    test_y = to_torch_2d(test_y, dtype=dtype)

    u_train, y_train, u_val, y_val = split_train_val(train_val_u, train_val_y, val_fraction=val_fraction)

    u_scaler = StandardScaler()
    y_scaler = StandardScaler()

    u_train_scaled = u_scaler.fit_transform(to_numpy_2d(u_train, "u_train"))
    y_train_scaled = y_scaler.fit_transform(to_numpy_2d(y_train, "y_train"))
    u_val_scaled = u_scaler.transform(to_numpy_2d(u_val, "u_val"))
    y_val_scaled = y_scaler.transform(to_numpy_2d(y_val, "y_val"))
    u_test_scaled = u_scaler.transform(to_numpy_2d(test_u, "u_test"))
    y_test_scaled = y_scaler.transform(to_numpy_2d(test_y, "y_test"))

    return {
        "u_train": to_torch_2d(u_train_scaled, dtype=dtype),
        "y_train": to_torch_2d(y_train_scaled, dtype=dtype),
        "u_val": to_torch_2d(u_val_scaled, dtype=dtype),
        "y_val": to_torch_2d(y_val_scaled, dtype=dtype),
        "u_test": to_torch_2d(u_test_scaled, dtype=dtype),
        "y_test": to_torch_2d(y_test_scaled, dtype=dtype),
        "u_scaler": u_scaler,
        "y_scaler": y_scaler,
    }


def log_stage(message: str) -> None:
    print(f"[startup] {message}", flush=True)


def select_device() -> torch.device:
    requested = os.environ.get("RESDYNET_DEVICE", "").strip().lower()
    valid_devices = {"", "auto", "cpu", "cuda"}
    if requested not in valid_devices:
        raise ValueError(
            "RESDYNET_DEVICE must be one of: auto, cpu, cuda. "
            f"Received: {requested!r}"
        )

    if requested in {"", "auto"}:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested via RESDYNET_DEVICE=cuda, "
                "but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Using device:", device, flush=True)
    if device.type == "cuda":
        print(
            "GPU info skipped during startup. "
            "Set RESDYNET_PRINT_GPU_INFO=1 if you want to query the device name.",
            flush=True,
        )
        if os.environ.get("RESDYNET_PRINT_GPU_INFO", "").strip() == "1":
            print("GPU:", torch.cuda.get_device_name(0), flush=True)
    return device

def main() -> None:
    log_stage("Entering main()")
    torch.manual_seed(0)

    log_stage("Selecting device")
    device = select_device()
    use_cuda = device.type == "cuda"

    log_stage("Building configuration")
    cfg = ResDyNetConfig(
        n_u=1,
        n_y=1,
        n_x=8,
        n_a=20,
        n_b=20,
        m=0,                  # m=0 -> only current prediction
        horizon=20,
        encoder_hidden=[256, 256],
        transition_hidden=256,
        transition_blocks=2,
        decoder_hidden=[256, 256],
        activation="tanh",
    )

    batch_size = 256
    num_epochs = 1500
    lr = 5e-5
    weight_decay = 0.0
    val_fraction = 0.4
    patience = 10000
    tail_start = 50
    checkpoint_path = "checkpoints/best_resdynet_wh4.pth"
    clip_grad_norm = 1.0

    gamma = torch.ones(cfg.horizon, dtype=torch.float32, device=device)

    log_stage("Preparing dataset")
    # data = prepare_cascaded_tanks_data(val_fraction=val_fraction, dtype=torch.float32)
    data   = prepare_hammerstein_data(val_fraction=val_fraction, dtype=torch.float32)

    log_stage("Creating torch datasets")
    train_ds = DynamicalSystemDataset(data["u_train"], data["y_train"], cfg)
    val_ds = DynamicalSystemDataset(data["u_val"], data["y_val"], cfg)
    test_ds = DynamicalSystemDataset(data["u_test"], data["y_test"], cfg)

    log_stage("Creating data loaders")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    print(f"Train samples: {len(train_ds)}", flush=True)
    print(f"Val samples:   {len(val_ds)}", flush=True)
    print(f"Test samples:  {len(test_ds)}", flush=True)

    log_stage("Creating model")
    model = AutoencoderResNetModel(cfg).to(device)
    print("Model parameter device:", next(model.parameters()).device, flush=True)

    log_stage("Creating optimizer and scheduler")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=50,
        min_lr=1e-8,
    )

    log_stage("Loading checkpoint")
    checkpoint = load_checkpoint_state(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", checkpoint_path, flush=True)
    print("Checkpoint epoch:", checkpoint["epoch"], flush=True)
    print("Checkpoint best val loss:", checkpoint["best_val_loss"], flush=True)

    log_stage("Starting training loop")

    history = train_model_multistep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gamma=gamma,
        m=cfg.m,
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        tail_start=tail_start,
        clip_grad_norm=clip_grad_norm,
    )

    print("\nTraining finished.", flush=True)
    print("Stop epoch:", history["stop_epoch"], flush=True)
    print("Best epoch:", history["best_epoch"], flush=True)
    print("Best val loss:", history["best_val_loss"], flush=True)

    log_stage("Running test rollout")
    test_rollout = rollout_on_loader(
        model=model,
        loader=test_loader,
        device=device,
    )
    test_rmse_norm = torch.sqrt(
        torch.mean((test_rollout["Y_hat_all"] - test_rollout["Y_true_all"]).pow(2))
    )
    print("\nTest rollout-window metric:", flush=True)
    print("Normalized RMSE on rollout windows:", float(test_rmse_norm), flush=True)

    log_stage("Running chunked test evaluation")
    test_eval_chunked = evaluate_chunked_test_sequence(
        model=model,
        u=data["u_test"],
        y=data["y_test"],
        cfg=cfg,
        device=device,
        y_scaler=data["y_scaler"],
    )

    print("\nFinal chunked test metrics:", flush=True)
    print("RMSE [volt]:", test_eval_chunked["rmse"].numpy(), flush=True)
    print("NRMSE [%]:  ", test_eval_chunked["nrmse_pct"].numpy(), flush=True)

    log_stage("Plotting predictions")
    plot_chunked_test_prediction(
        y_true=test_eval_chunked["y_true"],
        y_pred=test_eval_chunked["y_pred"],
        save_path="outputs/chunked_test_prediction.png",
        title="ResDyNet - chunked test prediction",
    )


if __name__ == "__main__":
    main()
