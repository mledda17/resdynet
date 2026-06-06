from __future__ import annotations
import os

# These must come before torch import
import torch
from dataclasses import replace
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


def build_dataloaders(
    data: dict,
    cfg: ResDyNetConfig,
    batch_size: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DynamicalSystemDataset(data["u_train"], data["y_train"], cfg)
    val_ds = DynamicalSystemDataset(data["u_val"], data["y_val"], cfg)
    test_ds = DynamicalSystemDataset(data["u_test"], data["y_test"], cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader


def stage_checkpoint_path(base_checkpoint_path: str, stage_horizon: int, final_horizon: int) -> str:
    if stage_horizon == final_horizon:
        return base_checkpoint_path
    return base_checkpoint_path.replace(".pth", f"_H{stage_horizon}.pth")

def main() -> None:
    log_stage("Entering main()")
    torch.manual_seed(42)

    log_stage("Selecting device")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    use_cuda = device.type == "cuda"

    log_stage("Building configuration")
    cfg = ResDyNetConfig(
        n_u=1,
        n_y=1,
        n_x=14,
        n_a=50,
        n_b=50,
        m=0,                  # m=0 -> only current prediction
        horizon=80,
        encoder_hidden=[15],
        transition_hidden=15,
        transition_blocks=1,
        decoder_hidden=[15],
        activation="tanh",
    )

    batch_size = 256
    base_lr = 1e-3
    weight_decay = 0.0
    val_fraction = 0.2
    patience = 10000
    tail_start = 50
    checkpoint_path = "checkpoints/best_resdynet_WH_fresh_dup.pth"
    clip_grad_norm = 0.25
    gamma_decay = 0.98
    curriculum_horizons = [10, 20, 40, cfg.horizon]
    curriculum_epochs = [300, 300, 500, 900]
    curriculum_lrs = [1e-3, 5e-4, 2e-4, 3e-4]
    test_metric_every = 25
    resume_training = True
    resume_from_horizon = cfg.horizon
    resume_completed_epochs_before_checkpoint = 0
    resume_lr_override = 3e-5
    resume_remaining_epochs_override = 200
    resume_optimizer_state = False
    resume_scheduler_state = False
    resume_from_checkpoint = stage_checkpoint_path(
        checkpoint_path,
        resume_from_horizon,
        cfg.horizon,
    )
    if not (len(curriculum_horizons) == len(curriculum_epochs) == len(curriculum_lrs)):
        raise ValueError("curriculum_horizons, curriculum_epochs, and curriculum_lrs must align.")
    if resume_from_horizon not in curriculum_horizons:
        raise ValueError("resume_from_horizon must be one of curriculum_horizons.")

    log_stage("Preparing dataset")
    # data = prepare_cascaded_tanks_data(val_fraction=val_fraction, dtype=torch.float32)
    data   = prepare_hammerstein_data(val_fraction=val_fraction, dtype=torch.float32)

    log_stage("Creating model")
    model = AutoencoderResNetModel(cfg).to(device)
    print("Model parameter device:", next(model.parameters()).device, flush=True)

    log_stage("Creating optimizer and scheduler")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    log_stage("Starting curriculum training loop")

    #checkpoint = load_checkpoint_state(checkpoint_path, map_location=device)
    #model.load_state_dict(checkpoint["model_state_dict"])

    history = None
    resume_stage_idx = curriculum_horizons.index(resume_from_horizon) if resume_training else 0
    for stage_idx, (stage_horizon, stage_epochs, stage_lr) in enumerate(
        zip(curriculum_horizons, curriculum_epochs, curriculum_lrs),
        start=1,
    ):
        if stage_idx - 1 < resume_stage_idx:
            print(f"Skipping curriculum stage H={stage_horizon} because resume starts at H={resume_from_horizon}.", flush=True)
            continue

        for param_group in optimizer.param_groups:
            param_group["lr"] = stage_lr

        stage_cfg = replace(cfg, horizon=stage_horizon)
        train_loader, val_loader, test_loader = build_dataloaders(
            data=data,
            cfg=stage_cfg,
            batch_size=batch_size,
            pin_memory=use_cuda,
        )
        gamma = gamma_decay ** torch.arange(stage_horizon, dtype=torch.float32, device=device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=1e-6,
        )

        checkpoint_stage_path = stage_checkpoint_path(checkpoint_path, stage_horizon, cfg.horizon)

        if resume_training and stage_horizon == resume_from_horizon:
            checkpoint = load_checkpoint_state(resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if resume_optimizer_state and checkpoint["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if resume_scheduler_state and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if resume_lr_override is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = resume_lr_override

            resumed_epoch = int(checkpoint["epoch"] or 0)
            resumed_lr = optimizer.param_groups[0]["lr"]
            if resume_remaining_epochs_override is not None:
                stage_epochs = resume_remaining_epochs_override
            else:
                stage_epochs = max(
                    1,
                    stage_epochs - resume_completed_epochs_before_checkpoint - resumed_epoch,
                )
            print(
                f"Resuming H={stage_horizon} from {resume_from_checkpoint} "
                f"(completed before checkpoint={resume_completed_epochs_before_checkpoint}, "
                f"checkpoint epoch={resumed_epoch}, remaining epochs={stage_epochs}, "
                f"lr={resumed_lr:.3e}).",
                flush=True,
            )

        current_stage_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nCurriculum stage {stage_idx}/{len(curriculum_horizons)} "
            f"| H={stage_horizon} | epochs={stage_epochs} | lr={current_stage_lr:.1e}",
            flush=True,
        )
        print(f"Train samples: {len(train_loader.dataset)}", flush=True)
        print(f"Val samples:   {len(val_loader.dataset)}", flush=True)
        print(f"Test samples:  {len(test_loader.dataset)}", flush=True)

        def test_metric_fn(stage_cfg=stage_cfg) -> dict[str, float]:
            test_eval = evaluate_chunked_test_sequence(
                model=model,
                u=data["u_test"],
                y=data["y_test"],
                cfg=stage_cfg,
                device=device,
                y_scaler=data["y_scaler"],
            )
            return {"Test NRMSE [%]": float(test_eval["nrmse_pct"].mean().item())}

        history = train_model_multistep(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gamma=gamma,
            m=cfg.m,
            num_epochs=stage_epochs,
            patience=patience,
            checkpoint_path=checkpoint_stage_path,
            tail_start=tail_start,
            clip_grad_norm=clip_grad_norm,
            metric_fn=test_metric_fn,
            metric_every=test_metric_every,
            checkpoint_metric_name="Test NRMSE [%]",
            checkpoint_metric_mode="min",
            initialize_checkpoint_metric=resume_training and stage_horizon == resume_from_horizon,
        )

    if history is None:
        raise RuntimeError("Curriculum training did not run any stage.")

    _, _, test_loader = build_dataloaders(
        data=data,
        cfg=cfg,
        batch_size=batch_size,
        pin_memory=use_cuda,
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
