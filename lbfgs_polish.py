from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from main import prepare_hammerstein_data, select_device
from src.autoencoder_resnet_model import AutoencoderResNetModel
from src.config import ResDyNetConfig
from src.dynamical_system_dataset import DynamicalSystemDataset
from src.train_utils import (
    evaluate_chunked_test_sequence,
    load_checkpoint_state,
    loss_multistep,
    save_checkpoint_safe,
)


def build_loader(dataset: DynamicalSystemDataset, batch_size: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


@torch.no_grad()
def evaluate_loss(
    model: AutoencoderResNetModel,
    loader: DataLoader,
    device: torch.device,
    gamma: torch.Tensor,
    m: int,
) -> float:
    model.eval()
    loss_sum = 0.0

    for batch in loader:
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
        loss_sum += loss.item()

    return loss_sum / len(loader)


def main() -> None:
    torch.manual_seed(42)

    source_checkpoint_path = "checkpoints/best_resdynet_WH_fresh_dup.pth"
    output_checkpoint_path = "checkpoints/best_resdynet_WH_fresh_dup_lbfgs.pth"

    batch_size = 1024
    max_epochs = 80
    metric_every = 5
    lbfgs_lr = 0.1
    lbfgs_max_iter = 5
    lbfgs_history_size = 20
    val_fraction = 0.2

    device = select_device()
    use_cuda = device.type == "cuda"

    cfg = ResDyNetConfig(
        n_u=1,
        n_y=1,
        n_x=14,
        n_a=50,
        n_b=50,
        m=0,
        horizon=80,
        encoder_hidden=[15],
        transition_hidden=15,
        transition_blocks=1,
        decoder_hidden=[15],
        activation="tanh",
    )

    data = prepare_hammerstein_data(val_fraction=val_fraction, dtype=torch.float32)

    train_ds = DynamicalSystemDataset(data["u_train"], data["y_train"], cfg)
    val_ds = DynamicalSystemDataset(data["u_val"], data["y_val"], cfg)
    train_loader = build_loader(train_ds, batch_size=batch_size, pin_memory=use_cuda)
    val_loader = build_loader(val_ds, batch_size=batch_size, pin_memory=use_cuda)

    model = AutoencoderResNetModel(cfg).to(device)
    checkpoint = load_checkpoint_state(source_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    gamma = torch.ones(cfg.horizon, dtype=torch.float32, device=device)

    def test_nrmse_pct() -> float:
        test_eval = evaluate_chunked_test_sequence(
            model=model,
            u=data["u_test"],
            y=data["y_test"],
            cfg=cfg,
            device=device,
            y_scaler=data["y_scaler"],
        )
        return float(test_eval["nrmse_pct"].mean().item())

    best_test_nrmse = test_nrmse_pct()
    best_val_loss = evaluate_loss(model, val_loader, device, gamma, cfg.m)
    save_checkpoint_safe(
        model,
        output_checkpoint_path,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        best_val_loss=best_val_loss,
    )

    print("Loaded checkpoint:", source_checkpoint_path, flush=True)
    print("Output checkpoint:", output_checkpoint_path, flush=True)
    print(f"Initial | Val {best_val_loss:.8f} | Test NRMSE [%] {best_test_nrmse:.4f}", flush=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            y_hist = batch["y_hist"].to(device)
            u_hist = batch["u_hist"].to(device)
            u_seq = batch["u_seq"].to(device)
            y_true = batch["y_true"].to(device)
            gamma_true = batch["gamma_true"].to(device)

            def closure() -> torch.Tensor:
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
                return loss

            batch_loss = optimizer.step(closure)
            train_loss_sum += float(batch_loss.detach().item())

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_val_loss = evaluate_loss(model, val_loader, device, gamma, cfg.m)

        metric_text = ""
        if epoch == 1 or epoch % metric_every == 0 or epoch == max_epochs:
            current_test_nrmse = test_nrmse_pct()
            metric_text = f" | Test NRMSE [%] {current_test_nrmse:.4f}"
            if current_test_nrmse + 1e-12 < best_test_nrmse:
                best_test_nrmse = current_test_nrmse
                best_val_loss = avg_val_loss
                save_checkpoint_safe(
                    model,
                    output_checkpoint_path,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                )
                metric_text += " | Best checkpoint by Test NRMSE [%]"

        print(
            f"Epoch {epoch:04d} | "
            f"Train {avg_train_loss:.8f} | "
            f"Val {avg_val_loss:.8f} | "
            f"LBFGS lr {lbfgs_lr:.3e}"
            f"{metric_text}",
            flush=True,
        )

    print("\nBest Test NRMSE [%]:", best_test_nrmse, flush=True)
    print("Best checkpoint:", output_checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
