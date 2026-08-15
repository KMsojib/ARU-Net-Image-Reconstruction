"""
Training loop for ARU-Net.

Faithful port of the `train()` function in `4_AttentionResidualUNet.ipynb`,
with one software correction documented below.

ORIGINAL NOTEBOOK ISSUE:
The notebook's `__main__` block calls:

    train_losses = train(
        model, train_loader, val_loader, device,
        epochs=150, optimizer=optimizer, scheduler=scheduler,
        save_dir=custom_save_dir, patience=20
    )

but the `train()` function's signature in the notebook is:

    def train(model, train_loader, val_loader, device, epochs=70, lr=1e-4,
              optimizer=None, scheduler=None, save_dir="./Output"):

`patience` is not a parameter of `train()`, and no early-stopping logic
exists anywhere in the function body. As written, the notebook's own
`__main__` block would raise `TypeError: train() got an unexpected
keyword argument 'patience'` if executed top-to-bottom.

RECOMMENDED REPOSITORY CORRECTION:
This module adds a `patience` parameter and implements early stopping on
validation loss (matching the thesis's Sec. 3.3 claim that "early
stopping is applied to prevent overfitting"), so the training entry point
in `scripts/train.py` can actually run as documented.

Does this change the scientific methodology? No -- it makes the
early-stopping behavior the thesis already claims to use actually
executable; it does not alter the model, loss, optimizer, learning rate,
or any other reported hyperparameter.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

from .losses import CombinedLoss


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 150,
    lr: float = 1e-4,
    optimizer=None,
    scheduler=None,
    save_dir: str = "./results",
    patience: int = 20,
    scheduler_step_size: int = 15,
    scheduler_gamma: float = 0.5,
):
    """Train ARU-Net with the notebook's composite loss, logging per-epoch
    train/val loss and PSNR/SSIM to a CSV file, with early stopping on
    validation loss (see module docstring for why this differs from the
    as-written notebook).
    """
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "training_validation_metrics.csv")

    criterion = CombinedLoss(device=device, use_vgg=True)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_psnr", "train_ssim", "val_loss", "val_psnr", "val_ssim"]
        )

    model.train()
    train_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        skipped_batches = 0

        # ---------------- Training ----------------
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:
                dist_img, clean_img, _ = batch
            else:
                dist_img, clean_img = batch
            dist_img, clean_img = dist_img.to(device), clean_img.to(device)

            if dist_img.shape != clean_img.shape:
                skipped_batches += 1
                continue

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    output = model(dist_img)
                    if output.shape != clean_img.shape:
                        output = F.interpolate(
                            output, size=clean_img.shape[2:], mode="bilinear", align_corners=False
                        )
                    loss = criterion(output, clean_img, None)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(dist_img)
                if output.shape != clean_img.shape:
                    output = F.interpolate(
                        output, size=clean_img.shape[2:], mode="bilinear", align_corners=False
                    )
                loss = criterion(output, clean_img, None)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:4d}, Loss: {loss.item():.6f}")

        scheduler.step()
        denom = max(len(train_loader) - skipped_batches, 1)
        avg_train_loss = running_loss / denom
        train_losses.append(avg_train_loss)

        # ---------------- Validation ----------------
        model.eval()
        val_loss, val_psnr_list, val_ssim_list = 0.0, [], []

        with torch.no_grad():
            for val_batch in val_loader:
                if len(val_batch) == 3:
                    dist_img, clean_img, _ = val_batch
                else:
                    dist_img, clean_img = val_batch
                dist_img, clean_img = dist_img.to(device), clean_img.to(device)

                output = model(dist_img)
                if output.shape != clean_img.shape:
                    output = F.interpolate(
                        output, size=clean_img.shape[2:], mode="bilinear", align_corners=False
                    )
                loss = criterion(output, clean_img, None)
                val_loss += loss.item()

                output_np = output.detach().cpu().numpy().transpose(0, 2, 3, 1)
                target_np = clean_img.cpu().numpy().transpose(0, 2, 3, 1)

                for i in range(output_np.shape[0]):
                    pred = np.clip(output_np[i], 0, 1)
                    target = np.clip(target_np[i], 0, 1)
                    val_psnr_list.append(psnr_sk(target, pred, data_range=1.0))
                    val_ssim_list.append(ssim_sk(target, pred, channel_axis=-1, data_range=1.0))

        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else float("nan")
        avg_val_ssim = float(np.mean(val_ssim_list)) if val_ssim_list else float("nan")

        # Small-sample train PSNR/SSIM (one batch only, for speed --
        # matches the notebook's `break` after the first batch).
        train_psnr_list, train_ssim_list = [], []
        with torch.no_grad():
            for dist_img, clean_img, _ in train_loader:
                dist_img, clean_img = dist_img.to(device), clean_img.to(device)
                output = model(dist_img)
                output_np = output.detach().cpu().numpy().transpose(0, 2, 3, 1)
                target_np = clean_img.cpu().numpy().transpose(0, 2, 3, 1)
                for i in range(output_np.shape[0]):
                    train_psnr_list.append(psnr_sk(target_np[i], output_np[i], data_range=1.0))
                    train_ssim_list.append(
                        ssim_sk(target_np[i], output_np[i], channel_axis=-1, data_range=1.0)
                    )
                break
        avg_train_psnr = float(np.mean(train_psnr_list)) if train_psnr_list else float("nan")
        avg_train_ssim = float(np.mean(train_ssim_list)) if train_ssim_list else float("nan")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, avg_train_loss, avg_train_psnr, avg_train_ssim, avg_val_loss, avg_val_psnr, avg_val_ssim]
            )

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train PSNR: {avg_train_psnr:.2f} | Val PSNR: {avg_val_psnr:.2f}"
        )

        model.train()

        # ---------------- Early stopping (repository correction) ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping triggered at epoch {epoch + 1} "
                    f"(no val_loss improvement for {patience} epochs)."
                )
                break

    print(f"\nTraining complete. Metrics saved to {csv_path}")
    return train_losses
