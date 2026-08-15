"""
Evaluation and visualization utilities for ARU-Net.

Faithful port of `evaluate()`, `visualize_training_progress()`,
`visualize_sample_results()`, `save_training_results()`, and
`save_metrics_csv()` from `4_AttentionResidualUNet.ipynb`.
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image

from .metrics import global_metrics, leaf_masked_metrics
from .utils import compute_leaf_mask


def evaluate(model, dataloader, device, save_dir: str = "./results"):
    """Full evaluation pass: computes leaf-masked PSNR/SSIM per image
    (falling back to global metrics for small/degenerate masks), saves
    reconstructed images, and writes a metrics CSV.
    """
    print("\nStarting evaluation (leaf-masked SSIM/PSNR per image)...")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "eval_metrics.csv")

    model.eval()
    psnr_scores, ssim_scores = [], []
    csv_rows = []

    with torch.no_grad():
        idx_counter = 0
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                dist_img, clean_img, paths = batch
            else:
                dist_img, clean_img = batch
                paths = [f"sample_{i}" for i in range(dist_img.shape[0])]

            dist_img, clean_img = dist_img.to(device), clean_img.to(device)

            if "cuda" in device:
                with torch.amp.autocast("cuda"):
                    output = model(dist_img)
            else:
                output = model(dist_img)

            if output.shape != clean_img.shape:
                output = torch.nn.functional.interpolate(
                    output, size=clean_img.shape[2:], mode="bilinear", align_corners=False
                )
            output = torch.clamp(output, 0, 1)

            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            clean_np = clean_img.cpu().numpy().transpose(0, 2, 3, 1)
            mask_tensor = compute_leaf_mask(clean_img)
            mask_np = mask_tensor.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(output_np.shape[0]):
                pred = np.clip(output_np[i], 0, 1)
                target = np.clip(clean_np[i], 0, 1)
                mask_i = (mask_np[i, :, :, 0] > 0.5).astype(np.float32)

                psnr_val, ssim_val = leaf_masked_metrics(target, pred, mask_i)

                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
                csv_rows.append([idx_counter, psnr_val, ssim_val])

                recon_name = f"Image_{idx_counter}_Reconstructed.jpg"
                save_path = os.path.join(save_dir, recon_name)
                save_image(torch.from_numpy(pred.transpose(2, 0, 1)), save_path)

                idx_counter += 1

            if batch_idx % 5 == 0:
                print(f"  Processed batch {batch_idx}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "PSNR", "SSIM"])
        writer.writerows(csv_rows)

    if not psnr_scores:
        print("No valid samples evaluated.")
        return 0.0, 0.0, [], []

    print("\nEvaluation results:")
    print(f"  PSNR (mean +/- std): {np.mean(psnr_scores):.4f} +/- {np.std(psnr_scores):.4f}")
    print(f"  SSIM (mean +/- std): {np.mean(ssim_scores):.4f} +/- {np.std(ssim_scores):.4f}")
    print(f"  Saved {len(psnr_scores)} reconstructed images and metrics to: {csv_path}")

    return np.mean(psnr_scores), np.mean(ssim_scores), psnr_scores, ssim_scores


def visualize_training_progress(train_losses):
    if not train_losses:
        print("No training losses to visualize.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, linewidth=2)
    plt.title("Training Loss Progress", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()
    print(f"Initial loss: {train_losses[0]:.6f}, Final loss: {train_losses[-1]:.6f}")


def visualize_sample_results(model, dataloader, device, num_samples: int = 3, save_dir=None):
    """Distorted | Reconstruction | Ground Truth qualitative comparison,
    matching the notebook's figure layout.
    """
    model.eval()
    samples_collected = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                dist_img, clean_img, paths = batch
            else:
                dist_img, clean_img = batch
                paths = [f"sample_{i}" for i in range(dist_img.shape[0])]

            dist_img, clean_img = dist_img.to(device), clean_img.to(device)
            output = model(dist_img)
            if output.shape != clean_img.shape:
                output = torch.nn.functional.interpolate(
                    output, size=clean_img.shape[2:], mode="bilinear", align_corners=False
                )

            mask_tensor = compute_leaf_mask(clean_img)
            mask_np = mask_tensor.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(dist_img.shape[0]):
                if samples_collected >= num_samples:
                    return

                dist_np = dist_img[i].cpu().numpy().transpose(1, 2, 0)
                out_np = output[i].cpu().numpy().transpose(1, 2, 0)
                clean_np = clean_img[i].cpu().numpy().transpose(1, 2, 0)

                pred = np.clip(out_np, 0, 1)
                target = np.clip(clean_np, 0, 1)
                mask_i = (mask_np[i, :, :, 0] > 0.5).astype(np.float32)

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(np.clip(dist_np, 0, 1))
                plt.title("Distorted")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(pred)
                plt.title("Reconstruction")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(target)
                plt.title("Ground Truth")
                plt.axis("off")
                plt.tight_layout()
                plt.show()

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    filename = (
                        os.path.basename(paths[i])
                        if isinstance(paths[i], str)
                        else f"sample_{samples_collected}.png"
                    )
                    recon_path = os.path.join(save_dir, f"recon_{filename}")
                    plt.imsave(recon_path, pred)
                    print(f"  Saved: {recon_path}")

                if samples_collected == 0:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(mask_np[i, :, :, 0], cmap="gray")
                    plt.title("Leaf Mask (white = leaf region)")
                    plt.axis("off")
                    plt.show()
                    print(f"Mask area (pixels): {mask_i.sum()}")

                samples_collected += 1


def save_training_results(train_losses, psnr_score, ssim_score, filename="training_results.json"):
    if not train_losses:
        print("No results to save.")
        return
    results = {
        "train_losses": [float(loss) for loss in train_losses],
        "final_psnr": float(psnr_score),
        "final_ssim": float(ssim_score),
        "epochs": len(train_losses),
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to {filename}")


def save_metrics_csv(model, dataloader, device, save_path):
    """Per-image PSNR/SSIM CSV using leaf-masked metrics with global
    fallback, matching the notebook's `save_metrics_csv()` exactly.
    """
    model.eval()
    records = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                dist_img, clean_img, paths = batch
            else:
                dist_img, clean_img = batch
                paths = [f"sample_{i}" for i in range(dist_img.shape[0])]

            dist_img, clean_img = dist_img.to(device), clean_img.to(device)
            output = model(dist_img)
            if output.shape != clean_img.shape:
                output = torch.nn.functional.interpolate(
                    output, size=clean_img.shape[2:], mode="bilinear", align_corners=False
                )

            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            clean_np = clean_img.cpu().numpy().transpose(0, 2, 3, 1)
            mask_tensor = compute_leaf_mask(clean_img)
            mask_np = mask_tensor.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(dist_img.shape[0]):
                pred = np.clip(output_np[i], 0, 1)
                target = np.clip(clean_np[i], 0, 1)
                mask_i = (mask_np[i, :, :, 0] > 0.5).astype(np.float32)

                psnr_val, ssim_val = leaf_masked_metrics(target, pred, mask_i)

                records.append(
                    {
                        "filename": os.path.basename(paths[i])
                        if isinstance(paths[i], str)
                        else f"sample_{i}",
                        "PSNR": float(psnr_val),
                        "SSIM": float(ssim_val),
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")
    return df
