#!/usr/bin/env python3
"""
Command-line training entry point for ARU-Net.

Example:
    python scripts/train.py --dataset-name Noise_Multiply_Strong --epochs 150

This replaces the notebook's `__main__` block (Google Drive paths,
Colab-only mounting, and inline hyperparameters) with a configurable CLI
built on top of `configs/config.py`.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import get_config
from src.dataset import ImagePairDataset, split_dataset
from src.evaluate import save_training_results, visualize_sample_results
from src.model import AttentionResidualUNet, count_parameters
from src.train import train
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train ARU-Net for image reconstruction.")
    p.add_argument("--dataset-name", type=str, default=None, help="Distortion folder name, e.g. Noise_Multiply_Strong")
    p.add_argument("--data-root", type=str, default=None, help="Root data directory (default: data/)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    p.add_argument("--save-dir", type=str, default=None, help="Where to write metrics/checkpoints")
    p.add_argument("--dist-suffix", type=str, default=None)
    p.add_argument("--clean-suffix", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    if args.dataset_name:
        cfg.paths.dataset_name = args.dataset_name
    if args.data_root:
        cfg.paths.data_root = args.data_root
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size_train = args.batch_size
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.patience:
        cfg.train.early_stopping_patience = args.patience
    if args.dist_suffix:
        cfg.dataset.dist_suffix = args.dist_suffix
    if args.clean_suffix:
        cfg.dataset.clean_suffix = args.clean_suffix

    save_dir = args.save_dir or str(cfg.paths.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    set_seed(cfg.train.random_seed)
    device = get_device()
    print(f"Using device: {device}")

    full_dataset = ImagePairDataset(
        distorted_dir=str(cfg.paths.distorted_dir),
        clean_dir=str(cfg.paths.clean_dir),
        transform=None,
        dist_suffix=cfg.dataset.dist_suffix,
        clean_suffix=cfg.dataset.clean_suffix,
        allowed_extensions=cfg.dataset.allowed_extensions,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, cfg)
    print(
        f"Train / Val / Test -> {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size_train, shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size_eval, shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size_eval, shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    model = AttentionResidualUNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.train.lr_scheduler_step_size,
        gamma=cfg.train.lr_scheduler_gamma,
    )

    train_losses = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=cfg.train.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        patience=cfg.train.early_stopping_patience,
    )

    save_training_results(
        train_losses,
        psnr_score=0.0,
        ssim_score=0.0,
        filename=os.path.join(save_dir, "training_results.json"),
    )

    checkpoint_path = os.path.join(
        cfg.paths.checkpoint_dir, f"{cfg.paths.dataset_name}_AttentionResidualUNet_final.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Final model saved -> {checkpoint_path}")


if __name__ == "__main__":
    main()
