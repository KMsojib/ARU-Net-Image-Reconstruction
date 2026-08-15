#!/usr/bin/env python3
"""
Command-line evaluation entry point for ARU-Net.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/Noise_Multiply_Strong_AttentionResidualUNet_final.pth \
        --dataset-name Noise_Multiply_Strong --split test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from configs.config import get_config
from src.dataset import ImagePairDataset, split_dataset
from src.evaluate import evaluate
from src.model import AttentionResidualUNet
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ARU-Net checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .pth state_dict")
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    if args.dataset_name:
        cfg.paths.dataset_name = args.dataset_name
    if args.data_root:
        cfg.paths.data_root = args.data_root

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
    split_map = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    dataset = split_map[args.split]

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size_eval, shuffle=False)

    model = AttentionResidualUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    save_dir = args.save_dir or os.path.join(str(cfg.paths.metrics_dir), args.split)
    os.makedirs(save_dir, exist_ok=True)

    mean_psnr, mean_ssim, _, _ = evaluate(model, loader, device, save_dir=save_dir)
    print(f"\n[{args.split.upper()}] Mean PSNR: {mean_psnr:.4f} | Mean SSIM: {mean_ssim:.4f}")


if __name__ == "__main__":
    main()
