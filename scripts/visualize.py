#!/usr/bin/env python3
"""
Visualization entry point: qualitative reconstructions and training curves.

Examples:
    python scripts/visualize.py samples --checkpoint checkpoints/model.pth \
        --dataset-name Noise_Multiply_Strong --num-samples 5

    python scripts/visualize.py curves --results-json results/reconstructions/\
Noise_Multiply_Strong/training_results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import get_config
from src.dataset import ImagePairDataset, split_dataset
from src.evaluate import visualize_sample_results
from src.model import AttentionResidualUNet
from src.utils import get_device


def cmd_samples(args):
    cfg = get_config()
    if args.dataset_name:
        cfg.paths.dataset_name = args.dataset_name

    device = get_device()
    full_dataset = ImagePairDataset(
        distorted_dir=str(cfg.paths.distorted_dir),
        clean_dir=str(cfg.paths.clean_dir),
        transform=None,
        dist_suffix=cfg.dataset.dist_suffix,
        clean_suffix=cfg.dataset.clean_suffix,
    )
    _, _, test_dataset = split_dataset(full_dataset, cfg)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = AttentionResidualUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    save_dir = args.save_dir or "results/figures/samples"
    visualize_sample_results(model, loader, device, num_samples=args.num_samples, save_dir=save_dir)


def cmd_curves(args):
    with open(args.results_json, "r") as f:
        results = json.load(f)

    losses = results["train_losses"]
    epochs = len(losses)
    lr_step = args.lr_step
    lrs = [args.initial_lr * (args.gamma ** (e // lr_step)) for e in range(epochs)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(1, epochs + 1), losses, "b-", label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_title("Training Loss & Learning Rate Over Epochs")

    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), lrs, "r--", label="LR")
    ax2.set_ylabel("Learning Rate", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output)
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="ARU-Net visualization utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_samples = sub.add_parser("samples", help="Distorted | Reconstruction | Ground Truth panels")
    p_samples.add_argument("--checkpoint", required=True)
    p_samples.add_argument("--dataset-name", default=None)
    p_samples.add_argument("--num-samples", type=int, default=5)
    p_samples.add_argument("--save-dir", default=None)
    p_samples.set_defaults(func=cmd_samples)

    p_curves = sub.add_parser("curves", help="Training loss + LR curve from training_results.json")
    p_curves.add_argument("--results-json", required=True)
    p_curves.add_argument("--output", default="results/figures/training_curves.png")
    p_curves.add_argument("--initial-lr", type=float, default=1e-4)
    p_curves.add_argument("--lr-step", type=int, default=15)
    p_curves.add_argument("--gamma", type=float, default=0.5)
    p_curves.set_defaults(func=cmd_curves)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
