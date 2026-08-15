"""
Paired distorted/clean image dataset for ARU-Net training and evaluation.

Faithful port of `ImagePairDataset` and the `TransformDataset` split
wrapper from `4_AttentionResidualUNet.ipynb`. Behavior is unchanged; only
the following software-organization changes were made:

  * Hard-coded Google Drive paths (e.g.
    `/content/drive/MyDrive/FrameReconstruction/dataset/...`) are replaced
    with configurable paths from `configs/config.py`.
  * `google.colab.drive.mount(...)` is removed (Colab-specific, not
    reproducible outside Colab).

Pairing scheme (unchanged from the notebook): a distorted filename's
`dist_suffix` substring is replaced with `clean_suffix` to look up its
ground-truth counterpart, e.g. "leaf01_noise4_multiply_strong.jpg" ->
"leaf01_original.jpg".
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImagePairDataset(Dataset):
    """Paired dataset for distorted -> clean image reconstruction.

    Returns: (dist_tensor, clean_tensor, dist_path)
    """

    def __init__(
        self,
        distorted_dir: str,
        clean_dir: str,
        transform: Optional[transforms.Compose] = None,
        dist_suffix: str = "_noise4_multiply_strong",
        clean_suffix: str = "_original",
        allowed_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ):
        self.distorted_dir = distorted_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.dist_suffix = dist_suffix
        self.clean_suffix = clean_suffix

        distorted_filenames = sorted(
            f
            for f in os.listdir(distorted_dir)
            if f.lower().endswith(tuple(allowed_extensions))
        )

        print(f"Found {len(distorted_filenames)} distorted images in {distorted_dir}")
        self.data_pairs: List[Tuple[str, str]] = []
        for dist_file in distorted_filenames:
            clean_file = dist_file.replace(dist_suffix, clean_suffix)
            dist_path = os.path.join(distorted_dir, dist_file)
            clean_path = os.path.join(clean_dir, clean_file)
            if os.path.exists(clean_path):
                self.data_pairs.append((dist_path, clean_path))
            else:
                print(f"  [warning] Clean image not found for {dist_file}")

        if not self.data_pairs:
            raise FileNotFoundError(
                "No valid distorted/clean image pairs found. Check "
                "distorted_dir/clean_dir and dist_suffix/clean_suffix in "
                "configs/config.py."
            )

        print(f"Dataset ready with {len(self.data_pairs)} image pairs")

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int):
        dist_path, clean_path = self.data_pairs[idx]
        dist_img = Image.open(dist_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        # Apply the same random seed to both images so paired
        # augmentations (crop/flip/rotation) stay spatially aligned.
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        if self.transform:
            dist_tensor = self.transform(dist_img)
        else:
            dist_tensor = transforms.ToTensor()(dist_img)

        torch.manual_seed(seed)
        if self.transform:
            clean_tensor = self.transform(clean_img)
        else:
            clean_tensor = transforms.ToTensor()(clean_img)

        return dist_tensor, clean_tensor, dist_path


class TransformDataset(Dataset):
    """Applies a specific transform to a subset of indices from an
    `ImagePairDataset`. Used to give the train split augmentation while
    keeping val/test on a plain ToTensor() (full-resolution) pipeline, as
    in the notebook's __main__ block.
    """

    def __init__(
        self,
        base_dataset: ImagePairDataset,
        indices: Sequence[int],
        transform: Optional[transforms.Compose],
    ):
        self.base = base_dataset
        self.idx = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        idx = self.idx[i]
        dist_path, clean_path = self.base.data_pairs[idx]
        dist_img = Image.open(dist_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        dist_tensor = (
            self.transform(dist_img) if self.transform else transforms.ToTensor()(dist_img)
        )
        torch.manual_seed(seed)
        clean_tensor = (
            self.transform(clean_img) if self.transform else transforms.ToTensor()(clean_img)
        )

        return dist_tensor, clean_tensor, dist_path


def build_train_transform(cfg) -> transforms.Compose:
    """Matches the notebook's `train_transform` exactly."""
    aug = cfg.augmentation
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(aug.train_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(aug.random_rotation_degrees),
            transforms.ColorJitter(
                brightness=aug.color_jitter_brightness,
                contrast=aug.color_jitter_contrast,
            ),
            transforms.ToTensor(),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Matches the notebook's `eval_transform` exactly (no resize)."""
    return transforms.Compose([transforms.ToTensor()])


def split_dataset(full_dataset: ImagePairDataset, cfg):
    """Reproduces the notebook's 80/10/10 train/val/test random_split with
    a fixed seed, then wraps each split in a TransformDataset with the
    appropriate transform.
    """
    n = len(full_dataset)
    train_n = int(cfg.dataset.train_fraction * n)
    val_n = int(cfg.dataset.val_fraction * n)
    test_n = n - train_n - val_n

    torch.manual_seed(cfg.dataset.split_seed)
    train_idx, val_idx, test_idx = torch.utils.data.random_split(
        range(n), [train_n, val_n, test_n]
    )

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform()

    train_dataset = TransformDataset(full_dataset, train_idx.indices, train_transform)
    val_dataset = TransformDataset(full_dataset, val_idx.indices, eval_transform)
    test_dataset = TransformDataset(full_dataset, test_idx.indices, eval_transform)

    return train_dataset, val_dataset, test_dataset
