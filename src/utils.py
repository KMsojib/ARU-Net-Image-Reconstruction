"""
Utility functions: leaf-region masking, reproducibility seeding, and
lightweight visualization helpers.

NOTE ON NAMING (see Research Implementation Audit in the top-level README):
The thesis (Sec. 3.1.4-3.1.7) describes pixel-level **disease masks**
obtained from Roboflow instance-segmentation annotations, used to
selectively degrade background pixels while preserving lesion regions.

The notebook does NOT load or use those disease masks anywhere. Instead,
it computes a **leaf-vs-background mask at runtime** via HSV green-hue
thresholding (`create_leaf_mask_from_rgb_tensor` in the notebook). This
mask is used to compute a "masked SSIM" during training/evaluation, but it
segments the whole leaf from its background -- it has no relationship to
disease lesions.

To avoid misrepresenting this as the thesis's disease-region mask, this
function is named `compute_leaf_mask` here (the notebook's original name,
`create_leaf_mask_from_rgb_tensor`, is kept as an alias for drop-in
compatibility with the rest of the ported code).
"""

import random
from typing import Optional

import numpy as np
import torch
import kornia
from kornia.color import rgb_to_hsv


def compute_leaf_mask(
    img_rgb: torch.Tensor,
    low_h: float = 0.18,
    high_h: float = 0.45,
    min_sat: float = 0.15,
    min_val: float = 0.05,
) -> torch.Tensor:
    """Create a soft binary mask for leaf (green) regions from an RGB tensor.

    This is a whole-leaf-vs-background segmentation based on HSV hue/
    saturation/value thresholds -- it is NOT the disease-lesion mask
    described in the thesis (see module docstring).

    Args:
        img_rgb: (B, 3, H, W) tensor, values in [0, 1].
        low_h, high_h: hue range (in [0, 1]) considered "leaf green".
        min_sat: minimum saturation to be considered foreground.
        min_val: minimum value/brightness to be considered foreground.

    Returns:
        mask: (B, 1, H, W) float tensor with values in {0, 1}.
    """
    hsv = rgb_to_hsv(img_rgb)  # (B, 3, H, W): h, s, v in [0, 1]
    h = hsv[:, 0:1, :, :]
    s = hsv[:, 1:2, :, :]
    v = hsv[:, 2:3, :, :]

    mask_h = (h >= low_h) & (h <= high_h)
    mask_sv = (s >= min_sat) & (v >= min_val)
    mask = (mask_h & mask_sv).float()

    try:
        mask = kornia.filters.median_blur(mask, (3, 3))
    except Exception:
        # Fall back to the unfiltered mask if the installed kornia
        # version's median_blur signature differs.
        pass

    return mask


# Backwards-compatible alias matching the notebook's original function name.
create_leaf_mask_from_rgb_tensor = compute_leaf_mask


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for numpy, random, and torch (CPU + CUDA) for
    reproducibility, as referenced in thesis Sec. 3.6 ("Random seeds are
    fixed to ensure reproducibility")."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor in [0, 1] to an (H, W, C) numpy array."""
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)
