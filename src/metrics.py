"""
Evaluation metrics: global (whole-image) and leaf-masked PSNR/SSIM.

Faithful port of the metric computation used in `evaluate()` and
`save_metrics_csv()` in `4_AttentionResidualUNet.ipynb`.

Two distinct metric families are computed, and this module keeps them
clearly separate per the project's Step 8 requirement:

  1. Global image-quality metrics: PSNR/SSIM over the entire image.
  2. Leaf-masked metrics: PSNR/SSIM computed within a bounding-box crop of
     the leaf region (from `compute_leaf_mask`, see src/utils.py), falling
     back to global metrics when the mask is too small or degenerate.

Note again: "leaf-masked" here refers to whole-leaf-vs-background
segmentation, not the disease-lesion ROI described in the thesis. See
src/utils.py and the top-level README's Research Implementation Audit.
"""

from typing import Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk


def global_metrics(target: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Whole-image PSNR and SSIM. Inputs are (H, W, 3) arrays in [0, 1]."""
    psnr_val = psnr_sk(target, pred, data_range=1.0)
    ssim_val = ssim_sk(target, pred, channel_axis=-1, data_range=1.0)
    return float(psnr_val), float(ssim_val)


def leaf_masked_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    min_mask_pixels: int = 49,
    min_crop_side: int = 7,
    ssim_win_size: int = 7,
) -> Tuple[float, float]:
    """PSNR/SSIM computed within the bounding box of a binary leaf mask.

    Falls back to whole-image metrics when the mask region is too small
    or degenerate to support a valid SSIM window, matching the notebook's
    `evaluate()`/`save_metrics_csv()` logic exactly (7x7 minimum window,
    49-pixel minimum mask area).

    Args:
        target, pred: (H, W, 3) arrays in [0, 1].
        mask: (H, W) binary array (1 = leaf/foreground, 0 = background).
    """
    mask = mask.astype(np.float32)

    if mask.sum() < min_mask_pixels:
        return global_metrics(target, pred)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return global_metrics(target, pred)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop_h, crop_w = y1 - y0 + 1, x1 - x0 + 1

    if crop_h < min_crop_side or crop_w < min_crop_side:
        return global_metrics(target, pred)

    pred_crop = pred[y0 : y1 + 1, x0 : x1 + 1, :]
    target_crop = target[y0 : y1 + 1, x0 : x1 + 1, :]

    win_size = min(ssim_win_size, min(crop_h, crop_w))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    psnr_val = psnr_sk(target_crop, pred_crop, data_range=1.0)
    try:
        ssim_val = ssim_sk(
            target_crop, pred_crop, channel_axis=-1, data_range=1.0, win_size=win_size
        )
    except ValueError:
        return global_metrics(target, pred)

    return float(psnr_val), float(ssim_val)
