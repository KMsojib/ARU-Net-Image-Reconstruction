"""
Composite reconstruction loss for ARU-Net training.

Faithful port of `CombinedLoss` from `4_AttentionResidualUNet.ipynb`.

IMPORTANT DISCREPANCY (see Research Implementation Audit in the top-level
README and docs/methodology.md):

The thesis (Sec. 3.4) describes the training objective as a 2-term
combination of L1 and (1 - SSIM):

    L_total = alpha * L_L1 + beta * (1 - SSIM)

with alpha/beta stated only symbolically ("selected empirically", no
numeric values given), and no mention of a perceptual or edge term.

The notebook's actual `CombinedLoss` implements FOUR terms with concrete,
hard-coded weights:

    L_total = 0.40 * L1
            + 0.25 * VGG16-perceptual (features[:16], frozen, pretrained)
            + 0.20 * Laplacian edge loss
            + 0.15 * (1 - masked SSIM)

This module implements the notebook's actual 4-term loss, since the code
is the source of truth for "actual implementation" per the project
instructions. The thesis's simplified 2-term description is documented as
incomplete rather than silently reconciled.

Also note: the "masked SSIM" term uses `compute_leaf_mask` (see
src/utils.py) -- a whole-leaf-vs-background mask, not a disease-lesion
mask. See src/utils.py for details on this separate discrepancy.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import kornia
from torchmetrics.image import StructuralSimilarityIndexMeasure

from .utils import compute_leaf_mask


class CombinedLoss(nn.Module):
    """L1 + VGG16 perceptual + Laplacian edge + masked SSIM.

    Args:
        device: "cpu" or "cuda".
        use_vgg: whether to include the VGG16 perceptual term (matches the
            notebook's `use_vgg=True` default).
        w_pixel_l1, w_perceptual, w_edge, w_ssim: term weights. Defaults
            match the notebook's hard-coded values (0.4 / 0.25 / 0.2 / 0.15).
        vgg_feature_layers: number of leading VGG16 feature layers to keep
            (notebook uses `vgg16(pretrained=True).features[:16]`).
    """

    def __init__(
        self,
        device: str = "cpu",
        use_vgg: bool = True,
        w_pixel_l1: float = 0.4,
        w_perceptual: float = 0.25,
        w_edge: float = 0.2,
        w_ssim: float = 0.15,
        vgg_feature_layers: int = 16,
    ):
        super().__init__()
        self.device = device
        self.use_vgg = use_vgg
        self.w_pixel_l1 = w_pixel_l1
        self.w_perceptual = w_perceptual
        self.w_edge = w_edge
        self.w_ssim = w_ssim

        if use_vgg:
            vgg = (
                models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                .features[:vgg_feature_layers]
                .eval()
                .to(device)
            )
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.laplace = kornia.filters.Laplacian(3)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.clamp(output, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        if mask is None:
            with torch.no_grad():
                mask = compute_leaf_mask(target)  # (B, 1, H, W)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1 and output.shape[1] == 3:
            mask_rgb = mask.repeat(1, 3, 1, 1)
        else:
            mask_rgb = mask

        content_loss = (
            F.l1_loss(self.vgg(output), self.vgg(target)) if self.use_vgg else 0.0
        )
        pixel_loss = F.l1_loss(output, target)
        edge_loss = F.l1_loss(self.laplace(output), self.laplace(target))

        output_masked = output * mask_rgb
        target_masked = target * mask_rgb
        if mask_rgb.sum() <= 1e-6:
            ssim_value = self.ssim(output, target).mean()
        else:
            ssim_value = self.ssim(output_masked, target_masked).mean()
        ssim_loss = 1.0 - ssim_value

        total = (
            self.w_pixel_l1 * pixel_loss
            + self.w_perceptual * content_loss
            + self.w_edge * edge_loss
            + self.w_ssim * ssim_loss
        )
        return total
