# Methodology

This document summarizes the full research methodology as described in the
thesis (Chapter 3), and explicitly notes where the actual notebook
implementation differs. See the top-level README's **Research
Implementation Audit** table for the complete, itemized comparison.

## Problem framing

Plant disease detection models trained on clean images degrade sharply
when deployed on real, field-captured images that suffer from noise,
blur, and illumination artifacts. This thesis proposes using an image
reconstruction model (ARU-Net) as a preprocessing stage: restore degraded
agricultural images before they reach a downstream disease-detection
pipeline, rather than modifying the detection pipeline itself.

**Scope note (thesis Sec. 1.5):** the thesis explicitly limits itself to
the reconstruction stage and its quantitative evaluation. Downstream
disease-detection/segmentation integration is stated as future work, not
implemented in the thesis or notebook.

## Four-stage pipeline (thesis Sec. 3, Fig. 3.1)

1. **Dataset construction using instance segmentation** -- Roboflow-based
   disease-region annotation exported as COCO, converted to binary disease
   masks (thesis Sec. 3.1.4-3.1.5).
2. **Mask-guided selective degradation** -- disease regions preserved,
   background degraded via `I_d = M*I + (1-M)*D(I)` (thesis Sec. 3.1.7).
   **Not present in the supplied notebook** -- see `data/README.md`.
3. **Image reconstruction via ARU-Net** -- see `docs/architecture.md`.
4. **(Future work) downstream disease segmentation evaluation** -- not
   implemented in either the thesis's experiments or the notebook.

## Dataset (thesis Sec. 3.1)

- 103 raw rice-leaf RGB images (83 diseased / 20 healthy), field-captured.
- Expanded to 1,033 images via 10 noise/blur degradation types: Gaussian
  Blur, Median Blur, Bilateral Blur, Noise Half Split, Noise Half Split
  Strong, Noise Multiply, Noise Multiply Strong, Noise Overlay, Noise
  Overlay Strong, Normal Noise (thesis Fig. 3.6).
- The notebook's `ImagePairDataset` consumes **one degradation-type folder
  at a time** (via `dataset_name`, e.g. `"Noise_Multiply_Strong"`), pairing
  each distorted file with its clean counterpart by filename-suffix
  substitution. It does not train across all 10 folders simultaneously in
  a single run.

## Loss function -- the most significant thesis/code discrepancy

**Thesis (Sec. 3.4):** a two-term loss,

```
L_total = alpha * L1 + beta * (1 - SSIM)
```

with `alpha`/`beta` described only as "weighting coefficients ... selected
empirically to balance convergence stability and reconstruction fidelity"
-- no concrete values are given anywhere in the thesis.

**Notebook (actual code, `CombinedLoss`):** a four-term loss with
hard-coded weights,

```
L_total = 0.40 * L1
        + 0.25 * VGG16-perceptual (features[:16], ImageNet-pretrained, frozen)
        + 0.20 * Laplacian edge loss (kornia.filters.Laplacian(3))
        + 0.15 * (1 - masked SSIM)
```

The perceptual and edge terms are entirely absent from the thesis's
written description of the loss, and the concrete 0.40/0.25/0.20/0.15
weighting is never stated in the thesis text. `src/losses.py` implements
the notebook's actual four-term loss, since the code is the authoritative
source for "actual implementation" per this project's instructions; this
document records the discrepancy rather than silently resolving it in
either direction.

## "Masked" SSIM -- the second major discrepancy

The thesis's masking concept (Sec. 3.1.4-3.1.7) is about **disease
lesions**: preserve disease pixels during degradation, and (implicitly)
evaluate reconstruction quality with awareness of those disease regions.

The notebook's masking function, `compute_leaf_mask()` (originally
`create_leaf_mask_from_rgb_tensor` in the notebook), performs **HSV
green-hue thresholding to separate the whole leaf from its background** --
it has no connection to disease lesions and does not use the COCO
annotations described in thesis Sec. 3.1.4-3.1.6 at all. The "masked SSIM"
term in `CombinedLoss`, and the "masked" PSNR/SSIM computed during
evaluation (`src/metrics.py`), are both leaf-vs-background metrics, not
disease-region metrics.

This repository is careful to name the function and its outputs
(`compute_leaf_mask`, `leaf_masked_metrics`) so this distinction is clear
and is never presented as if it were the thesis's disease mask.

## Training strategy (thesis Sec. 3.3)

| Aspect | Thesis text/table | Notebook (actual) |
|---|---|---|
| Optimizer | Adam | Adam (`lr=1e-4`) — matches |
| LR scheduler | not mentioned | `StepLR(step_size=15, gamma=0.5)` |
| Epochs | 120 (Table 3.1) | `train()` default 70, but `__main__` runs 150 |
| Batch size | "8/10/12" (ambiguous) | 4 (train), 1 (val/test) |
| Split | 80/20 (train/val only) | 80/10/10 (train/val/test) |
| Early stopping | "applied" (Sec 3.3) | claimed via `patience=20` kwarg in `__main__`, but `train()` doesn't accept/implement it as written -- see `src/train.py` docstring for the repository's correction |
| Mixed precision | not mentioned | `torch.cuda.amp` autocast + GradScaler when CUDA available |
| Image size | 640x640 (Sec 3.1.3 text) vs 256x256 (Table 3.1) | 224x224 random-resized crop (train), native resolution (val/test) |
| Augmentation | flip + rotation | RandomResizedCrop(224), HorizontalFlip, Rotation(5deg), ColorJitter(brightness=0.1, contrast=0.1) |

## Evaluation metrics (thesis Sec. 3.5, 4.3-4.5)

- **PSNR** and **SSIM**, computed per degradation type, reported as mean
  +/- standard deviation (thesis Tables 4.1, 4.2).
- MS-SSIM is mentioned in the abstract/keywords and literature review
  (Sec. 2.6) but is **not implemented** anywhere in the thesis's own
  experiments or in the notebook -- this repository does not implement it
  either, to avoid fabricating an unused metric.
- The thesis's reported Table 4.2 results (SSIM/PSNR per noise/blur type)
  are reproduced verbatim in the top-level README, clearly labeled as
  "reported in the thesis," since no raw results files were provided with
  this repository to regenerate them from.
