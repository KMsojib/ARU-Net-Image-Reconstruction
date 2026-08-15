# Attention Residual U-Net for Image Reconstruction

Bachelor's thesis project: *"ARU-Net: Encoder-decoder based image
reconstruction towards efficient downstream tasks"* -- Mobashara (221031021)
& Md. Kawsar Mahmud (221031005), Department of Computer Science and
Engineering, Feni University, supervised by Shazzad Hossain Mazumder
(March 2026).

## Overview

Agricultural images captured in the field with low-cost devices are
frequently degraded by sensor noise, motion blur, and uneven illumination.
These degradations reduce the reliability of deep learning-based plant
disease detection, which depends on subtle lesion textures and edges. This
project proposes **ARU-Net (Attention Residual U-Net)**, an
encoder-decoder image reconstruction model that combines residual learning
and attention gating to restore degraded rice-leaf images, as a
preprocessing step intended to improve the reliability of downstream
agricultural computer-vision pipelines.

## Motivation

Existing crop-disease detection research (surveyed in the thesis, 2023-2026)
largely evaluates models on curated, high-quality laboratory images and
rarely accounts for real-field degradations such as blur, noise, and poor
illumination. Data augmentation alone exposes models to distortions during
training but does not restore structural information already lost in a
degraded input. This project investigates reconstruction-driven
preprocessing as a way to close that gap.

## Research Objective

Design and evaluate a deep learning-based image reconstruction framework
for degraded agricultural images, using an Attention Residual U-Net that
integrates residual learning and attention gating, and quantitatively
assess reconstruction quality (SSIM, PSNR) across multiple noise and blur
degradation types.

## Proposed Method: ARU-Net

ARU-Net integrates three architectural principles inside a U-Net backbone:
(1) hierarchical encoder-decoder feature extraction with skip connections,
(2) residual blocks for stable gradient flow, and (3) additive attention
gates on every skip connection to suppress irrelevant/noisy activations
before fusion with the decoder path. See `docs/architecture.md` for full
layer-by-layer detail.

```
Input -> Encoder (64->128->256->512) -> Bottleneck (1024)
      -> Decoder (512->256->128->64) w/ Attention-gated skip connections
      -> 1x1 Conv -> Sigmoid -> Reconstructed Output
```

## Architecture

- **Encoder:** 4 stages, each two 3x3 conv+ReLU layers followed by a
  residual block, then 2x2 max-pooling. Channels: 64 -> 128 -> 256 -> 512.
- **Bottleneck:** 3x3 conv (512->1024) + residual block.
- **Decoder:** 4 stages, each a 2x2 transposed convolution upsample,
  attention-gated skip fusion with the matching encoder stage, then two
  3x3 conv+ReLU + residual block. Channels: 512 -> 256 -> 128 -> 64.
- **Attention gates:** additive (Oktay-style) gates on all four skip
  connections.
- **Output:** 1x1 convolution to 3 channels + Sigmoid.
- **Parameters:** 50,224,815 (verified via `python src/model.py`).

**Known discrepancy:** the thesis text (Sec. 3.2.1) states BatchNorm is
applied inside residual blocks; the actual `ResidualBlock` in the notebook
(and this repository) contains no BatchNorm layers. See
`docs/architecture.md` for details.

## Loss Function

**Implemented (source of truth: the notebook's `CombinedLoss`):**

```
L_total = 0.40 * L1(output, target)
        + 0.25 * VGG16-perceptual(output, target)      # features[:16], frozen, ImageNet-pretrained
        + 0.20 * Laplacian-edge(output, target)         # kornia.filters.Laplacian(3)
        + 0.15 * (1 - masked_SSIM(output, target))      # leaf-vs-background mask, see below
```

The thesis (Sec. 3.4) describes a simpler two-term loss,
`L = alpha*L1 + beta*(1-SSIM)`, with unspecified `alpha`/`beta` weights and
no mention of the perceptual or edge terms. This repository implements the
loss actually present in the code; the discrepancy is documented in full
in `docs/methodology.md`.

**On "masked":** the mask used here is a **leaf-vs-background** mask
computed at runtime via HSV green-hue thresholding
(`src/utils.compute_leaf_mask`) -- it is *not* the disease-lesion mask
described in the thesis's dataset chapter (Sec. 3.1.4-3.1.7), which is
never loaded or used anywhere in the notebook. See `docs/methodology.md`
for the full explanation.

## Dataset

- **Source:** 103 rice-leaf RGB images collected under natural field
  conditions (83 diseased, 20 healthy), thesis Sec. 3.1.1-3.1.2.
- **Expansion:** 1,033 images via 10 degradation types (Gaussian Blur,
  Median Blur, Bilateral Blur, Noise Half Split, Noise Half Split Strong,
  Noise Multiply, Noise Multiply Strong, Noise Overlay, Noise Overlay
  Strong, Normal Noise), thesis Fig. 3.6.
- **Annotation:** pixel-level disease-region instance segmentation via
  Roboflow, exported as COCO, converted to binary masks (thesis Sec.
  3.1.4-3.1.5). **Not used anywhere in the notebook's training/evaluation
  code** -- see `docs/methodology.md`.
- Dataset is not redistributed with this repository; see `data/README.md`
  for the expected local layout and how to prepare your own copy.

## Experimental Setup

| Parameter | Value used in this repo (= notebook `__main__`) | Thesis Table 3.1 |
|---|---|---|
| Batch size | 4 (train) / 1 (val, test) | "8/10/12" |
| Learning rate | 1e-4 (Adam) | 1e-4 |
| Optimizer | Adam | Adam |
| Epochs | 150 | 120 |
| Scheduler | StepLR(step_size=15, gamma=0.5) | not specified |
| Image size | 224x224 random-resized crop (train); native resolution (val/test) | 256x256 (text elsewhere says 640x640) |
| Train/val/test split | 80% / 10% / 10% | 80% / 20% (val only, no test mentioned) |
| Data augmentation | RandomResizedCrop, HFlip, Rotation(5deg), ColorJitter | Flip, Rotation |
| Loss function | 0.4 L1 + 0.25 VGG-perceptual + 0.2 edge + 0.15 masked-SSIM | "Reconstruction loss" (alpha*L1 + beta*(1-SSIM)) |
| Mixed precision | Yes (CUDA only) | not mentioned |
| Hardware | GPU (falls back to CPU) | GPU |
| Framework | PyTorch | PyTorch |

Every cell in the right column that differs from the left is explained in
the **Research Implementation Audit** below and in `docs/methodology.md`.

## Results

The following table reproduces thesis Table 4.2 **as reported in the
thesis** (this repository does not ship trained weights or raw metrics
CSVs, so these numbers were not independently regenerated -- see
`docs/reproducibility.md` Section 8):

| Degradation Type | SSIM (Reconstructed) | PSNR (dB, Reconstructed) |
|---|---:|---:|
| Bilateral Blur | 0.9478 +/- 0.0518 | 32.31 +/- 3.58 |
| Gaussian Blur | 0.9420 +/- 0.0554 | 30.54 +/- 3.89 |
| Median Blur | 0.9264 +/- 0.928* | 30.85 +/- 3.96 |
| Noise Half Split | 0.9187 +/- 0.0852 | 30.23 +/- 3.69 |
| Noise Half Split Strong | 0.9063 +/- 0.1029 | 30.78 +/- 3.79 |
| Noise Multiply | 0.9167 +/- 0.0879 | 29.28 +/- 3.21 |
| Noise Multiply Strong | 0.9026 +/- 0.1006 | 28.57 +/- 3.32 |
| Noise Overlay | 0.8976 +/- 0.1056 | 28.04 +/- 3.70 |
| Noise Overlay Strong | 0.8943 +/- 0.1121 | 28.60 +/- 3.13 |
| Normal Noise | 0.9001 +/- 0.1091 | 29.05 +/- 3.36 |

*The `+/- 0.928` value for Median Blur SSIM is reproduced verbatim from
thesis Table 4.2; it is almost certainly a typo for `+/- 0.0928` (an SSIM
standard deviation cannot exceed 1.0), but is left unmodified here per the
"do not invent or silently correct reported results" instruction --
flagged for the authors to verify against their original run logs.

Reconstructed images consistently achieve higher SSIM than degraded inputs
across every degradation type (thesis Table 4.1); the largest gains are on
severely degraded inputs (e.g. Noise Multiply: SSIM 0.2061 -> 0.9167).

## Qualitative Results

Qualitative Distorted | Reconstruction | Ground-Truth comparisons (thesis
Fig. 4.1, 4.2) are not bundled as image files in this repository (no
result images were supplied with the source materials). Regenerate them
with:

```bash
python scripts/visualize.py samples --checkpoint <path_to_checkpoint> --dataset-name <name>
```

## Installation

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ARU-Net-Image-Reconstruction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

See `data/README.md`. Place clean images in `data/clean/` and each
degradation type's images in `data/distorted/<dataset_name>/`.

## Training

```bash
python scripts/train.py --dataset-name Noise_Multiply_Strong --epochs 150
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/Noise_Multiply_Strong_AttentionResidualUNet_final.pth \
    --dataset-name Noise_Multiply_Strong --split test
```

## Reproducibility

See `docs/reproducibility.md` for environment setup, seeding, and an
explicit list of what can and cannot be reproduced from the materials
provided (no raw dataset, no trained weights, and no degradation-generation
script were included in the source upload).

## Limitations (from thesis Sec. 5.2)

- Degradations are synthetically generated and may not capture all
  real-world sensor imperfections or uncontrolled field conditions.
- Reconstruction quality (PSNR/SSIM) was evaluated in isolation; the
  effect of reconstruction on actual downstream disease
  segmentation/classification performance was **not** experimentally
  evaluated in the thesis, despite the abstract and Chapter 4 introduction
  framing this as an "ablation study" -- no downstream detection code or
  results appear in the notebook or thesis body.
- The attention + residual design adds computational overhead relative to
  a plain U-Net, potentially limiting deployment on low-power/edge devices.
- Dataset size (103 raw images) was intentionally small for controlled
  experimentation; larger datasets may reveal different generalization
  behavior.

## Future Work (from thesis Sec. 5.3)

- Experimentally evaluate reconstructed images on downstream disease
  segmentation/classification/severity-estimation tasks.
- Explore end-to-end joint optimization of reconstruction and downstream
  networks.
- Validate on real (not synthetically degraded) field images from farmers'
  mobile devices.
- Lightweight ARU-Net variants via pruning, quantization, or knowledge
  distillation for real-time/edge use.
- Transformer-based attention for improved long-range dependency modeling.
- Adaptive, degradation-severity-aware reconstruction.

## Citation

See `CITATION.cff`. Plain-text form:

> Mobashara and Md. Kawsar Mahmud, "ARU-Net: Encoder-decoder based image
> reconstruction towards efficient downstream tasks," B.Sc. thesis,
> Department of Computer Science and Engineering, Feni University,
> supervised by Shazzad Hossain Mazumder, March 2026.

---

## Research Implementation Audit

A full component-by-component comparison between the thesis text and the
actual notebook code, including every identified discrepancy and how this
repository resolves it, is provided separately in the pull request /
project notes accompanying this repository, and summarized in
`docs/methodology.md` and `docs/architecture.md`. The two most significant
findings:

1. **Loss function:** thesis describes L1+SSIM; code implements
   L1+VGG-perceptual+edge+masked-SSIM (4 terms, concrete weights only in
   code).
2. **"Masked" evaluation/loss:** thesis's mask concept is about disease
   lesions (COCO-annotated); the code's mask is a leaf-vs-background HSV
   threshold with no relation to disease regions.

Every other discrepancy (image resolution, epochs, batch size, split
ratios, early stopping, scheduler, augmentation) is itemized with
thesis-value vs. code-value vs. repository-decision in `docs/methodology.md`.

## Repository Structure

```
ARU-Net-Image-Reconstruction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── configs/
│   └── config.py
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── notebooks/
│   └── demo.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reconstructions/
├── checkpoints/
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── reproducibility.md
└── data/
    └── README.md
```
