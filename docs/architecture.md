# Architecture: Attention Residual U-Net (ARU-Net)

This document describes the model exactly as implemented in `src/model.py`
(a faithful port of the notebook's `AttentionResidualUNet`). It corresponds
to thesis Figures 3.11-3.14 (the thesis's own methodology diagrams), **not**
Figure 2.1, which is reproduced from a different paper ([31] in the
thesis's references, a SAR-interferometry paper) during the literature
review and does not describe this thesis's own implementation.

## Overview

ARU-Net combines three components inside a standard U-Net encoder-decoder
skeleton:

1. **Encoder-decoder structure** for dense pixel-wise reconstruction via
   hierarchical, multi-scale feature extraction with skip connections.
2. **Residual blocks** at every encoder/decoder stage and in the
   bottleneck, to stabilize gradient flow.
3. **Attention gates** on every skip connection, to suppress irrelevant
   (background/noisy) encoder activations before fusing them with decoder
   features.

## Input / output

- Input: `(B, 3, H, W)` RGB tensor, values in `[0, 1]`.
- Output: `(B, 3, H, W)` RGB tensor, values in `[0, 1]` (enforced by a
  final `Sigmoid`).
- `H` and `W` must each be divisible by 16 (4 pooling stages of stride 2),
  or the skip-connection concatenations will fail to align.

## Encoder

| Stage | Input channels | Output channels | Layers |
|---|---:|---:|---|
| enc1 | 3   | 64   | Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> ResidualBlock(64) |
| pool1 | 64  | 64   | MaxPool2d(2) |
| enc2 | 64  | 128  | Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> ResidualBlock(128) |
| pool2 | 128 | 128  | MaxPool2d(2) |
| enc3 | 128 | 256  | Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> ResidualBlock(256) |
| pool3 | 256 | 256  | MaxPool2d(2) |
| enc4 | 256 | 512  | Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> ResidualBlock(512) |
| pool4 | 512 | 512  | MaxPool2d(2) |

Every convolution uses `kernel_size=3, stride=1, padding=1`, matching
thesis Sec. 3.2.1's description of the residual block's internal
convolutions.

## Bottleneck

`Conv3x3(512 -> 1024) -> ReLU -> ResidualBlock(1024)`

This is the deepest, most semantically abstract layer, matching thesis
Sec. 3.2.3.

## Decoder

Each decoder stage: `ConvTranspose2d(stride=2)` upsampling, followed by an
attention-gated skip-connection concatenation, followed by
`Conv3x3 -> ReLU -> ResidualBlock`.

| Stage | Upsample | Attention gate | Concat channels | Conv output |
|---|---|---|---:|---:|
| dec4 | ConvTranspose2d(1024->512, k=2, s=2) | att4: F_g=512, F_l=512, F_int=256 | 1024 | 512 |
| dec3 | ConvTranspose2d(512->256, k=2, s=2)  | att3: F_g=256, F_l=256, F_int=128 | 512  | 256 |
| dec2 | ConvTranspose2d(256->128, k=2, s=2)  | att2: F_g=128, F_l=128, F_int=64  | 256  | 128 |
| dec1 | ConvTranspose2d(128->64,  k=2, s=2)  | att1: F_g=64,  F_l=64,  F_int=32  | 128  | 64  |

## Output layer

`Conv2d(64 -> 3, kernel_size=1)` followed by `Sigmoid`, producing the final
reconstructed RGB image in `[0, 1]`.

## Residual blocks

```
ResidualBlock(x):
    out = ReLU(Conv3x3(x))
    out = Conv3x3(out)
    return ReLU(out + x)
```

**Discrepancy with the thesis:** Sec. 3.2.1 of the thesis states that
"Batch normalization is applied after each convolutional layer" within
residual blocks. **The notebook's `ResidualBlock` contains no BatchNorm
layers at all** -- only two convolutions, a ReLU, and the identity skip
addition. This implementation matches the notebook (the actual code, per
the project's "notebook = source of truth for implementation" rule) and
does not add BatchNorm, since doing so would silently change the
architecture beyond what was actually trained.

## Attention gates

Additive attention gates (Oktay et al. style), matching thesis Sec. 3.2.5:

```
AttentionBlock(g, x):
    psi = ReLU(W_g(g) + W_x(x))     # W_g, W_x: 1x1 Conv + BatchNorm
    psi = Sigmoid(BatchNorm(Conv1x1(psi)))
    return x * psi
```

`g` is the decoder's upsampled gating signal; `x` is the corresponding
encoder feature map. The resulting `psi` is a per-pixel attention
coefficient in `[0, 1]`, used to scale (suppress or preserve) encoder
activations before concatenation with the decoder path.

## Parameter count

Verified by running `python src/model.py`:

```
AttentionResidualUNet parameters: 50,224,815
```

The thesis does not report a parameter count anywhere, so this figure is
provided here as a new, code-derived fact (not present in the thesis) for
completeness.
