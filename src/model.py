"""
Attention Residual U-Net (ARU-Net) for image reconstruction.

This is a faithful, line-for-line reconstruction of the model defined in
`4_AttentionResidualUNet.ipynb` (cell "3. Model: Attention Residual U-Net
(ARU-Net)"). No layers, channel widths, activation functions, or
connectivity have been changed, added, or removed.

IMPORTANT (see Research Implementation Audit in the top-level README):
The thesis (Sec. 3.2.1) states that "Batch normalization is applied after
each convolutional layer" inside residual blocks. The notebook's
`ResidualBlock`, reproduced below, does NOT contain any BatchNorm layers.
This is a known thesis/code discrepancy -- the implementation below matches
the notebook exactly, as instructed, and is NOT "corrected" to add
BatchNorm, since doing so would silently change the architecture.

Architecture summary (input 3-channel RGB image, output 3-channel RGB
reconstruction):

    Encoder:    enc1 (3->64)   -> pool -> enc2 (64->128)  -> pool ->
                enc3 (128->256)-> pool -> enc4 (256->512)  -> pool
    Bottleneck: 512 -> 1024
    Decoder:    up4+att4 (1024->512) -> up3+att3 (512->256) ->
                up2+att2 (256->128)  -> up1+att1 (128->64)
    Output:     1x1 conv (64->3) + Sigmoid

Each encoder/decoder stage = two 3x3 conv+ReLU layers followed by one
`ResidualBlock`. Attention gates (Oktay-style) filter each encoder skip
connection before concatenation with the corresponding decoder feature map.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a residual (identity) connection.

    Matches the notebook exactly: no BatchNorm is present (see module
    docstring for the discrepancy this creates with the thesis text).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + res)


class AttentionBlock(nn.Module):
    """Additive attention gate (Oktay et al., "Attention U-Net", 2018 style).

    Args:
        F_g: number of channels in the decoder gating signal.
        F_l: number of channels in the encoder feature map to be gated.
        F_int: number of intermediate channels used for the attention
            coefficient computation.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionResidualUNet(nn.Module):
    """Attention Residual U-Net (ARU-Net).

    Input:  (B, 3, H, W) RGB image, values expected in [0, 1].
    Output: (B, 3, H, W) reconstructed RGB image, values in [0, 1]
            (enforced by a final Sigmoid activation).
    """

    def __init__(self):
        super().__init__()
        # ---------------- Encoder ----------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            ResidualBlock(64),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            ResidualBlock(128),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            ResidualBlock(256),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            ResidualBlock(512),
        )
        self.pool4 = nn.MaxPool2d(2)

        # ---------------- Bottleneck (Center) ----------------
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1), nn.ReLU(True), ResidualBlock(1024)
        )

        # ---------------- Attention gates ----------------
        self.att4 = AttentionBlock(512, 512, 256)
        self.att3 = AttentionBlock(256, 256, 128)
        self.att2 = AttentionBlock(128, 128, 64)
        self.att1 = AttentionBlock(64, 64, 32)

        # ---------------- Decoder ----------------
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(True), ResidualBlock(512)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(True), ResidualBlock(256)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(True), ResidualBlock(128)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(True), ResidualBlock(64)
        )

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        c = self.center(p4)

        d4 = self.up4(c)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return torch.sigmoid(self.out(d1))


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count (matches notebook's reported style)."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = AttentionResidualUNet()
    n_params = count_parameters(model)
    print(f"AttentionResidualUNet parameters: {n_params:,}")

    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
