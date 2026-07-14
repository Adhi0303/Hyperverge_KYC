import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """The basic U-Net block: (3x3 conv -> BatchNorm -> ReLU) applied twice.

    `padding=1` keeps the height/width unchanged, so the block only changes the
    channel count (from `in_ch` to `out_ch`) while mixing in local context."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # first conv: in_ch -> out_ch
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # second conv: out_ch -> out_ch
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small 4-level U-Net (base=32 -> ~1.9M parameters).

    Encoder-decoder with skip connections:
      * the encoder halves the resolution and doubles the channels four times;
      * the decoder upsamples back to full resolution, and at each step
        concatenates the matching encoder feature map (the "skip") so the fine
        spatial detail lost during pooling is recovered.
    The output is a single-channel logit map (before sigmoid) at input size."""

    def __init__(self, base=32):
        super().__init__()

        # ---- encoder (contracting path) ----
        self.d1 = DoubleConv(3, base)               # RGB in     -> base channels
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)     # deepest level (bottleneck)
        self.pool = nn.MaxPool2d(2)                  # halves height and width

        # ---- decoder (expanding path) ----
        # each ConvTranspose doubles the resolution; the following DoubleConv
        # then fuses the upsampled features with the concatenated encoder skip,
        # so its input channel count is (upsampled + skip).
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)     # base*8 = up3 (base*4) + skip enc3 (base*4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.u1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, kernel_size=1)  # 1x1 conv -> one logit per pixel

    def forward(self, x):
        # ---- encoder: keep each level's output so the decoder can use it ----
        enc1 = self.d1(x)                       # full resolution
        enc2 = self.d2(self.pool(enc1))         # 1/2
        enc3 = self.d3(self.pool(enc2))         # 1/4
        bottleneck = self.d4(self.pool(enc3))   # 1/8 (deepest features)

        # ---- decoder: upsample, concatenate the matching skip, then fuse ----
        dec3 = self.u3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.u2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.u1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.out(dec1)                   # (N, 1, H, W) logits
