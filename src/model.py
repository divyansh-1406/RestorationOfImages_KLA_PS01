"""
Fourmer-inspired restoration network.

Copied verbatim from the notebook cells that define FourierUnit, SpatialUnit,
FourmerBlock and FourmerNet (no logic changes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit(nn.Module):
    """Global mixing branch: rFFT2 -> reweight amplitude & phase with a
    small 1x1-conv MLP over the (real, imag) stack -> irFFT2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = torch.fft.rfft2(x, norm="ortho")
        f_stack = torch.cat([f.real, f.imag], dim=1)  # (B, 2C, H, W//2+1)
        f_stack = self.conv(f_stack)
        real, imag = torch.chunk(f_stack, 2, dim=1)
        f_out = torch.complex(real, imag)
        out = torch.fft.irfft2(f_out, s=(h, w), norm="ortho")
        return out


class SpatialUnit(nn.Module):
    """Local mixing branch: plain depthwise-separable conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.act(self.dw(x)))


class FourmerBlock(nn.Module):
    """Fuses the Fourier (global) branch and spatial (local) branch with a
    learnable gate, residual connection, then a small feed-forward conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.fourier = FourierUnit(channels)
        self.spatial = SpatialUnit(channels)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)

        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        f_branch = self.fourier(y)
        s_branch = self.spatial(y)
        fused = self.gate(torch.cat([f_branch, s_branch], dim=1))
        x = x + fused
        x = x + self.ffn(self.norm2(x))
        return x


class FourmerNet(nn.Module):
    """
    Input:  (B, 1, 128, 128) grayscale, noisy/downsampled
    Output: (B, 1, 256, 256) grayscale, clean/super-resolved
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 48,
                 num_blocks: int = 8, scale: int = 2):
        # NOTE: scale=2 matches the data described (128x128 -> 256x256).
        # If you actually need 4x, fix your LR/GT resolutions first, then
        # pass scale=4 here (the 4x upsample path is already wired below).
        super().__init__()
        assert scale in (2, 4), "only 2x and 4x wired up below"
        self.scale = scale
        self.head = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[FourmerBlock(base_ch) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)

        if scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            )
        else:  # scale == 4
            self.up = nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            )
        self.tail = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_base = F.interpolate(x, scale_factor=self.scale, mode="bicubic",
                                       align_corners=False)

        feat = self.head(x)
        feat = self.body(feat) + feat
        feat = self.body_tail(feat)

        feat = self.up(feat)
        out = self.tail(feat)

        return out + residual_base
