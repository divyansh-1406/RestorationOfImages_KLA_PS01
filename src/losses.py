"""
Loss functions: Charbonnier, gradient, SSIM, VGG perceptual, and the
combined RestorationLoss.

Copied verbatim from the notebook's loss cell (no logic changes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def grad(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy
    pgx, pgy = grad(pred)
    tgx, tgy = grad(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a fixed Gaussian window. Single-channel only."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = g.t() @ g
        self.register_buffer("window", window_2d.unsqueeze(0).unsqueeze(0))
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.window.to(pred.dtype)
        pad = self.window_size // 2
        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, window, padding=pad) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / \
                   ((mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))
        return 1 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Frozen VGG16 (up to relu3_3-ish) feature L1 loss. Grayscale inputs are
    replicated to 3 channels and normalized with ImageNet stats before being
    fed through VGG, since it was trained on RGB ImageNet data."""

    def __init__(self, layer_idx: int = 9):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:layer_idx]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p3 = pred.repeat(1, 3, 1, 1)
        t3 = target.repeat(1, 3, 1, 1)
        p3 = (p3 - self.mean) / self.std
        t3 = (t3 - self.mean) / self.std
        return F.l1_loss(self.vgg(p3), self.vgg(t3))


class RestorationLoss(nn.Module):
    def __init__(self, w_char=1.0, w_ssim=0.2, w_grad=0.1, w_amp=0.1,
                 w_phase=0.0, w_perc=0.0, use_phase_loss: bool = False):
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_amp = w_amp
        self.w_phase = w_phase if use_phase_loss else 0.0
        self.w_perc = w_perc
        self.ssim = SSIMLoss()
        self.perceptual = VGGPerceptualLoss() if self.w_perc > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_char = charbonnier_loss(pred, target)
        l_ssim = self.ssim(pred, target)
        l_grad = gradient_loss(pred, target)

        fft_pred = torch.fft.fft2(pred, norm="ortho")
        fft_target = torch.fft.fft2(target, norm="ortho")
        amp_pred, amp_target = torch.abs(fft_pred), torch.abs(fft_target)
        l_amp = F.l1_loss(amp_pred, amp_target)

        total = (self.w_char * l_char + self.w_ssim * l_ssim +
                 self.w_grad * l_grad + self.w_amp * l_amp)

        logs = {"char": l_char.item(), "ssim": l_ssim.item(),
                "grad": l_grad.item(), "amp": l_amp.item()}

        if self.w_phase > 0:
            phase_pred = torch.angle(fft_pred)
            phase_target = torch.angle(fft_target)
            l_phase = F.l1_loss(phase_pred, phase_target)
            total = total + self.w_phase * l_phase
            logs["phase"] = l_phase.item()

        if self.perceptual is not None:
            l_perc = self.perceptual(pred, target)
            total = total + self.w_perc * l_perc
            logs["perc"] = l_perc.item()

        return total, logs
