"""
Metrics used during training (running PSNR) and final evaluation
(PSNR / SSIM / LPIPS via torchmetrics).

Copied verbatim from the notebook's metric cells (no logic changes).
"""

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def evaluate(model, loader, device):
    model.eval()
    total_psnr, n = 0.0, 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr).clamp(0, 1)
            total_psnr += psnr(pred, hr) * lr.size(0)
            n += lr.size(0)
    model.train()
    return total_psnr / max(n, 1)


def test_Fourmer(model, test_loader, device):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    model.to(device)

    # Evaluation:
    model.eval()

    with torch.no_grad():
        for noisy, fixed in test_loader:
            noisy = noisy.to(device)
            fixed = fixed.to(device)

            # Forward pass
            predicted = model(noisy)
            predicted = torch.clamp(predicted, 0.0, 1.0)

            # Calculating metrics
            psnr_metric.update(predicted, fixed)
            ssim_metric.update(predicted, fixed)

            # Prepare 1-channel grayscale to 3-channel [-1, 1] for LPIPS
            predicted_3ch = predicted.repeat(1, 3, 1, 1) * 2.0 - 1.0
            fixed_3ch = fixed.repeat(1, 3, 1, 1) * 2.0 - 1.0

            # Feed LPIPS accumulator
            lpips_metric.update(predicted_3ch, fixed_3ch)

    final_psnr = psnr_metric.compute()
    final_ssim = ssim_metric.compute()
    final_lpips = lpips_metric.compute()

    print("\n=== Final Test Results ===")
    print(f"Average PSNR:  {final_psnr.item():.4f} dB")
    print(f"Average SSIM:  {final_ssim.item():.4f}")
    print(f"Average LPIPS: {final_lpips.item():.4f}")

    psnr_metric.reset()
    ssim_metric.reset()
    lpips_metric.reset()

    return {
        "psnr": final_psnr.item(),
        "ssim": final_ssim.item(),
        "lpips": final_lpips.item(),
    }
