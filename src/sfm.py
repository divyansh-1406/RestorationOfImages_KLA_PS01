"""
Stochastic Frequency Masking (SFM), El Helou et al., ECCV 2020.

Copied verbatim from the notebook's SFM cells (no logic changes).
"""

import numpy as np
import torch
from scipy.fftpack import dct, idct


def _dct2(x: np.ndarray) -> np.ndarray:
    return dct(dct(x, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(x: np.ndarray) -> np.ndarray:
    return idct(idct(x, axis=1, norm="ortho"), axis=0, norm="ortho")


def _radius_map(h: int, w: int) -> np.ndarray:
    """Distance of each DCT coefficient from the DC term (top-left corner),
    normalized to [0, 1] by the max possible radius."""
    i = np.arange(h).reshape(-1, 1)
    j = np.arange(w).reshape(1, -1)
    r = np.sqrt(i.astype(np.float64) ** 2 + j.astype(np.float64) ** 2)
    return r / r.max()


def _soft_band_mask(h: int, w: int, center_perc: float, sigma_perc: float) -> np.ndarray:
    """1 everywhere except a Gaussian notch (dipping toward 0) centered at
    radius=center_perc*max_radius with spread sigma_perc*max_radius. A soft
    (not hard-cutoff) mask avoids ringing artifacts from a sharp frequency cut."""
    r = _radius_map(h, w)
    sigma = max(sigma_perc, 1e-3)
    notch = np.exp(-((r - center_perc) ** 2) / (2 * sigma ** 2))
    return 1.0 - notch


def random_drop(img: np.ndarray, mode: int = 2,
                 SFM_center_radius_perc: float = 0.85,
                 SFM_center_sigma_perc: float = 0.15,
                 rng: np.random.Generator = None):
    """
    Apply SFM to a single-channel image.

    img: 2D numpy array (H, W), single grayscale image, float, any range.
    mode: 1 = "central" (wide band, used for SR augmentation)
          2 = "targeted" (narrow band near a target radius, used for denoising)
    SFM_center_radius_perc / SFM_center_sigma_perc: only used in targeted mode
        (defaults match the values shown in the paper's official README usage
        example for denoising).

    Returns (masked_img, mask) both as numpy arrays same shape as img.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape

    if mode == 1:
        # Central mode: sample two radii uniformly to delimit a wider masking
        # band (paper: "slow probability decay that covers wider bands").
        r_lo, r_hi = sorted(rng.uniform(0.05, 0.95, size=2))
        center = (r_lo + r_hi) / 2
        sigma = max((r_hi - r_lo) / 2, 0.08)
        mask = _soft_band_mask(h, w, center, sigma)
    elif mode == 2:
        # Targeted mode: narrow notch near a fixed high-frequency target,
        # with a bit of jitter so it's not the exact same band every call.
        jitter = rng.normal(0, 0.03)
        center = float(np.clip(SFM_center_radius_perc + jitter, 0.05, 0.98))
        mask = _soft_band_mask(h, w, center, SFM_center_sigma_perc)
    else:
        raise ValueError("mode must be 1 (central) or 2 (targeted)")

    coeffs = _dct2(img.astype(np.float64))
    coeffs_masked = coeffs * mask
    out = _idct2(coeffs_masked)
    # DCT-domain masking can introduce mild ringing that pushes values
    # slightly outside the original range -- clip back to [0, 1], which is
    # what the dataset below normalizes images to.
    out = np.clip(out, 0.0, 1.0)
    return out.astype(img.dtype), mask


def apply_sfm_batch(images: np.ndarray, dor: float = 0.5, mode: int = 2,
                     SFM_center_radius_perc: float = 0.85,
                     SFM_center_sigma_perc: float = 0.15,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    images: numpy array (N, H, W) or (N, 1, H, W), single-channel batch.
    dor: DCT dropout rate -- probability each sample in the batch gets SFM
         applied (paper calls this the "SFM rate").
    Returns a new array, same shape, with SFM stochastically applied.
    Only ever call this on the network's INPUT, never on ground truth.
    """
    if rng is None:
        rng = np.random.default_rng()
    squeeze_channel = images.ndim == 4
    if squeeze_channel:
        images = images[:, 0]

    out = images.copy()
    apply_bool = rng.choice([1, 0], size=(images.shape[0],), p=[dor, 1 - dor])
    for idx in range(images.shape[0]):
        if apply_bool[idx] == 1:
            out[idx], _ = random_drop(
                images[idx], mode=mode,
                SFM_center_radius_perc=SFM_center_radius_perc,
                SFM_center_sigma_perc=SFM_center_sigma_perc,
                rng=rng,
            )
    if squeeze_channel:
        out = out[:, None]
    return out


def sfm_collate_factory(dor: float = 0.5, mode: int = 2,
                         center_radius_perc: float = 0.85,
                         center_sigma_perc: float = 0.15):
    """Batches (lr, hr) pairs and applies SFM to the LR batch only, with
    probability `dor` per sample. Use for the TRAIN DataLoader only."""
    def collate(batch):
        lrs = torch.stack([b[0] for b in batch])
        hrs = torch.stack([b[1] for b in batch])
        lrs_np = lrs.numpy()
        lrs_np = apply_sfm_batch(lrs_np, dor=dor, mode=mode,
                                  SFM_center_radius_perc=center_radius_perc,
                                  SFM_center_sigma_perc=center_sigma_perc)
        lrs = torch.from_numpy(lrs_np).float()
        return lrs, hrs
    return collate
