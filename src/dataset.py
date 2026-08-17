"""
Paired LR/HR dataset loading, train/val/test split building.

Copied verbatim from the notebook's dataset cell (no logic changes).
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_gray(path: str, expected_size: int) -> np.ndarray:
    # Load the numpy array
    img_array = np.load(path)

    # Ensure float32 and normalize to [0, 1]
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    # Assuming the loaded images might be in range [0, 255] or already [0, 1].
    # Normalize if the max value suggests it's not [0, 1].
    if img_array.max() > 1.0:
        img_array = img_array / 255.0

    # Resize if necessary using PIL for convenience
    if img_array.shape != (expected_size, expected_size):
        # Convert float32 [0,1] to uint8 [0,255] for PIL, then back to float32 after resize
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
        img_pil = img_pil.resize((expected_size, expected_size), Image.BICUBIC)
        img_array = np.asarray(img_pil, dtype=np.float32) / 255.0
    return img_array


class PairedRestorationDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, filenames, lr_size: int = 128,
                 hr_size: int = 256):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filenames = filenames
        self.lr_size = lr_size
        self.hr_size = hr_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = _load_gray(os.path.join(self.lr_dir, fname), self.lr_size)
        hr = _load_gray(os.path.join(self.hr_dir, fname), self.hr_size)
        lr_t = torch.from_numpy(lr).unsqueeze(0)  # (1, H, W)
        hr_t = torch.from_numpy(hr).unsqueeze(0)
        return lr_t, hr_t


def build_splits(lr_dir: str, hr_dir: str, seed: int = 42,
                  train_frac: float = 0.95, val_frac: float = 0.04,
                  strict_filename_match: bool = True):
    """
    Returns (train_files, val_files, test_files), each a list of filenames.
    Split ratios: 95% train / 4% val / 1% test.
    """
    lr_files = sorted(os.listdir(lr_dir))
    hr_files = sorted(os.listdir(hr_dir))

    if strict_filename_match:
        common = sorted(set(lr_files) & set(hr_files))
        if len(common) != len(lr_files) or len(common) != len(hr_files):
            print(f"[warn] lr_dir has {len(lr_files)} files, hr_dir has "
                  f"{len(hr_files)} files, {len(common)} filenames match in "
                  f"both. Using only the {len(common)} matched pairs.")
        files = common
    else:
        assert len(lr_files) == len(hr_files), \
            "lr_dir and hr_dir must have the same number of files if not matching by name"
        files = lr_files

    n = len(files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    files = [files[i] for i in perm]

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"Split sizes -> train: {len(train_files)}, val: {len(val_files)}, "
          f"test: {len(test_files)} (of {n} total)")
    if n_test < 15:
        print(f"[note] test set is only {n_test} images -- PSNR/SSIM on it will "
              f"be noisy. Fine for a hackathon demo, don't over-read small "
              f"differences between checkpoints from it.")
    return train_files, val_files, test_files
