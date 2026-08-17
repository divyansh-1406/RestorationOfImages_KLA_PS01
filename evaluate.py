"""
Standalone inference/evaluation script (NOT a notebook).

Loads the trained FourmerNet checkpoint, runs inference on every image in
the input directory, and writes the restored (denoised + super-resolved)
output for each one to the output directory, using the same filename.

Input images are expected in the same format used throughout the notebook:
single-channel .npy arrays (any range; normalized internally to [0, 1] and
resized to 128x128 if not already that size, matching `_load_gray` in the
notebook). Outputs are written as .npy arrays in [0, 1], shape (256, 256)
(or (128*scale, 128*scale) if a different --scale was used at training time).

Usage:
    python evaluate.py --input_dir /path/to/test_images \
                        --output_dir /path/to/restored_outputs \
                        --checkpoint ./checkpoints/best1.pt
"""

import argparse
import os
import time

import numpy as np
import torch

from src.dataset import _load_gray
from src.model import FourmerNet


def parse_args():
    p = argparse.ArgumentParser(description="Run FourmerNet inference on a directory of test images")
    p.add_argument("--input_dir", type=str, required=True,
                    help="Path to directory of test images (.npy, single-channel).")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Path to directory where restored outputs will be written.")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_finetuned.pt",
                    help="Path to trained model checkpoint (.pt).")
    p.add_argument("--base_ch", type=int, default=96,
                    help="Must match the base_ch used to train the checkpoint.")
    p.add_argument("--num_blocks", type=int, default=8,
                    help="Must match the num_blocks used to train the checkpoint.")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4],
                    help="Must match the scale used to train the checkpoint.")
    p.add_argument("--input_size", type=int, default=128,
                    help="Expected LR input resolution; inputs are resized to this if needed.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = FourmerNet(in_ch=1, out_ch=1, base_ch=args.base_ch,
                        num_blocks=args.num_blocks, scale=args.scale).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}, val_psnr {ckpt.get('val_psnr', float('nan')):.2f} dB)")

    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not filenames:
        raise RuntimeError(f"No .npy files found in {args.input_dir}")
    print(f"Found {len(filenames)} test images.")

    t0 = time.time()
    with torch.no_grad():
        for fname in filenames:
            in_path = os.path.join(args.input_dir, fname)
            lr = _load_gray(in_path, args.input_size)
            lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            pred = model(lr_t).clamp(0, 1)
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            out_path = os.path.join(args.output_dir, fname)
            np.save(out_path, pred_np)

    dt = time.time() - t0
    print(f"Wrote {len(filenames)} restored outputs to {args.output_dir} "
          f"in {dt:.1f}s ({dt / len(filenames) * 1000:.1f} ms/image).")


if __name__ == "__main__":
    main()
