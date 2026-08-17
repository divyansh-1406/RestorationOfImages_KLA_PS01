"""
Train the FourmerNet joint denoising + super-resolution (128x128 -> 256x256)
model from scratch.

Reproduces the notebook's main training loop (data loading, SFM-augmented
DataLoader, model/optimizer/scheduler, epoch loop, best-checkpoint saving on
validation PSNR) with the same defaults used in the notebook
(base_ch=96, num_blocks=8, scale=2, 100 epochs, batch size 16, lr 2e-4).

Usage:
    python train.py --data_root /path/to/kla-hackathon-data \
                     --output_dir ./checkpoints
"""

import argparse
import os
import time

import numpy as np
import torch

from src.dataset import PairedRestorationDataset, build_splits
from src.losses import RestorationLoss
from src.metrics import evaluate
from src.model import FourmerNet
from src.sfm import sfm_collate_factory


def parse_args():
    p = argparse.ArgumentParser(description="Train FourmerNet from scratch")
    p.add_argument("--data_root", type=str, required=True,
                    help="Directory containing GT/ and NoisyLR_train/ subfolders "
                         "of .npy grayscale images.")
    p.add_argument("--output_dir", type=str, default="./checkpoints",
                    help="Where to write the best checkpoint (best1.pt).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sfm_rate", type=float, default=0.5,
                    help="Fraction of each training batch SFM is applied to.")
    p.add_argument("--sfm_mode", type=int, default=2, choices=[1, 2],
                    help="1 = central (SR-style), 2 = targeted (denoising-style).")
    p.add_argument("--use_phase_loss", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--base_ch", type=int, default=96)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    lr_dir = os.path.join(args.data_root, "NoisyLR_train")
    hr_dir = os.path.join(args.data_root, "GT")

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "best1.pt")

    train_files, val_files, test_files = build_splits(
        lr_dir, hr_dir, seed=args.seed, train_frac=0.95, val_frac=0.04,
    )

    train_ds = PairedRestorationDataset(lr_dir, hr_dir, train_files)
    val_ds = PairedRestorationDataset(lr_dir, hr_dir, val_files)

    train_collate = sfm_collate_factory(dor=args.sfm_rate, mode=args.sfm_mode)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=train_collate, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = FourmerNet(in_ch=1, out_ch=1, base_ch=args.base_ch,
                        num_blocks=args.num_blocks, scale=args.scale).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = RestorationLoss(use_phase_loss=args.use_phase_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_psnr = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            pred = model(lr_img)
            loss, logs = criterion(pred, hr_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_psnr = evaluate(model, val_loader, device)
        dt = time.time() - t0
        avg_loss = running_loss / len(train_loader)
        print(f"epoch {epoch:03d}/{args.epochs} | loss {avg_loss:.4f} | "
              f"val PSNR {val_psnr:.2f} dB | {dt:.1f}s")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_psnr": float(val_psnr)},
                       best_ckpt_path)

    print(f"Training complete. Best val PSNR: {best_val_psnr:.2f} dB "
          f"(saved to {best_ckpt_path})")
    print(f"Held-out test filenames ({len(test_files)}): {test_files}")


if __name__ == "__main__":
    main()
