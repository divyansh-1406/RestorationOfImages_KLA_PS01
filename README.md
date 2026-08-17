# Fourmer Restoration — Joint Denoising + Super-Resolution (128×128 → 256×256)

A Fourmer-inspired network (global FFT-based mixing branch + local depthwise-conv
branch per block) trained with Stochastic Frequency Masking (SFM) augmentation
and a composite Charbonnier + SSIM + gradient + FFT-amplitude (+ optional
phase / VGG-perceptual) loss, to jointly denoise and 2x super-resolve
single-channel 128×128 images to 256×256.

## 1. Setup

```bash
git clone https://github.com/divyansh-1406/RestorationOfImages_KLA_PS01
cd RestorationOfImages_KLA_PS01
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## 2. Data layout

Each `.npy` file is a single-channel 2D array (any numeric range; the loader
normalizes to `[0, 1]` and resizes to the expected resolution if needed).

## 3. Training from scratch

```bash
python train.py --data_root /path/to/kla-hackathon-data --output_dir ./checkpoints
```

This reproduces the notebook's main training run: 95/4/1 train/val/test split
(seeded), SFM augmentation applied to a fraction of each training batch
(`--sfm_rate`, `--sfm_mode`), AdamW + cosine LR schedule, 100 epochs by
default, and the best checkpoint (by validation PSNR) saved to
`checkpoints/best1.pt`. Run `python train.py --help` for all hyperparameters
(defaults match the notebook: `base_ch=96`, `num_blocks=8`, `scale=2`,
`batch_size=16`, `lr=2e-4`).

## 4. Evaluation / inference (standalone script)

```bash
python evaluate.py --input_dir <insert_input_dir> --output_dir <insert_output_dir> --checkpoint ./checkpoints/best_finetuned.pt
```

- `--input_dir`: directory of `.npy` test images (128×128 grayscale).
- `--output_dir`: directory to write restored `.npy` outputs to (same
  filenames as the inputs), created if it doesn't exist.
- `--checkpoint`: path to a trained model checkpoint. If the checkpoint was
  trained with non-default `--base_ch` / `--num_blocks` / `--scale`, pass the
  matching values here too.
- In place of <insert_input_dir> and <insert_output_dir> place the input and output directories where the data is and where you want the generated images to be. 

This script loads the model, runs inference on every `.npy` file in
`input_dir`, and writes one restored `.npy` array per input to `output_dir`.
It does not require the training data, a GPU, or any manual edits to run.

## 5. Reproducing the reported metrics

To compute PSNR / SSIM / LPIPS on a held-out set with paired ground truth
(not just produce restored images), use `src/metrics.py`'s `test_Fourmer`
function against a `DataLoader` built from `src/dataset.py`'s
`PairedRestorationDataset` — see `train.py` for an example of wiring the
dataset and model together.

## 6. Project structure

```
RestorationOfImages_KLA_PS01/
├── README.md
├── requirements.txt
├── train.py              # standalone training script (from scratch)
├── evaluate.py            # standalone inference script (input dir -> output dir)
├── src/
│   ├── model.py           # FourierUnit, SpatialUnit, FourmerBlock, FourmerNet
│   ├── sfm.py              # Stochastic Frequency Masking augmentation
│   ├── losses.py           # Charbonnier / gradient / SSIM / VGG-perceptual / RestorationLoss
│   ├── dataset.py          # PairedRestorationDataset, build_splits
│   └── metrics.py          # running PSNR (train/val), final PSNR/SSIM/LPIPS (test_Fourmer)
├── checkpoints/            # trained weights go here (best_finetuned.pt)
└── outputs/                 # output directory along with predicted images in .png format contained in a zip file
```
