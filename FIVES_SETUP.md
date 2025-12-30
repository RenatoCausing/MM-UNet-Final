# FIVES Dataset Setup Guide

This guide explains how to set up and train with the FIVES dataset.

## Directory Structure

The training expects the following structure in the repository root:

```
MM-UNet-Final/
├── Original/           # Original images (0001.png, 0002.png, ..., 0800.png, 0801.png, ...)
├── Segmented/          # Segmented images (same names as Original)
├── preprocess.py       # Data augmentation script
├── train.py            # Training script
├── config.yml          # Configuration file
└── src/
    └── FIVESLoader.py  # FIVES dataset loader
```

## Step 1: Data Preprocessing (Run in Cloud)

Before training, run the preprocessing script to augment your dataset:

```bash
# Default settings (reads from D:\DRIVE\FIVES\Original\PNG and D:\DRIVE\FIVES\Segmented\PNG)
python preprocess.py

# Custom paths (for Google Colab/Cloud)
python preprocess.py \
    --original-dir "/content/drive/MyDrive/FIVES/Original/PNG" \
    --segmented-dir "/content/drive/MyDrive/FIVES/Segmented/PNG" \
    --output-original-dir "./Original" \
    --output-segmented-dir "./Segmented" \
    --start-index 801 \
    --augmentations-per-image 5 \
    --image-size 1024 1024
```

### Preprocessing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--original-dir` | `D:\DRIVE\FIVES\Original\PNG` | Input original images path |
| `--segmented-dir` | `D:\DRIVE\FIVES\Segmented\PNG` | Input segmented images path |
| `--output-original-dir` | `./Original` | Output original images path |
| `--output-segmented-dir` | `./Segmented` | Output segmented images path |
| `--start-index` | 801 | Starting index for augmented images |
| `--augmentations-per-image` | 5 | Number of augmented versions per image |
| `--image-size` | 1024 1024 | Target image size (width height) |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-copy-originals` | False | Skip copying original images |

### Augmentations Applied

- **Rotation**: 90°, 180°, 270° rotations
- **Flipping**: Horizontal, vertical, and combined flips
- **Scaling**: 85%, 90%, 110%, 115% scaling with center crop/pad
- **Brightness**: 80%, 90%, 110%, 120% brightness adjustment
- **Contrast**: 80%, 90%, 110%, 120% contrast adjustment
- **Combinations**: Mixed augmentations (rotation + brightness, flip + contrast, etc.)

## Step 2: Naming Convention

- Original images: `0001.png` to `0800.png` (your existing images)
- Augmented images: Start from `0801.png` onwards
- Segmented images: **Same filename** as corresponding original image

Example:
```
Original/0001.png  ↔  Segmented/0001.png
Original/0002.png  ↔  Segmented/0002.png
...
Original/0801.png  ↔  Segmented/0801.png  (augmented)
Original/0802.png  ↔  Segmented/0802.png  (augmented)
```

## Step 3: Configuration

The `config.yml` is already configured for FIVES. Key settings:

```yaml
trainer:
  dataset_choose: FIVES  # Use FIVES dataset

dataset:
  FIVES:
    data_root: ./              # Repository root
    batch_size: 2              # Adjust based on GPU memory (1024x1024 images are large)
    num_workers: 4
    image_size: 1024           # 1024x1024 images
    train_ratio: 0.8           # 80% train, 20% validation
    image_subdir: Original
    label_subdir: Segmented
```

### Adjusting Batch Size

For 1024x1024 images, you may need to reduce batch size based on your GPU:

- **8GB VRAM**: batch_size: 1
- **12GB VRAM**: batch_size: 2
- **16GB+ VRAM**: batch_size: 2-4

## Step 4: Training

```bash
# Start training
python train.py

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Step 5: Google Colab / Cloud Setup

For running in Google Colab:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone <your-repo-url>
%cd MM-UNet-Final

# Run preprocessing (from Google Drive data)
!python preprocess.py \
    --original-dir "/content/drive/MyDrive/FIVES/Original/PNG" \
    --segmented-dir "/content/drive/MyDrive/FIVES/Segmented/PNG" \
    --output-original-dir "./Original" \
    --output-segmented-dir "./Segmented"

# Install dependencies
!pip install -r requirements/requirements.txt

# Start training
!python train.py
```

## Output

- **Model checkpoints**: `./model_store/{model_name}/best/` and `./model_store/{model_name}/checkpoint/`
- **Logs**: `./logs/`
- **Visualization outputs**: `./visualization/FIVES/output/`
- **Numpy validation outputs**: `./visualization/FIVES/output/numpy/`

## Troubleshooting

1. **Out of Memory**: Reduce `batch_size` in config.yml
2. **No images found**: Check that `Original/` and `Segmented/` folders exist with images
3. **CUDA errors**: Set `CUDA_VISIBLE_DEVICES` to correct GPU index
