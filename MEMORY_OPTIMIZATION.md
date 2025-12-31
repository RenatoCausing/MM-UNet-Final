# Memory Optimization Guide

## Changes Applied

### 1. Reduced Batch Size
- **Before:** `batch_size: 4`
- **After:** `batch_size: 1`
- Reduces memory usage by 4x

### 2. Gradient Accumulation
- **Added:** `gradient_accumulation_steps: 4`
- Maintains effective batch size of 4 while using less memory
- Gradients are accumulated over 4 forward passes before updating weights

### 3. Mixed Precision Training (FP16)
- **Added:** `mixed_precision: fp16`
- Uses 16-bit floats instead of 32-bit, reducing memory by ~50%
- Maintains model accuracy with automatic loss scaling

### 4. Memory Cleanup
- Added `torch.cuda.empty_cache()` before training and after each epoch
- Releases unused GPU memory back to PyTorch

## Current Configuration (config.yml)

```yaml
trainer:
  gradient_accumulation_steps: 4
  mixed_precision: fp16

dataset:
  FIVES:
    batch_size: 1
```

## Additional Memory-Saving Tips (If Still Running Out)

### Option 1: Reduce Image Size
Edit `config.yml`:
```yaml
dataset:
  FIVES:
    image_size: 768  # Instead of 1024
```

### Option 2: Enable PyTorch Memory Fragmentation Fix
Before running training:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

On Windows PowerShell:
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python train.py
```

### Option 3: Use Gradient Checkpointing
If your model supports it, gradient checkpointing trades computation for memory by recomputing activations during backward pass instead of storing them.

### Option 4: Reduce Number of Workers
Edit `config.yml`:
```yaml
dataset:
  FIVES:
    num_workers: 2  # Instead of 4
```

## Monitor GPU Memory

Use this command to monitor GPU memory usage in real-time:
```bash
watch -n 1 nvidia-smi
```

On Windows PowerShell:
```powershell
while($true) { cls; nvidia-smi; Start-Sleep -Seconds 1 }
```

## Expected Memory Usage

With current settings:
- **Batch size:** 1
- **Image size:** 1024×1024
- **Mixed precision:** FP16
- **Expected GPU memory:** ~10-12 GB (should fit in your 16GB GPU)

## Training Performance

- **Effective batch size:** 4 (1 batch × 4 accumulation steps)
- **Training time:** ~4x longer per epoch due to smaller batch size
- **Model quality:** Should be identical to batch_size=4 without accumulation
