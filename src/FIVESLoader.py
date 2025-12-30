"""
FIVES Dataset Loader

Handles the FIVES dataset structure:
- Original images: ./Original/0001.png, 0002.png, ...
- Segmented images: ./Segmented/0001.png, 0002.png, ...

Both directories have the same filenames (no suffix pattern needed).
Performs train/validation/test split from the same directory.

IMPORTANT: Normalization statistics are computed from TRAINING SET ONLY
to prevent data leakage.
"""

import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Sampler
import numpy as np
from PIL import Image
import random
import json
from easydict import EasyDict
from typing import Callable, Tuple, List, Dict, Optional, Iterator
import math


# Global variable to store computed normalization stats
_NORMALIZATION_STATS = None
_NORMALIZATION_STATS_FILE = "./normalization_stats.json"


class Colors:
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def center_padding(img, target_size: List[int], pad_digit: int = 0):
    """Pad image to target size, centering the original image."""
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        original_mode = img.mode
        img_tensor = TF.to_tensor(img)
    else:
        img_tensor = img

    if img_tensor.ndim == 4:
        in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    elif img_tensor.ndim == 3:
        in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    elif img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)
        in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    else:
        raise ValueError(f"Unsupported tensor ndim: {img_tensor.ndim}")

    target_h, target_w = target_size[0], target_size[1]

    if in_h >= target_h and in_w >= target_w:
        return img

    pad_left = max(0, (target_w - in_w) // 2)
    pad_right = max(0, target_w - in_w - pad_left)
    pad_top = max(0, (target_h - in_h) // 2)
    pad_bot = max(0, target_h - in_h - pad_top)

    import torch.nn.functional as F
    tensor_padded = F.pad(img_tensor, [pad_left, pad_right, pad_top, pad_bot], 'constant', pad_digit)

    if is_pil:
        pil_image_padded = TF.to_pil_image(
            tensor_padded.squeeze(0) if tensor_padded.ndim == 4 and tensor_padded.shape[0] == 1 else tensor_padded,
            mode=original_mode if original_mode in ['L', 'RGB'] else None)
        return pil_image_padded
    else:
        return tensor_padded


def compute_normalization_stats(
    train_samples: List[Dict[str, str]],
    save_path: str = _NORMALIZATION_STATS_FILE
) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std from TRAINING images only to prevent data leakage.
    
    Args:
        train_samples: List of training image paths
        save_path: Path to save computed stats
        
    Returns:
        Tuple of (mean, std) as lists [R, G, B]
    """
    global _NORMALIZATION_STATS
    
    print(f"{Colors.CYAN}Computing normalization statistics from training set only...{Colors.END}")
    
    # Accumulate pixel values
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    
    for sample in train_samples:
        img = Image.open(sample["image"]).convert('RGB')
        img_np = np.array(img, dtype=np.float64) / 255.0  # Normalize to [0, 1]
        
        pixel_sum += img_np.sum(axis=(0, 1))
        pixel_sq_sum += (img_np ** 2).sum(axis=(0, 1))
        pixel_count += img_np.shape[0] * img_np.shape[1]
    
    # Compute mean and std
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    # Save stats to file
    stats = {
        "mean": mean_list,
        "std": std_list,
        "num_training_images": len(train_samples),
        "total_pixels": pixel_count
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"{Colors.GREEN}Normalization stats computed from {len(train_samples)} training images:{Colors.END}")
    print(f"  Mean: {mean_list}")
    print(f"  Std:  {std_list}")
    print(f"{Colors.GREEN}Stats saved to: {save_path}{Colors.END}")
    
    _NORMALIZATION_STATS = (mean_list, std_list)
    return mean_list, std_list


def load_normalization_stats(stats_path: str = _NORMALIZATION_STATS_FILE) -> Optional[Tuple[List[float], List[float]]]:
    """
    Load previously computed normalization stats.
    
    Args:
        stats_path: Path to stats file
        
    Returns:
        Tuple of (mean, std) or None if not found
    """
    global _NORMALIZATION_STATS
    
    if _NORMALIZATION_STATS is not None:
        return _NORMALIZATION_STATS
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        _NORMALIZATION_STATS = (stats["mean"], stats["std"])
        print(f"{Colors.GREEN}Loaded normalization stats from {stats_path}{Colors.END}")
        print(f"  Mean: {stats['mean']}")
        print(f"  Std:  {stats['std']}")
        return _NORMALIZATION_STATS
    
    return None


def generate_fives_dataset_list(
    data_root: str,
    original_subdir: str = "Original",
    segmented_subdir: str = "Segmented",
) -> List[Dict[str, str]]:
    """
    Generate list of image-label pairs for FIVES dataset.
    
    Both original and segmented images have the SAME filename.
    
    Args:
        data_root: Root directory containing Original and Segmented folders
        original_subdir: Subdirectory name for original images
        segmented_subdir: Subdirectory name for segmented images
        
    Returns:
        List of dictionaries with 'image' and 'label' paths
    """
    dataset_list = []
    original_folder = os.path.join(data_root, original_subdir)
    segmented_folder = os.path.join(data_root, segmented_subdir)
    
    if not os.path.isdir(original_folder):
        print(f"{Colors.RED}Error: Original folder not found: {original_folder}{Colors.END}")
        return dataset_list
    if not os.path.isdir(segmented_folder):
        print(f"{Colors.RED}Error: Segmented folder not found: {segmented_folder}{Colors.END}")
        return dataset_list
    
    # Get all image files from original folder
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    original_files = sorted([
        f for f in os.listdir(original_folder) 
        if f.lower().endswith(image_extensions)
    ])
    
    for img_filename in original_files:
        original_path = os.path.join(original_folder, img_filename)
        # For FIVES, segmented has the SAME filename
        segmented_path = os.path.join(segmented_folder, img_filename)
        
        if os.path.exists(segmented_path):
            dataset_list.append({
                "image": original_path,
                "label": segmented_path
            })
        else:
            # Try common extensions if exact match not found
            base_name = os.path.splitext(img_filename)[0]
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_path = os.path.join(segmented_folder, base_name + ext)
                if os.path.exists(alt_path):
                    dataset_list.append({
                        "image": original_path,
                        "label": alt_path
                    })
                    found = True
                    break
            if not found:
                print(f"{Colors.YELLOW}Warning: No segmented image found for {img_filename}{Colors.END}")
    
    return dataset_list


def split_dataset(
    dataset_list: List[Dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset_list: Full list of image-label pairs
        train_ratio: Proportion for training (default 0.7 = 70%)
        val_ratio: Proportion for validation (default 0.15 = 15%)
        test_ratio: Proportion for testing (default 0.15 = 15%)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    random.seed(seed)
    shuffled = dataset_list.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_samples = shuffled[:train_end]
    val_samples = shuffled[train_end:val_end]
    test_samples = shuffled[val_end:]
    
    return train_samples, val_samples, test_samples


class FIVESDataset(Dataset):
    """FIVES Dataset class for 1024x1024 retinal images."""
    
    def __init__(
        self,
        samples: List[Dict[str, str]],
        mode: str,
        dataset_config: EasyDict,
        image_mean: List[float],
        image_std: List[float],
        loader: Callable = Image.open,
    ):
        """
        Initialize FIVES dataset.
        
        Args:
            samples: List of {'image': path, 'label': path} dictionaries
            mode: 'train', 'validation', or 'test'
            dataset_config: Configuration from config.yml
            image_mean: Normalization mean (computed from training set)
            image_std: Normalization std (computed from training set)
            loader: Image loading function
        """
        super().__init__()
        self.samples = samples
        self.mode = mode
        self.args = dataset_config
        self.loader = loader
        
        # Handle image size
        if isinstance(self.args.image_size, int):
            self.args.image_size = [self.args.image_size, self.args.image_size]
        
        # Use normalization stats computed from training set only
        self.image_mean = image_mean
        self.image_std = image_std
        
        self.img_paths_x = [s["image"] for s in self.samples]
        self.img_paths_y = [s["label"] for s in self.samples]
        
        # Load images into memory
        print(f'{Colors.LIGHT_RED}Loading FIVES data into memory... Mode: {self.mode}{Colors.END}')
        self.images_pil_x = []
        self.images_pil_y = []
        
        for idx in range(len(self.img_paths_x)):
            try:
                img_x = self.loader(self.img_paths_x[idx]).convert('RGB')
                img_y = self.loader(self.img_paths_y[idx]).convert('L')
                self.images_pil_x.append(img_x)
                self.images_pil_y.append(img_y)
            except Exception as e:
                print(f"{Colors.RED}Error loading: {self.img_paths_x[idx]} or {self.img_paths_y[idx]}. Error: {e}{Colors.END}")
                raise
        
        if not self.images_pil_x:
            print(f"{Colors.RED}Warning: No images loaded for mode {self.mode}{Colors.END}")
    
    def __len__(self) -> int:
        return len(self.images_pil_x)
    
    def _transform(self, image: Image.Image, target: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image and target."""
        image_p = image.copy()
        target_p = target.copy()
        target_h, target_w = self.args.image_size[0], self.args.image_size[1]
        
        # Validation/test: only resize/pad
        if self.mode == 'validation' or self.mode == 'test':
            img_w_orig, img_h_orig = image_p.size
            if img_h_orig < target_h or img_w_orig < target_w:
                image_p = center_padding(image_p, [target_h, target_w], pad_digit=0)
                target_p = center_padding(target_p, [target_h, target_w], pad_digit=0)
        
        # Training: apply augmentations
        if self.mode == 'train':
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image_p = TF.hflip(image_p)
                target_p = TF.hflip(target_p)
            
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image_p = TF.vflip(image_p)
                target_p = TF.vflip(target_p)
            
            # Random rotation (0, 90, 180, 270 degrees)
            if torch.rand(1).item() > 0.5:
                angle = random.choice([90, 180, 270])
                image_p = TF.rotate(image_p, angle)
                target_p = TF.rotate(target_p, angle)
            
            # Random resized crop (scaling)
            if hasattr(self.args, 'transform_random_resized_crop') and self.args.transform_random_resized_crop:
                if random.random() < self.args.get('transform_random_resized_crop_prob', 0.3):
                    scale = self.args.get('transform_random_resized_crop_scale', (0.8, 1.2))
                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        image_p, scale=scale, ratio=(0.9, 1.1)
                    )
                    image_p = TF.resized_crop(image_p, i, j, h, w, [target_h, target_w], antialias=True)
                    target_p = TF.resized_crop(
                        target_p, i, j, h, w, [target_h, target_w],
                        interpolation=InterpolationMode.NEAREST
                    )
        
        # Build image transform pipeline
        img_transform_list = [
            transforms.Resize([target_h, target_w], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ]
        
        # Training: add color jitter (brightness/contrast)
        if self.mode == 'train':
            if hasattr(self.args, 'transform_jitter') and self.args.transform_jitter:
                if random.random() < self.args.get('transform_jitter_prob', 0.5):
                    jitter_params = self.args.get('jitter_params', {
                        'brightness': 0.2,
                        'contrast': 0.2,
                        'saturation': 0.1,
                        'hue': 0.05
                    })
                    # Insert jitter before ToTensor
                    img_transform_list.insert(0, transforms.ColorJitter(**jitter_params))
            
            # Optional blur
            if hasattr(self.args, 'transform_blur') and self.args.transform_blur:
                if random.random() < self.args.get('transform_blur_prob', 0.3):
                    kernel_size = random.choice([3, 5])
                    blur_sigma = self.args.get('blur_sigma', (0.1, 1.0))
                    img_transform_list.insert(0, transforms.GaussianBlur(kernel_size=kernel_size, sigma=blur_sigma))
        
        # Apply transforms to image
        image_tensor = transforms.Compose(img_transform_list)(image_p)
        
        # Process target (binary mask)
        if target_p.mode != 'L':
            target_p = target_p.convert('L')
        lbl_tensor_raw = TF.to_tensor(target_p)
        lbl_tensor_binary = (lbl_tensor_raw > 0.5).float()
        target_tensor = TF.resize(
            lbl_tensor_binary, [target_h, target_w],
            interpolation=InterpolationMode.NEAREST, antialias=False
        )
        
        return image_tensor, target_tensor
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """Get item by index."""
        if index >= len(self.images_pil_x):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.images_pil_x)}")
        
        img_x_pil = self.images_pil_x[index]
        img_y_pil = self.images_pil_y[index]
        
        img_x_tensor, img_y_tensor = self._transform(img_x_pil, img_y_pil)
        return img_x_tensor, img_y_tensor, self.img_paths_x[index], self.img_paths_y[index]


class FIVESDataLoader:
    """DataLoader wrapper for FIVES dataset."""
    
    def __init__(
        self,
        samples: List[Dict[str, str]],
        mode: str,
        dataset_config: EasyDict,
        image_mean: List[float],
        image_std: List[float],
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize FIVES DataLoader.
        
        Args:
            samples: List of image-label pairs
            mode: 'train', 'validation', or 'test'
            dataset_config: Configuration
            image_mean: Normalization mean (from training set)
            image_std: Normalization std (from training set)
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
        """
        self.dataset_config = dataset_config
        self.mode = mode
        self.dataset = FIVESDataset(
            samples=samples,
            mode=mode,
            dataset_config=dataset_config,
            image_mean=image_mean,
            image_std=image_std
        )
        
        g = torch.Generator()
        g.manual_seed(self.dataset_config.get('random_seed', 3407))
        
        self.Loader = TorchDataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            drop_last=(mode == 'train')
        )
    
    def __len__(self):
        return len(self.dataset)


def get_dataloader(config: EasyDict) -> Tuple[Optional[TorchDataLoader], Optional[TorchDataLoader], Optional[TorchDataLoader]]:
    """
    Get train, validation, and test dataloaders for FIVES dataset.
    
    Args:
        config: Full configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_params = config.dataset.FIVES
    dataset_params.name = "FIVES"
    
    data_root = dataset_params.data_root
    batch_size = dataset_params.batch_size
    num_workers = dataset_params.num_workers
    train_ratio = dataset_params.get('train_ratio', 0.7)
    val_ratio = dataset_params.get('val_ratio', 0.15)
    test_ratio = dataset_params.get('test_ratio', 0.15)
    
    # Get subdirectory names
    original_subdir = dataset_params.get('image_subdir', 'Original')
    segmented_subdir = dataset_params.get('label_subdir', 'Segmented')
    
    print(f"{Colors.CYAN}Loading FIVES dataset from: {data_root}{Colors.END}")
    print(f"{Colors.CYAN}Original folder: {original_subdir}{Colors.END}")
    print(f"{Colors.CYAN}Segmented folder: {segmented_subdir}{Colors.END}")
    
    # Generate dataset list
    all_samples = generate_fives_dataset_list(
        data_root=data_root,
        original_subdir=original_subdir,
        segmented_subdir=segmented_subdir
    )
    
    if not all_samples:
        raise ValueError(f"No image pairs found in {data_root}")
    
    print(f"{Colors.GREEN}Found {len(all_samples)} image pairs{Colors.END}")
    
    # Split into train/val/test (DETERMINISTIC with seed=42)
    train_samples, val_samples, test_samples = split_dataset(
        all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42
    )
    
    print(f"{Colors.GREEN}Train: {len(train_samples)}, Val: {len(val_samples) if val_samples else 0}, Test: {len(test_samples)}{Colors.END}")
    
    # Compute or load normalization stats FROM TRAINING SET ONLY
    # This prevents data leakage - test/val data never influences normalization
    stats = load_normalization_stats()
    if stats is None:
        print(f"{Colors.YELLOW}Computing normalization stats from training set...{Colors.END}")
        image_mean, image_std = compute_normalization_stats(train_samples)
    else:
        image_mean, image_std = stats
    
    # Store stats in config for test.py to access
    dataset_params.computed_mean = image_mean
    dataset_params.computed_std = image_std
    
    # Create dataloaders - ALL use stats computed from training set only
    train_loader = None
    val_loader = None
    test_loader = None
    
    if train_samples:
        train_dataloader_wrapper = FIVESDataLoader(
            samples=train_samples,
            mode='train',
            dataset_config=dataset_params,
            image_mean=image_mean,
            image_std=image_std,
            batch_size=batch_size,
            num_workers=num_workers
        )
        train_loader = train_dataloader_wrapper.Loader
        print(f"{Colors.GREEN}Created train loader with {len(train_samples)} samples{Colors.END}")
    
    if val_samples and len(val_samples) > 0:
        val_dataloader_wrapper = FIVESDataLoader(
            samples=val_samples,
            mode='validation',
            dataset_config=dataset_params,
            image_mean=image_mean,
            image_std=image_std,
            batch_size=batch_size,
            num_workers=num_workers
        )
        val_loader = val_dataloader_wrapper.Loader
        print(f"{Colors.GREEN}Created validation loader with {len(val_samples)} samples{Colors.END}")
    else:
        print(f"{Colors.YELLOW}No validation set (val_ratio=0.0){Colors.END}")
    
    if test_samples:
        test_dataloader_wrapper = FIVESDataLoader(
            samples=test_samples,
            mode='test',
            dataset_config=dataset_params,
            image_mean=image_mean,
            image_std=image_std,
            batch_size=batch_size,
            num_workers=num_workers
        )
        test_loader = test_dataloader_wrapper.Loader
        print(f"{Colors.GREEN}Created test loader with {len(test_samples)} samples{Colors.END}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the loader
    import yaml
    
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    print("Testing FIVES DataLoader...")
    train_loader, val_loader, test_loader = get_dataloader(config)
    
    if train_loader:
        print(f"\nTrain loader batches: {len(train_loader)}")
        for batch in train_loader:
            images, masks, img_paths, mask_paths = batch
            print(f"Batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
            print(f"Image path example: {img_paths[0]}")
            break
    
    if val_loader:
        print(f"\nVal loader batches: {len(val_loader)}")
        for batch in val_loader:
            images, masks, img_paths, mask_paths = batch
            print(f"Batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
            break
    
    if test_loader:
        print(f"\nTest loader batches: {len(test_loader)}")
        for batch in test_loader:
            images, masks, img_paths, mask_paths = batch
            print(f"Batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
            break
