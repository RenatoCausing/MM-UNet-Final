"""
Data Augmentation Preprocessing Script for FIVES Dataset

This script performs data augmentation on the FIVES dataset:
- Random rotation (0, 90, 180, 270 degrees)
- Vertical and horizontal flipping
- Random scaling
- Brightness and contrast adjustment

Input:
- Original images: D:\DRIVE\FIVES\Original\PNG
- Segmented images: D:\DRIVE\FIVES\Segmented\PNG

Output:
- Augmented originals: ./Original (in repo, expects user to place files)
- Augmented segmented: ./Segmented (in repo, expects user to place files)

Naming convention:
- Original images: 0001.png to 0800.png (existing)
- Augmented images: start from 0801.png
"""

import os
import random
import argparse
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
from typing import Tuple, List


class DataAugmentor:
    """Data augmentation class for image segmentation datasets."""
    
    def __init__(
        self,
        original_dir: str,
        segmented_dir: str,
        output_original_dir: str,
        output_segmented_dir: str,
        start_index: int = 801,
        image_size: Tuple[int, int] = (1024, 1024),
        seed: int = 42
    ):
        """
        Initialize the augmentor.
        
        Args:
            original_dir: Path to original images
            segmented_dir: Path to segmented/label images
            output_original_dir: Output path for augmented original images
            output_segmented_dir: Output path for augmented segmented images
            start_index: Starting index for augmented image naming
            image_size: Target image size (width, height)
            seed: Random seed for reproducibility
        """
        self.original_dir = original_dir
        self.segmented_dir = segmented_dir
        self.output_original_dir = output_original_dir
        self.output_segmented_dir = output_segmented_dir
        self.start_index = start_index
        self.image_size = image_size
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directories
        os.makedirs(self.output_original_dir, exist_ok=True)
        os.makedirs(self.output_segmented_dir, exist_ok=True)
        
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """
        Get pairs of original and segmented images.
        
        Returns:
            List of tuples (original_path, segmented_path)
        """
        pairs = []
        
        # List all files in original directory
        if not os.path.exists(self.original_dir):
            raise FileNotFoundError(f"Original directory not found: {self.original_dir}")
        if not os.path.exists(self.segmented_dir):
            raise FileNotFoundError(f"Segmented directory not found: {self.segmented_dir}")
            
        original_files = sorted([f for f in os.listdir(self.original_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        for orig_file in original_files:
            # Get base name without extension
            base_name = os.path.splitext(orig_file)[0]
            orig_path = os.path.join(self.original_dir, orig_file)
            
            # Try to find corresponding segmented image
            # The segmented image should have the same name or with _segment suffix
            seg_candidates = [
                f"{base_name}.png",
                f"{base_name}_segment.png",
                f"{base_name}.jpg",
                f"{base_name}_segment.jpg",
            ]
            
            seg_path = None
            for candidate in seg_candidates:
                full_path = os.path.join(self.segmented_dir, candidate)
                if os.path.exists(full_path):
                    seg_path = full_path
                    break
            
            if seg_path:
                pairs.append((orig_path, seg_path))
            else:
                print(f"Warning: No segmented image found for {orig_file}")
                
        return pairs
    
    def rotate_image(self, image: Image.Image, angle: int) -> Image.Image:
        """Rotate image by specified angle (0, 90, 180, 270)."""
        if angle == 90:
            return image.transpose(Image.ROTATE_90)
        elif angle == 180:
            return image.transpose(Image.ROTATE_180)
        elif angle == 270:
            return image.transpose(Image.ROTATE_270)
        return image
    
    def flip_horizontal(self, image: Image.Image) -> Image.Image:
        """Flip image horizontally."""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def flip_vertical(self, image: Image.Image) -> Image.Image:
        """Flip image vertically."""
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def scale_image(self, image: Image.Image, scale_factor: float) -> Image.Image:
        """
        Scale image by factor and crop/pad to original size.
        
        Args:
            image: Input image
            scale_factor: Scale factor (e.g., 0.8 for 80%, 1.2 for 120%)
            
        Returns:
            Scaled and cropped/padded image
        """
        orig_width, orig_height = image.size
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        # Determine resampling method based on image mode
        if image.mode == 'L':
            resample = Image.NEAREST  # For masks, use nearest neighbor
        else:
            resample = Image.BILINEAR
        
        # Scale the image
        scaled = image.resize((new_width, new_height), resample=resample)
        
        # Create output image (same size as original)
        if image.mode == 'L':
            result = Image.new('L', (orig_width, orig_height), 0)
        else:
            result = Image.new('RGB', (orig_width, orig_height), (0, 0, 0))
        
        if scale_factor >= 1.0:
            # Crop center
            left = (new_width - orig_width) // 2
            top = (new_height - orig_height) // 2
            scaled = scaled.crop((left, top, left + orig_width, top + orig_height))
            result = scaled
        else:
            # Pad with black (center the scaled image)
            left = (orig_width - new_width) // 2
            top = (orig_height - new_height) // 2
            result.paste(scaled, (left, top))
            
        return result
    
    def adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust brightness of image.
        
        Args:
            image: Input image
            factor: Brightness factor (1.0 = original, <1 darker, >1 brighter)
        """
        if image.mode == 'L':
            return image  # Don't adjust brightness for masks
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust contrast of image.
        
        Args:
            image: Input image
            factor: Contrast factor (1.0 = original, <1 less contrast, >1 more contrast)
        """
        if image.mode == 'L':
            return image  # Don't adjust contrast for masks
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def apply_augmentations(
        self,
        original: Image.Image,
        segmented: Image.Image,
        augmentation_type: str
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply augmentation to image pair.
        
        Args:
            original: Original image
            segmented: Segmented/label image
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Tuple of augmented (original, segmented)
        """
        if augmentation_type == 'rotate_90':
            return self.rotate_image(original, 90), self.rotate_image(segmented, 90)
        
        elif augmentation_type == 'rotate_180':
            return self.rotate_image(original, 180), self.rotate_image(segmented, 180)
        
        elif augmentation_type == 'rotate_270':
            return self.rotate_image(original, 270), self.rotate_image(segmented, 270)
        
        elif augmentation_type == 'flip_h':
            return self.flip_horizontal(original), self.flip_horizontal(segmented)
        
        elif augmentation_type == 'flip_v':
            return self.flip_vertical(original), self.flip_vertical(segmented)
        
        elif augmentation_type == 'flip_both':
            orig = self.flip_horizontal(self.flip_vertical(original))
            seg = self.flip_horizontal(self.flip_vertical(segmented))
            return orig, seg
        
        elif augmentation_type.startswith('scale_'):
            scale = float(augmentation_type.split('_')[1])
            return self.scale_image(original, scale), self.scale_image(segmented, scale)
        
        elif augmentation_type.startswith('bright_'):
            factor = float(augmentation_type.split('_')[1])
            return self.adjust_brightness(original, factor), segmented
        
        elif augmentation_type.startswith('contrast_'):
            factor = float(augmentation_type.split('_')[1])
            return self.adjust_contrast(original, factor), segmented
        
        elif augmentation_type == 'combo_1':
            # Rotate 90 + brightness adjustment
            orig = self.rotate_image(original, 90)
            seg = self.rotate_image(segmented, 90)
            orig = self.adjust_brightness(orig, random.uniform(0.8, 1.2))
            return orig, seg
        
        elif augmentation_type == 'combo_2':
            # Flip horizontal + contrast adjustment
            orig = self.flip_horizontal(original)
            seg = self.flip_horizontal(segmented)
            orig = self.adjust_contrast(orig, random.uniform(0.8, 1.2))
            return orig, seg
        
        elif augmentation_type == 'combo_3':
            # Scale + brightness + flip
            scale = random.uniform(0.85, 1.15)
            orig = self.scale_image(original, scale)
            seg = self.scale_image(segmented, scale)
            orig = self.adjust_brightness(orig, random.uniform(0.85, 1.15))
            if random.random() > 0.5:
                orig = self.flip_horizontal(orig)
                seg = self.flip_horizontal(seg)
            return orig, seg
            
        return original, segmented
    
    def process(self, augmentations_per_image: int = 5, copy_originals: bool = True):
        """
        Process all images and generate augmented versions.
        
        Args:
            augmentations_per_image: Number of augmented versions per original image
            copy_originals: Whether to copy original images to output directory
        """
        pairs = self.get_image_pairs()
        print(f"Found {len(pairs)} image pairs")
        
        if len(pairs) == 0:
            print("No image pairs found. Please check input directories.")
            return
        
        # Define augmentation types to use
        augmentation_types = [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_h', 'flip_v', 'flip_both',
            'scale_0.85', 'scale_0.9', 'scale_1.1', 'scale_1.15',
            'bright_0.8', 'bright_0.9', 'bright_1.1', 'bright_1.2',
            'contrast_0.8', 'contrast_0.9', 'contrast_1.1', 'contrast_1.2',
            'combo_1', 'combo_2', 'combo_3'
        ]
        
        current_index = self.start_index
        
        # Copy original images first if requested
        if copy_originals:
            print("\nCopying original images...")
            for orig_path, seg_path in tqdm(pairs, desc="Copying originals"):
                # Load and resize to target size
                orig_img = Image.open(orig_path).convert('RGB')
                seg_img = Image.open(seg_path).convert('L')
                
                # Resize to target size if needed
                if orig_img.size != self.image_size:
                    orig_img = orig_img.resize(self.image_size, Image.BILINEAR)
                if seg_img.size != self.image_size:
                    seg_img = seg_img.resize(self.image_size, Image.NEAREST)
                
                # Get original filename
                orig_filename = os.path.basename(orig_path)
                seg_filename = os.path.basename(seg_path)
                
                # Save with original name
                orig_img.save(os.path.join(self.output_original_dir, orig_filename))
                seg_img.save(os.path.join(self.output_segmented_dir, orig_filename))  # Use same name for segmented
        
        # Generate augmented images
        print(f"\nGenerating {augmentations_per_image} augmented versions per image...")
        total_augmented = 0
        
        for orig_path, seg_path in tqdm(pairs, desc="Augmenting"):
            # Load images
            orig_img = Image.open(orig_path).convert('RGB')
            seg_img = Image.open(seg_path).convert('L')
            
            # Resize to target size if needed
            if orig_img.size != self.image_size:
                orig_img = orig_img.resize(self.image_size, Image.BILINEAR)
            if seg_img.size != self.image_size:
                seg_img = seg_img.resize(self.image_size, Image.NEAREST)
            
            # Select random augmentations
            selected_augs = random.sample(augmentation_types, min(augmentations_per_image, len(augmentation_types)))
            
            for aug_type in selected_augs:
                # Apply augmentation
                aug_orig, aug_seg = self.apply_augmentations(orig_img.copy(), seg_img.copy(), aug_type)
                
                # Generate filename with 4-digit zero-padded index
                filename = f"{current_index:04d}.png"
                
                # Save augmented images
                aug_orig.save(os.path.join(self.output_original_dir, filename))
                aug_seg.save(os.path.join(self.output_segmented_dir, filename))
                
                current_index += 1
                total_augmented += 1
        
        print(f"\nAugmentation complete!")
        print(f"Total original pairs: {len(pairs)}")
        print(f"Total augmented images generated: {total_augmented}")
        print(f"Final image index: {current_index - 1:04d}")
        print(f"Output directories:")
        print(f"  - Original: {os.path.abspath(self.output_original_dir)}")
        print(f"  - Segmented: {os.path.abspath(self.output_segmented_dir)}")


def main():
    parser = argparse.ArgumentParser(description='Data Augmentation for FIVES Dataset')
    
    # Input directories (default to Google Drive paths)
    parser.add_argument('--original-dir', type=str, 
                        default=r'D:\DRIVE\FIVES\Original\PNG',
                        help='Path to original images directory')
    parser.add_argument('--segmented-dir', type=str,
                        default=r'D:\DRIVE\FIVES\Segmented\PNG',
                        help='Path to segmented images directory')
    
    # Output directories (in repo, expects Cloud/local setup)
    parser.add_argument('--output-original-dir', type=str,
                        default='./Original',
                        help='Output directory for augmented original images')
    parser.add_argument('--output-segmented-dir', type=str,
                        default='./Segmented',
                        help='Output directory for augmented segmented images')
    
    # Augmentation parameters
    parser.add_argument('--start-index', type=int, default=801,
                        help='Starting index for augmented image naming (default: 801)')
    parser.add_argument('--augmentations-per-image', type=int, default=5,
                        help='Number of augmented versions per original image (default: 5)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[1024, 1024],
                        help='Target image size as width height (default: 1024 1024)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-copy-originals', action='store_true',
                        help='Do not copy original images to output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FIVES Dataset Augmentation Preprocessing")
    print("=" * 60)
    print(f"Original images: {args.original_dir}")
    print(f"Segmented images: {args.segmented_dir}")
    print(f"Output original: {args.output_original_dir}")
    print(f"Output segmented: {args.output_segmented_dir}")
    print(f"Starting index: {args.start_index:04d}")
    print(f"Augmentations per image: {args.augmentations_per_image}")
    print(f"Target image size: {args.image_size[0]}x{args.image_size[1]}")
    print("=" * 60)
    
    augmentor = DataAugmentor(
        original_dir=args.original_dir,
        segmented_dir=args.segmented_dir,
        output_original_dir=args.output_original_dir,
        output_segmented_dir=args.output_segmented_dir,
        start_index=args.start_index,
        image_size=tuple(args.image_size),
        seed=args.seed
    )
    
    augmentor.process(
        augmentations_per_image=args.augmentations_per_image,
        copy_originals=not args.no_copy_originals
    )


if __name__ == '__main__':
    main()
