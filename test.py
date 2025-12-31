"""
Automated Testing Script for FIVES Dataset

This script loads a trained model and evaluates it on the test set.
It computes metrics (Dice, IoU, F1, Precision, Recall, Accuracy) and 
optionally saves visualizations.

Usage:
    python test.py --checkpoint ./model_store/MM_Net/best
    python test.py --checkpoint ./model_store/MM_Net/best --save-vis
    python test.py --checkpoint ./model_store/MM_Net/best --batch-size 1
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import argparse
from typing import Dict, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import monai
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from tqdm import tqdm

from src import utils
from src.models import give_model
from src.FIVESLoader import get_dataloader

import warnings
warnings.filterwarnings('ignore')


class TestMetrics:
    """Class to compute and store test metrics."""
    
    def __init__(self, include_background: bool = True):
        self.include_background = include_background
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.accuracy_scores = []
        self.specificity_scores = []
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predicted binary mask (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)
        """
        pred = pred.float()
        target = target.float()
        
        # Flatten for metric computation
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        for i in range(pred.size(0)):
            p = pred_flat[i]
            t = target_flat[i]
            
            # True positives, false positives, false negatives, true negatives
            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()
            tn = ((1 - p) * (1 - t)).sum().item()
            
            # Dice coefficient
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            self.dice_scores.append(dice)
            
            # IoU (Jaccard)
            iou = tp / (tp + fp + fn + 1e-8)
            self.iou_scores.append(iou)
            
            # Precision
            precision = tp / (tp + fp + 1e-8)
            self.precision_scores.append(precision)
            
            # Recall (Sensitivity)
            recall = tp / (tp + fn + 1e-8)
            self.recall_scores.append(recall)
            
            # F1 Score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            self.f1_scores.append(f1)
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            self.accuracy_scores.append(accuracy)
            
            # Specificity
            specificity = tn / (tn + fp + 1e-8)
            self.specificity_scores.append(specificity)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        return {
            'Dice': np.mean(self.dice_scores),
            'IoU': np.mean(self.iou_scores),
            'F1': np.mean(self.f1_scores),
            'Precision': np.mean(self.precision_scores),
            'Recall': np.mean(self.recall_scores),
            'Accuracy': np.mean(self.accuracy_scores),
            'Specificity': np.mean(self.specificity_scores),
            'Dice_std': np.std(self.dice_scores),
            'IoU_std': np.std(self.iou_scores),
        }


def denormalize_image(tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized numpy array (H, W, C) in range [0, 255]
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp and convert to uint8
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    return (np_img * 255).astype(np.uint8)


def save_visualization(
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    mean: list,
    std: list,
    image_name: str = ""
):
    """
    Save visualization of image, ground truth, and prediction.
    
    Args:
        image: Input image tensor (C, H, W)
        mask: Ground truth mask (1, H, W)
        pred: Predicted mask (1, H, W)
        save_path: Path to save the visualization
        mean: Normalization mean for denormalization
        std: Normalization std for denormalization
        image_name: Optional image name for title
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Denormalize image
    img_np = denormalize_image(image, mean, std)
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    mask_np = mask.squeeze().cpu().numpy()
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    pred_np = pred.squeeze().cpu().numpy()
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay (prediction on original)
    overlay = img_np.copy()
    # Red channel for false negatives (missed)
    overlay[:, :, 0] = np.where((mask_np > 0.5) & (pred_np < 0.5), 255, overlay[:, :, 0])
    # Green channel for true positives
    overlay[:, :, 1] = np.where((mask_np > 0.5) & (pred_np > 0.5), 255, overlay[:, :, 1])
    # Blue channel for false positives
    overlay[:, :, 2] = np.where((mask_np < 0.5) & (pred_np > 0.5), 255, overlay[:, :, 2])
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (G=TP, R=FN, B=FP)')
    axes[3].axis('off')
    
    if image_name:
        fig.suptitle(f'Test Result: {image_name}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    accelerator: Accelerator,
    save_visualizations: bool = False,
    output_dir: str = "./test_results"
) -> Dict[str, float]:
    """
    Test the model on the test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config: Configuration
        accelerator: Accelerator for device management
        save_visualizations: Whether to save visualization images
        output_dir: Directory to save results
        
    Returns:
        Dictionary of test metrics
    """
    model.eval()
    
    # Get image size for inference
    image_size = config.dataset.FIVES.image_size
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    
    # Create sliding window inferer for large images
    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size[0], 2),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device
    )
    
    # Post-processing transforms
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    # Metrics
    metrics = TestMetrics(include_background=True)
    
    # Normalization params for denormalization (computed from training set only)
    # These are loaded from the saved stats file during get_dataloader()
    mean = config.dataset.FIVES.get('computed_mean', config.dataset.FIVES.image_mean)
    std = config.dataset.FIVES.get('computed_std', config.dataset.FIVES.image_std)
    
    # Create output directory
    if save_visualizations:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nTesting on {len(test_loader)} batches...")
    print(f"Using normalization - Mean: {mean}, Std: {std}")
    
    all_predictions = []
    all_image_paths = []
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        images, masks, img_paths, mask_paths = batch
        
        # Run inference
        logits = inference(images, model)
        predictions = post_trans(logits)
        
        # Update metrics
        metrics.update(predictions, masks)
        
        # Save visualizations
        if save_visualizations:
            for i in range(images.size(0)):
                img_name = os.path.basename(img_paths[i])
                save_path = os.path.join(vis_dir, f"test_{img_name}")
                save_visualization(
                    images[i], masks[i], predictions[i],
                    save_path, mean, std, img_name
                )
        
        # Store predictions
        for i in range(predictions.size(0)):
            all_predictions.append(predictions[i].cpu().numpy())
            all_image_paths.append(img_paths[i])
    
    # Compute final metrics
    results = metrics.compute()
    
    # Save predictions as numpy
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    for pred, path in zip(all_predictions, all_image_paths):
        img_name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(pred_dir, f"{img_name}.npy"), pred)
    
    return results


def print_results(results: Dict[str, float]):
    """Print formatted test results."""
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>15} {'Std':>15}")
    print("-" * 60)
    print(f"{'Dice Coefficient':<20} {results['Dice']:>15.4f} {results['Dice_std']:>15.4f}")
    print(f"{'IoU (Jaccard)':<20} {results['IoU']:>15.4f} {results['IoU_std']:>15.4f}")
    print(f"{'F1 Score':<20} {results['F1']:>15.4f}")
    print(f"{'Precision':<20} {results['Precision']:>15.4f}")
    print(f"{'Recall':<20} {results['Recall']:>15.4f}")
    print(f"{'Accuracy':<20} {results['Accuracy']:>15.4f}")
    print(f"{'Specificity':<20} {results['Specificity']:>15.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Test trained model on FIVES dataset')
    parser.add_argument('--checkpoint', type=str, default='./model_store/MM_Net/best',
                        help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to config file')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Load config
    config = EasyDict(yaml.load(open(args.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # Override batch size if specified
    if args.batch_size:
        config.dataset.FIVES.batch_size = args.batch_size
    
    # Initialize accelerator
    accelerator = Accelerator(cpu=False)
    
    print("=" * 60)
    print("FIVES Dataset - Automated Testing")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {accelerator.device}")
    
    # Load model
    print("\nLoading model...")
    model = give_model(config)
    
    # Load dataloader (we only need test_loader)
    print("\nLoading test data...")
    train_loader, val_loader, test_loader = get_dataloader(config)
    
    if test_loader is None:
        print("ERROR: No test data found!")
        sys.exit(1)
    
    # Prepare model
    model = accelerator.prepare(model)
    
    # Load checkpoint (if it exists)
    if os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        try:
            accelerator.load_state(args.checkpoint)
            print("Checkpoint loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load full accelerator state: {e}")
            print("Trying to load model weights directly...")
            
            # Try loading model weights directly
            model_path = os.path.join(args.checkpoint, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=accelerator.device)
                model.load_state_dict(state_dict)
                print("Model weights loaded successfully!")
            else:
                print(f"Warning: Could not find model weights at {model_path}")
                print("Proceeding with untrained model (not recommended for testing)...")
    else:
        print(f"\nWarning: Checkpoint path '{args.checkpoint}' does not exist.")
        print("Proceeding with untrained model (not recommended for testing)...")
        print("To test a trained model, first run training or specify a valid checkpoint path.")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run testing
    results = test_model(
        model=model,
        test_loader=test_loader,
        config=config,
        accelerator=accelerator,
        save_visualizations=args.save_vis,
        output_dir=output_dir
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    results_file = os.path.join(output_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write("FIVES Dataset Test Results\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 40 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {os.path.join(output_dir, 'predictions')}")
    if args.save_vis:
        print(f"Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")


if __name__ == '__main__':
    main()
