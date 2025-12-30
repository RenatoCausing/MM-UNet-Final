"""
Complete Pipeline Script for FIVES Dataset Training and Testing

This script handles:
1. Data splitting (train/val/test)
2. Computing normalization stats from TRAINING SET ONLY (no data leakage)
3. Training the model
4. Testing and evaluation

Usage:
    python run_pipeline.py --mode train
    python run_pipeline.py --mode test --checkpoint ./model_store/MM_Net/best
    python run_pipeline.py --mode all  # Train then test
"""

import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60 + '\n')
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✓ {description} completed successfully\n")
    return result


def main():
    parser = argparse.ArgumentParser(description='FIVES Training/Testing Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'all'], default='all',
                        help='Mode: train, test, or all (train then test)')
    parser.add_argument('--checkpoint', type=str, default='./model_store/MM_Net/best',
                        help='Checkpoint path for testing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save-vis', action='store_true', help='Save test visualizations')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("="*60)
    print("FIVES Dataset - Complete Pipeline")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")
    
    if args.mode in ['train', 'all']:
        # Training
        run_command(
            'python train.py',
            'Model Training'
        )
    
    if args.mode in ['test', 'all']:
        # Testing
        vis_flag = '--save-vis' if args.save_vis else ''
        run_command(
            f'python test.py --checkpoint "{args.checkpoint}" {vis_flag}',
            'Model Testing'
        )
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
