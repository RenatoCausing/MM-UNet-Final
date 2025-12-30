"""
Complete Installation Script for FIVES Training Environment
Designed for Vast.ai with PyTorch template (Python 3.10.x)

Usage:
    python3.10 install_requirements.py
    
IMPORTANT: This script will REPLACE the existing PyTorch with a compatible version!
"""

import subprocess
import sys
import os

def run_command(cmd, description="", ignore_error=False):
    """Run shell command and print output."""
    if description:
        print(f"\n{'='*60}")
        print(f"➤ {description}")
        print('='*60)
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0 and not ignore_error:
        print(f"⚠ Warning: Command returned code {result.returncode}")
    return result.returncode

def main():
    print("="*60)
    print("FIVES Training Environment - Complete Installation")
    print("For Vast.ai / Cloud with Python 3.10.x")
    print("="*60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Get the python executable
    python = sys.executable
    pip = f"{python} -m pip"
    
    # Step 1: Set CUDA environment variables
    print("\n" + "="*60)
    print("Step 1: Setting CUDA Environment")
    print("="*60)
    
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"
    os.environ["FORCE_CUDA"] = "1"
    os.environ["MAX_JOBS"] = "4"
    
    print("✓ CUDA_HOME=/usr/local/cuda")
    print("✓ TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0")
    
    # Step 2: Upgrade pip
    run_command(
        f"{pip} install --upgrade pip wheel",
        "Step 2: Upgrading pip"
    )
    
    # Step 3: COMPLETELY UNINSTALL existing PyTorch (critical for Vast.ai templates)
    print("\n" + "="*60)
    print("Step 3: Removing existing PyTorch (to avoid version conflicts)")
    print("="*60)
    run_command(f"{pip} uninstall -y torch torchvision torchaudio", ignore_error=True)
    run_command(f"{pip} uninstall -y torch torchvision torchaudio", ignore_error=True)  # Run twice to be sure
    
    # Step 4: Install PyTorch 2.3.0 with CUDA 12.1 (best mamba compatibility)
    # This version has pre-built mamba wheels available!
    run_command(
        f"{pip} install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121",
        "Step 4: Installing PyTorch 2.3.0 with CUDA 12.1"
    )
    
    # Verify PyTorch
    print("\nVerifying PyTorch installation...")
    run_command(
        f'{python} -c "import torch; print(f\'PyTorch: {{torch.__version__}}\'); print(f\'CUDA available: {{torch.cuda.is_available()}}\'); print(f\'GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}\')"',
        ignore_error=True
    )
    
    # Step 5: Install packaging and ninja (needed for Mamba compilation)
    run_command(
        f"{pip} install packaging ninja numpy",
        "Step 5: Installing build tools"
    )
    
    # Step 6: Install main ML packages
    main_packages = [
        "timm==0.9.12",  # Compatible with PyTorch 2.3
        "monai",
        "accelerate>=0.20.0",
        "einops",
    ]
    run_command(
        f"{pip} install {' '.join(main_packages)}",
        "Step 6: Installing ML libraries (timm, monai, accelerate)"
    )
    
    # Step 7: Install data processing packages
    data_packages = [
        "pandas",
        "scikit-learn",
        "opencv-python",
        "Pillow",
        "tqdm",
    ]
    run_command(
        f"{pip} install {' '.join(data_packages)}",
        "Step 7: Installing data processing packages"
    )
    
    # Step 8: Install utility packages
    util_packages = [
        "easydict",
        "pyyaml",
        "objprint",
        "openpyxl",
        "matplotlib",
        "tensorboard",
        "seaborn",
    ]
    run_command(
        f"{pip} install {' '.join(util_packages)}",
        "Step 8: Installing utility packages"
    )
    
    # Step 9: Install medical imaging packages
    run_command(
        f"{pip} install SimpleITK nibabel",
        "Step 9: Installing medical imaging packages"
    )
    
    # Step 10: Install Mamba - use EXACT versions known to work with PyTorch 2.3 + CUDA 12.1
    print("\n" + "="*60)
    print("Step 10: Installing Mamba SSM (this may take a few minutes)")
    print("="*60)
    
    mamba_installed = False
    
    # Method 1: Try specific versions that have pre-built wheels for torch 2.3 + cu121
    print("\nTrying Method 1: Pre-built wheels for PyTorch 2.3 + CUDA 12.1...")
    
    # First install causal-conv1d
    ret1 = run_command(
        f"{pip} install causal-conv1d==1.2.2.post1",
        ignore_error=True
    )
    
    # Then install mamba-ssm
    ret2 = run_command(
        f"{pip} install mamba-ssm==2.0.4",
        ignore_error=True
    )
    
    # Verify
    ret_verify = run_command(
        f'{python} -c "from mamba_ssm import Mamba; print(\'✓ Mamba installed successfully!\')"',
        ignore_error=True
    )
    if ret_verify == 0:
        mamba_installed = True
    
    # Method 2: Try older stable versions
    if not mamba_installed:
        print("\nMethod 1 failed. Trying Method 2: Older stable versions...")
        run_command(f"{pip} uninstall -y causal-conv1d mamba-ssm", ignore_error=True)
        
        run_command(f"{pip} install causal-conv1d==1.1.3.post1", ignore_error=True)
        run_command(f"{pip} install mamba-ssm==1.2.2", ignore_error=True)
        
        ret_verify = run_command(
            f'{python} -c "from mamba_ssm import Mamba; print(\'✓ Mamba v1.2.2 installed!\')"',
            ignore_error=True
        )
        if ret_verify == 0:
            mamba_installed = True
    
    # Method 3: Build from source with --no-build-isolation
    if not mamba_installed:
        print("\nMethod 2 failed. Trying Method 3: Build from source...")
        run_command(f"{pip} uninstall -y causal-conv1d mamba-ssm", ignore_error=True)
        
        run_command(
            f"{pip} install causal-conv1d --no-build-isolation",
            ignore_error=True
        )
        run_command(
            f"{pip} install mamba-ssm --no-build-isolation",
            ignore_error=True
        )
        
        ret_verify = run_command(
            f'{python} -c "from mamba_ssm import Mamba; print(\'✓ Mamba built from source!\')"',
            ignore_error=True
        )
        if ret_verify == 0:
            mamba_installed = True
    
    # Step 11: Final verification
    print("\n" + "="*60)
    print("Step 11: Final Verification")
    print("="*60)
    
    packages_to_verify = [
        ("torch", "import torch; print(f'✓ PyTorch {torch.__version__}')"),
        ("torchvision", "import torchvision; print(f'✓ TorchVision {torchvision.__version__}')"),
        ("monai", "import monai; print(f'✓ MONAI {monai.__version__}')"),
        ("timm", "import timm; print(f'✓ timm {timm.__version__}')"),
        ("accelerate", "import accelerate; print(f'✓ Accelerate {accelerate.__version__}')"),
        ("cv2", "import cv2; print(f'✓ OpenCV {cv2.__version__}')"),
        ("numpy", "import numpy; print(f'✓ NumPy {numpy.__version__}')"),
        ("mamba_ssm", "from mamba_ssm import Mamba; print('✓ Mamba SSM')"),
    ]
    
    all_success = True
    for name, cmd in packages_to_verify:
        ret = run_command(f'{python} -c "{cmd}"', ignore_error=True)
        if ret != 0:
            print(f"✗ {name} - NOT INSTALLED")
            if name == "mamba_ssm":
                print("  (Mamba is optional but required for some models)")
            all_success = False
    
    # Final GPU check
    print("\n" + "="*60)
    print("GPU Status")
    print("="*60)
    run_command(
        f'{python} -c "import torch; print(f\'CUDA Available: {{torch.cuda.is_available()}}\'); print(f\'GPU Count: {{torch.cuda.device_count()}}\'); print(f\'Current GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}\')"',
        ignore_error=True
    )
    
    # Summary
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE!")
    print("="*60)
    
    if mamba_installed:
        print("✓ All packages installed successfully!")
        print("\nYou can now run:")
        print("  python3.10 train.py")
    else:
        print("⚠ Most packages installed, but Mamba SSM failed.")
        print("  The model requires Mamba. Try running manually:")
        print("  pip install mamba-ssm==1.1.1 --no-build-isolation")
        print("\nOr check if there's a CUDA version mismatch.")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
