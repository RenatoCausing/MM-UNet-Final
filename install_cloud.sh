#!/bin/bash
# =============================================================================
# FIVES Training Environment - One-Click Cloud Installation
# For Vast.ai / RunPod / Lambda Labs / Any cloud with CUDA
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "FIVES Training - Cloud Installation Script"
echo "============================================================"

# Navigate to project directory
cd /workspace/MM-UNet-Final

# Step 1: Install Python 3.10 if not present
echo ""
echo "============================================================"
echo "Step 1: Checking Python 3.10..."
echo "============================================================"

if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    apt update
    apt install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt update
    apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
fi

python3.10 --version

# Step 2: Set CUDA environment
echo ""
echo "============================================================"
echo "Step 2: Setting CUDA environment..."
echo "============================================================"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export FORCE_CUDA=1
export MAX_JOBS=4

echo "CUDA_HOME=$CUDA_HOME"
nvcc --version || echo "nvcc not found, but continuing..."

# Step 3: Upgrade pip
echo ""
echo "============================================================"
echo "Step 3: Upgrading pip..."
echo "============================================================"

python3.10 -m pip install --upgrade pip wheel

# Step 4: Uninstall existing PyTorch (avoid conflicts)
echo ""
echo "============================================================"
echo "Step 4: Removing existing PyTorch..."
echo "============================================================"

python3.10 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python3.10 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Step 5: Install PyTorch 2.3.0 + CUDA 12.1 (best mamba compatibility)
echo ""
echo "============================================================"
echo "Step 5: Installing PyTorch 2.3.0 + CUDA 12.1..."
echo "============================================================"

python3.10 -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch
python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Step 6: Install build tools
echo ""
echo "============================================================"
echo "Step 6: Installing build tools..."
echo "============================================================"

python3.10 -m pip install packaging ninja numpy

# Step 7: Install ML libraries
echo ""
echo "============================================================"
echo "Step 7: Installing ML libraries..."
echo "============================================================"

python3.10 -m pip install timm==0.9.12 monai accelerate>=0.20.0 einops

# Step 8: Install data processing packages
echo ""
echo "============================================================"
echo "Step 8: Installing data processing packages..."
echo "============================================================"

python3.10 -m pip install pandas scikit-learn opencv-python Pillow tqdm

# Step 9: Install utility packages
echo ""
echo "============================================================"
echo "Step 9: Installing utility packages..."
echo "============================================================"

python3.10 -m pip install easydict pyyaml objprint openpyxl matplotlib tensorboard seaborn

# Step 10: Install medical imaging packages
echo ""
echo "============================================================"
echo "Step 10: Installing medical imaging packages..."
echo "============================================================"

python3.10 -m pip install SimpleITK nibabel

# Step 11: Install Mamba SSM
echo ""
echo "============================================================"
echo "Step 11: Installing Mamba SSM..."
echo "============================================================"

# MUST use --no-build-isolation so it uses the torch we just installed
python3.10 -m pip install causal-conv1d --no-build-isolation || true
python3.10 -m pip install mamba-ssm --no-build-isolation || true

# Verify mamba
if ! python3.10 -c "from mamba_ssm import Mamba" 2>/dev/null; then
    echo "First attempt failed, trying with specific versions..."
    python3.10 -m pip uninstall -y causal-conv1d mamba-ssm 2>/dev/null || true
    python3.10 -m pip install causal-conv1d==1.2.2.post1 --no-build-isolation || true
    python3.10 -m pip install mamba-ssm==2.0.4 --no-build-isolation || true
fi

# Step 12: Final verification
echo ""
echo "============================================================"
echo "Step 12: Final Verification"
echo "============================================================"

python3.10 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch FAILED"
python3.10 -c "import torchvision; print(f'✓ TorchVision {torchvision.__version__}')" || echo "✗ TorchVision FAILED"
python3.10 -c "import monai; print(f'✓ MONAI {monai.__version__}')" || echo "✗ MONAI FAILED"
python3.10 -c "import timm; print(f'✓ timm {timm.__version__}')" || echo "✗ timm FAILED"
python3.10 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" || echo "✗ OpenCV FAILED"
python3.10 -c "from mamba_ssm import Mamba; print('✓ Mamba SSM')" || echo "✗ Mamba FAILED"

# GPU check
echo ""
echo "============================================================"
echo "GPU Status"
echo "============================================================"
python3.10 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo ""
echo "============================================================"
echo "INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "To start training, run:"
echo "  cd /workspace/MM-UNet-Final"
echo "  python3.10 train.py"
echo ""
