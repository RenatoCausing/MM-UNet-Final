#!/bin/bash
# Complete Installation Script for FIVES Training Environment
# This installs all dependencies including PyTorch, MONAI, and Mamba

set -e  # Exit on error

echo "============================================="
echo "Installing FIVES Training Requirements"
echo "============================================="

# Check Python version
python --version

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8:
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (if you have newer GPU):
# pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install main requirements
echo "Installing main requirements..."
pip install timm==0.4.12
pip install objprint==0.2.3
pip install accelerate==0.18.0
pip install easydict
pip install pyyaml
pip install opencv-python
pip install openpyxl
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install tqdm
pip install Pillow

# Install medical imaging libraries
echo "Installing medical imaging libraries..."
pip install monai
pip install nibabel
pip install SimpleITK

# Install Jupyter (if needed)
echo "Installing Jupyter..."
pip install jupyter ipykernel

# Install Mamba dependencies (compiled from source)
echo "Installing Mamba dependencies..."

# Install causal-conv1d
cd requirements/Mamba/causal-conv1d
pip install -e .
cd ../../..

# Install mamba-ssm
cd requirements/Mamba/mamba
pip install -e .
cd ../../..

# Verify installations
echo ""
echo "============================================="
echo "Verifying installations..."
echo "============================================="
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'✓ TorchVision {torchvision.__version__}')"
python -c "import monai; print(f'✓ MONAI {monai.__version__}')"
python -c "import timm; print(f'✓ timm {timm.__version__}')"
python -c "import accelerate; print(f'✓ Accelerate {accelerate.__version__}')"
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python -c "from mamba_ssm import Mamba; print('✓ Mamba installed')" || echo "⚠ Mamba not installed (optional)"
python -c "import causal_conv1d; print('✓ Causal Conv1D installed')" || echo "⚠ Causal Conv1D not installed (optional)"

echo ""
echo "============================================="
echo "Installation completed!"
echo "============================================="
echo "You can now run: python train.py"
