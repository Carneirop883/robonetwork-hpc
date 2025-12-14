#!/bin/bash
#SBATCH -A F202500016INOVIAMOMENTAOMINUTOG
#SBATCH -p normal-x86
#SBATCH -t 00:20:00
#SBATCH --job-name=create_torch_env
#SBATCH -o logs/create_env_%j.out
#SBATCH -e logs/create_env_%j.err

set -e

cd /projects/F202500016INOVIAMOMENTAOMINUTO

mkdir -p .mamba logs

# micromamba standalone
export PATH=$PWD/tools/bin:$PATH

# root isolado (HPC-safe)
export MAMBA_ROOT_PREFIX=$PWD/.mamba
export CONDARC=/dev/null

echo "Micromamba version:"
micromamba --version

echo "Creating torch_env..."

# criar ambiente
micromamba create -y -n torch_env python=3.10 pip

echo "Upgrading pip..."
micromamba run -n torch_env pip install --upgrade pip

echo "Installing PyTorch (CUDA 12.1 wheels)..."
micromamba run -n torch_env pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "Sanity check (CPU node):"
micromamba run -n torch_env python - << 'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Environment successfully created."

