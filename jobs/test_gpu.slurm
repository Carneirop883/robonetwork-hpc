#!/bin/bash
#SBATCH -A F202500016INOVIAMOMENTAOMINUTOG
#SBATCH -p normal-a100-80
#SBATCH --gpus=1
#SBATCH --job-name=test_gpu
#SBATCH -t 00:05:00
#SBATCH -o /projects/F202500016INOVIAMOMENTAOMINUTO/logs/test_gpu_%j.out
#SBATCH -e /projects/F202500016INOVIAMOMENTAOMINUTO/logs/test_gpu_%j.err

echo "===== NODE INFO ====="
hostname
echo
echo "===== NVIDIA-SMI ====="
nvidia-smi
