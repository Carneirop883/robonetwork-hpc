#!/bin/bash
#SBATCH -A F202500016INOVIAMOMENTAOMINUTOG
#SBATCH -p normal-a100-80
#SBATCH --gpus=1
#SBATCH --job-name=build_container
#SBATCH -t 00:10:00
#SBATCH -o /projects/F202500016INOVIAMOMENTAOMINUTO/logs/build_base_%j.out
#SBATCH -e /projects/F202500016INOVIAMOMENTAOMINUTO/logs/build_base_%j.err

echo "=== STARTING BASE CONTAINER BUILD ==="

CONTAINERS=/projects/F202500016INOVIAMOMENTAOMINUTO/containers
mkdir -p $CONTAINERS

echo "Using Singularity:"
which singularity
singularity --version

echo "Pulling NVIDIA CUDA 12.4.1 + cuDNN9 (DEVEL)..."
/usr/bin/singularity pull \
    $CONTAINERS/cuda_base.sif \
    docker://nvcr.io/nvidia/cuda:12.4.1-cudnn9-devel

echo "=== BUILD COMPLETED ==="

