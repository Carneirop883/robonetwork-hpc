#!/bin/bash
#SBATCH -A F202500016INOVIAMOMENTAOMINUTOG
#SBATCH -p normal-a100-80
#SBATCH --gpus=1
#SBATCH -t 00:05:00
#SBATCH --job-name=test_pytorch_container
#SBATCH -o /projects/F202500016INOVIAMOMENTAOMINUTO/logs/test_gpu_%j.out
#SBATCH -e /projects/F202500016INOVIAMOMENTAOMINUTO/logs/test_gpu_%j.err

cd /projects/F202500016INOVIAMOMENTAOMINUTO

singularity run \
  --nv \
  -B /projects:/projects \
  /share/apps-x86/containers/singularity/pytorch_2.6.0-cuda12.4-cudnn9-runtime.sif \
  python test_torch_gpu.py
