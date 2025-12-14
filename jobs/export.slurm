#!/bin/bash
#SBATCH -A F202500016INOVIAMOMENTAOMINUTOG
#SBATCH -p normal-a100-80
#SBATCH --gpus=1
#SBATCH -t 00:10:00
#SBATCH --job-name=export_model
#SBATCH -o logs/export_%j.out
#SBATCH -e logs/export_%j.err

cd /projects/F202500016INOVIAMOMENTAOMINUTO

RUN_ID=${RUN_ID:-run_0001}

singularity run \
  --nv \
  -B /projects:/projects \
  /share/apps-x86/containers/singularity/pytorch_2.6.0-cuda12.4-cudnn9-runtime.sif \
  python scripts/export_model.py \
    --run_id ${RUN_ID}
