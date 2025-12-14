#!/bin/bash
#SBATCH -A F202500016INOVIAMOMENTAOMINUTOG
#SBATCH -p normal-a100-80
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH --job-name=train_model
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

cd /projects/F202500016INOVIAMOMENTAOMINUTO

RUN_ID=${RUN_ID:-run_0001}
EPOCHS=${EPOCHS:-20}

singularity run \
  --nv \
  -B /projects:/projects \
  /share/apps-x86/containers/singularity/pytorch_2.6.0-cuda12.4-cudnn9-runtime.sif \
  python scripts/train.py \
    --run_id ${RUN_ID} \
    --data datasets_processed/default \
    --epochs ${EPOCHS}
