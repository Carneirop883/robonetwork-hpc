#!/bin/bash
#SBATCH --job-name=isaac_rl
#SBATCH --partition=dev-a100-40
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/projects/F202500016INOVIAMOMENTAOMINUTO/outputs/%x_%j.out
#SBATCH --error=/projects/F202500016INOVIAMOMENTAOMINUTO/outputs/%x_%j.err

set -euo pipefail

echo "=== ISAAC RL SLURM JOB START ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

echo "RUN_ID: ${RUN_ID:-}"
echo "CONFIG_PATH: ${CONFIG_PATH:-}"
echo "OUTPUTS_DIR: ${OUTPUTS_DIR:-}"

# ------------------------------------------------------------------
# 1) Validate required environment variables
# ------------------------------------------------------------------
if [[ -z "${RUN_ID:-}" || -z "${CONFIG_PATH:-}" || -z "${OUTPUTS_DIR:-}" ]]; then
  echo "ERROR: Missing required variables:"
  echo "RUN_ID, CONFIG_PATH and OUTPUTS_DIR must be set"
  exit 1
fi

# ------------------------------------------------------------------
# 2) Prepare output directories
# ------------------------------------------------------------------
mkdir -p "${OUTPUTS_DIR}/logs"
mkdir -p "${OUTPUTS_DIR}/checkpoints"

# ------------------------------------------------------------------
# 3) Debug info
# ------------------------------------------------------------------
echo "Working directory:"
pwd

echo "Repository contents:"
ls -la

# ------------------------------------------------------------------
# 4) Singularity sanity check (Deucalion requirement)
# ------------------------------------------------------------------
if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity not found on compute node"
  exit 1
fi

echo "Singularity version:"
singularity --version

# ------------------------------------------------------------------
# 5) Execute Isaac Gym runner (MVP mode)
# ------------------------------------------------------------------
echo "Launching Isaac Gym runner..."
python3 runners/isaac_gym_runner.py \
  --config "${CONFIG_PATH}" \
  --outputs "${OUTPUTS_DIR}" \
  2>&1 | tee "${OUTPUTS_DIR}/logs/train.log"

# ------------------------------------------------------------------
# 6) Job end
# ------------------------------------------------------------------
echo "=== ISAAC RL SLURM JOB END ==="
echo "Date: $(date)"

