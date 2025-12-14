import json
import os
import subprocess
from pathlib import Path
from datetime import datetime


PROJECT_ID = "F202500016INOVIAMOMENTAOMINUTO"
OUTPUTS_BASE = f"/projects/{PROJECT_ID}/outputs"
SLURM_SCRIPT = "jobs/isaac_train.slurm"


def submit(run_id: str, payload: dict):
    """
    Entry point called by the backend when engine == isaac_rl
    """

    # ------------------------------------------------------------------
    # 1) Prepare output directory
    # ------------------------------------------------------------------
    run_dir = Path(OUTPUTS_BASE) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Normalize payload
    # ------------------------------------------------------------------
    config = {
        "run_id": run_id,
        "engine": "isaac_rl",
        "algorithm": payload.get("algorithm"),
        "environment": payload.get("environment"),
        "params": payload.get("params", {}),
        "resources": payload.get("resources", {}),
        "created_at": datetime.utcnow().isoformat(),
        "outputs_dir": str(run_dir)
    }

    # Defaults safety (backend stays dumb)
    config["resources"].setdefault("partition", "dev-a100-40")
    config["resources"].setdefault("gpus", 1)
    config["resources"].setdefault("cpus", 16)
    config["resources"].setdefault("mem_gb", 64)
    config["resources"].setdefault("time_limit_min", 240)
    config["resources"].setdefault("gpu_type", "a100")

    # ------------------------------------------------------------------
    # 3) Write train_config.json
    # ------------------------------------------------------------------
    config_path = run_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # 4) Submit SLURM job
    # ------------------------------------------------------------------
    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "CONFIG_PATH": str(config_path),
        "OUTPUTS_DIR": str(run_dir)
    })

    cmd = [
        "sbatch",
        SLURM_SCRIPT
    ]

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to submit Isaac RL job: {result.stderr}"
        )

    return {
        "status": "submitted",
        "run_id": run_id,
        "slurm_output": result.stdout.strip()
    }

