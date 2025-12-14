import argparse
import json
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    outputs_dir = Path(args.outputs)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "logs").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Write a basic metrics.json
    metrics = {
        "run_id": cfg.get("run_id"),
        "engine": cfg.get("engine"),
        "algorithm": cfg.get("algorithm"),
        "environment": cfg.get("environment"),
        "status": "completed_mvp",
        "timestamp": datetime.utcnow().isoformat(),
        "notes": "MVP runner executed successfully. Isaac Gym training not enabled yet."
    }

    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Create a dummy checkpoint file to validate artifacts pipeline
    ckpt = outputs_dir / "checkpoints" / "model_dummy.pt"
    ckpt.write_bytes(b"DUMMY_CHECKPOINT")

    print("Runner OK. Wrote metrics.json and dummy checkpoint.")

if __name__ == "__main__":
    main()

