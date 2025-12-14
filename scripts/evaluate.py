import argparse
import json
import os
import torch
import torch.nn as nn
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = f"models/{args.run_id}/model.pt"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    # Dummy validation data (substituível por dados reais depois)
    x_val = torch.randn(512, 128, device=device)
    y_val = torch.randint(0, 10, (512,), device=device)

    with torch.no_grad():
        logits = model(x_val)
        loss = loss_fn(logits, y_val)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_val).float().mean().item()

    result = {
        "run_id": args.run_id,
        "evaluated_at": datetime.utcnow().isoformat(),
        "device": device,
        "metrics": {
            "loss": float(loss.item()),
            "accuracy": accuracy
        }
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.run_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{args.run_id}] Evaluation complete")
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
