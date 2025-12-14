import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = f"models/{args.run_id}"
    ckpt_dir = f"checkpoints/{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(1024, 128, device=device)
    y = torch.randint(0, 10, (1024,), device=device)

    history = []

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        history.append({"epoch": epoch, "loss": float(loss.item())})
        print(f"[{args.run_id}] Epoch {epoch}/{args.epochs} | Loss {loss.item():.4f}", flush=True)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{ckpt_dir}/epoch_{epoch}.pt")

    torch.save(model.state_dict(), f"{run_dir}/model.pt")

    with open(f"runs/{args.run_id}.json", "w") as f:
        json.dump({
            "run_id": args.run_id,
            "epochs": args.epochs,
            "finished_at": datetime.utcnow().isoformat(),
            "device": device,
            "history": history
        }, f, indent=2)

if __name__ == "__main__":
    main()
