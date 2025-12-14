import argparse
import os
import torch
import torch.nn as nn

print(">>> export_model.py started")

def build_model():
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--opset", type=int, default=13)  # 🔧 compatível com torch 1.10
    args = parser.parse_args()

    print(f">>> run_id = {args.run_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> device = {device}")

    model_path = f"models/{args.run_id}/model.pt"
    print(f">>> loading model from {model_path}")
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 128, device=device)

    os.makedirs("models/exported", exist_ok=True)
    onnx_path = f"models/exported/{args.run_id}.onnx"

    print(f">>> exporting to ONNX (opset {args.opset})")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=args.opset
    )

    print(f">>> ONNX export complete")
    print(f">>> saved to: {onnx_path}")

if __name__ == "__main__":
    main()

