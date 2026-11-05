"""CPU-only state-dict coverage check for the uploaded IML-ViT checkpoint."""

import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from iml_vit_model import iml_vit_model

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model = iml_vit_model()
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    shape_mismatch = [
        {"key": key, "model": list(model_state[key].shape),
         "checkpoint": list(state[key].shape)}
        for key in model_state if key in state
        and tuple(model_state[key].shape) != tuple(state[key].shape)
    ]
    result = {
        "repo": args.repo,
        "checkpoint": args.checkpoint,
        "checkpoint_bytes": os.path.getsize(args.checkpoint),
        "checkpoint_tensors": len(state),
        "model_tensors": len(model_state),
        "checkpoint_params": sum(t.numel() for t in state.values() if hasattr(t, "numel")),
        "model_params": sum(t.numel() for t in model_state.values()),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "strict_load": False,
    }
    if not missing and not unexpected and not shape_mismatch:
        model.load_state_dict(state, strict=True)
        result["strict_load"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
