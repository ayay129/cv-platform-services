import argparse
import os
from typing import Dict

import timm
import torch


def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def load_state_dict(pth_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")
    return strip_prefix(ckpt, "module.")


def build_features_module(checkpoint_path: str, truncate_last_blocks: int) -> torch.nn.Module:
    model = timm.create_model(
        "tf_efficientnet_b7_ns",
        pretrained=False,
        checkpoint_path=checkpoint_path,
    )
    features = torch.nn.Sequential(*list(model.children())[:-truncate_last_blocks])
    features.eval()
    return features


def export_onnx(
    checkpoint_path: str,
    onnx_output_path: str,
    image_size: int,
    opset: int,
    truncate_last_blocks: int,
    dynamic: bool,
) -> None:
    features = build_features_module(checkpoint_path, truncate_last_blocks)
    dummy = torch.randn(1, 3, image_size, image_size)

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        }

    os.makedirs(os.path.dirname(onnx_output_path) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            features,
            dummy,
            onnx_output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--truncate-last-blocks", type=int, default=5)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    state_dict = load_state_dict(args.pth)
    model = build_features_module(args.pth, args.truncate_last_blocks)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    export_onnx(
        checkpoint_path=args.pth,
        onnx_output_path=args.onnx,
        image_size=args.image_size,
        opset=args.opset,
        truncate_last_blocks=args.truncate_last_blocks,
        dynamic=args.dynamic,
    )
    print(f"Exported: {args.onnx}")


if __name__ == "__main__":
    main()
