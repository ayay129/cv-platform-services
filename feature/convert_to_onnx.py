import argparse
import os
import sys

import torch
import timm


def build_features_module(checkpoint_path: str, truncate_last_blocks: int = 5, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """
    构建并返回用于特征导出的子模块：
    - 使用 timm 加载 `tf_efficientnet_b7_ns` 模型，并加载指定 checkpoint。
    - 通过裁剪 children() 的后若干层（默认 5 层）得到中间特征模块，和原有 PyTorch 版本保持一致。
    """
    model = timm.create_model(
        "tf_efficientnet_b7_ns", pretrained=False, checkpoint_path=checkpoint_path
    )
    features = torch.nn.Sequential(*list(model.children())[:-truncate_last_blocks])
    features.eval()
    features.to(device)
    return features


def export_onnx(
    checkpoint_path: str,
    onnx_output_path: str,
    image_size: int = 600,
    opset: int = 13,
    use_dynamic_axes: bool = False,
    truncate_last_blocks: int = 5,
    use_half: bool = False,
):
    """
    将 EfficientNet-B7(NS) 的 pth checkpoint 导出为 ONNX：
    - 导出的计算图是中间特征模块（裁剪后），输出通常为 4D 特征图，利于下游做全局平均池化。
    - 默认输入尺寸为 600x600（与现有项目一致）。
    - 支持启用动态维度（batch/height/width），默认关闭以保证推理稳定性。
    """
    device = torch.device("cpu")
    features = build_features_module(
        checkpoint_path=checkpoint_path,
        truncate_last_blocks=truncate_last_blocks,
        device=device,
    )

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    if use_half:
        # 半精度导出在某些算子上可能不兼容，谨慎启用
        features.half()
        dummy = dummy.half()

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if use_dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        }

    os.makedirs(os.path.dirname(onnx_output_path) or ".", exist_ok=True)

    print(f"[ONNX Export] checkpoint: {checkpoint_path}")
    print(f"[ONNX Export] output: {onnx_output_path}")
    print(f"[ONNX Export] image_size: {image_size}, opset: {opset}, dynamic_axes: {bool(dynamic_axes)}")

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

    print("[ONNX Export] Done.")

    # 合并外部数据为单文件，避免生成 .onnx.data
    try:
        import onnx
        m = onnx.load(onnx_output_path, load_external_data=True)
        onnx.save_model(m, onnx_output_path, save_as_external_data=False)
        data_path = onnx_output_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
        print("[ONNX Export] External data merged; single-file ONNX ready.")
    except Exception as e:
        print(f"[ONNX Export] External data merge skipped or failed: {e}")

    # 可选：简单验证 ONNX 是否能被加载与运行
    try:
        import numpy as np
        import onnxruntime as ort

        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in set(ort.get_available_providers())]
        print(f"[ONNX Verify] providers: {providers}")
        sess = ort.InferenceSession(onnx_output_path, providers=providers or None)
        input_name = sess.get_inputs()[0].name
        x = dummy.cpu().numpy().astype(np.float32)
        outputs = sess.run(None, {input_name: x})
        y = outputs[0]
        print(f"[ONNX Verify] output shape: {y.shape}")
    except Exception as e:
        print(f"[ONNX Verify] Skipped or failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert EfficientNet-B7(NS) checkpoint (.pth) to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="resources/tf_efficientnet_b7_ns-1dbc32de.pth",
        help="pth checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        default="resources/efnet_b7.onnx",
        help="导出的 ONNX 文件路径",
    )
    parser.add_argument("--image_size", type=int, default=600, help="导出时的输入尺寸（方形）")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本")
    parser.add_argument("--dynamic", action="store_true", help="是否为输入启用动态维度")
    parser.add_argument("--truncate_last_blocks", type=int, default=5, help="裁剪模型 children() 的后几层")
    parser.add_argument("--half", action="store_true", help="是否使用半精度导出（不建议）")

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        onnx_output_path=args.output,
        image_size=args.image_size,
        opset=args.opset,
        use_dynamic_axes=args.dynamic,
        truncate_last_blocks=args.truncate_last_blocks,
        use_half=args.half,
    )


if __name__ == "__main__":
    main()
