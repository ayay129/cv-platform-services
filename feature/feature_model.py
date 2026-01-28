import base64
import glob
import io
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms

from .base_model_onnx import BaseOnnxModel


class EfficientNetB7Onnx(BaseOnnxModel):
    def __init__(
        self,
        onnx_model_path: str = "resources/tf_efficientnet_b7_ns-1dbc32de.onnx",
        providers: Optional[List[str]] = None,
    ):
        image_size = 600
        tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        super().__init__(
            onnx_model_path=onnx_model_path,
            transforms=tfms,
            providers=providers,
        )

    def decode(self, outputs: List[np.ndarray]) -> np.ndarray:
        # EfficientNet typically emits a 4D feature map before classifier head.
        # Apply global average pooling over H, W to get (C) vector.
        y = outputs[0]
        if y.ndim == 4:
            y = y.mean(axis=(2, 3))
        elif y.ndim == 2:
            # Already (N, C)
            pass
        else:
            # Fallback: flatten
            y = y.reshape((y.shape[0], -1))
        return y[0]


class EfficientNetB7AisBench:
    def __init__(
        self,
        om_path: Optional[str] = None,
        input_name: Optional[str] = None,
        input_shape: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        output_shape: Optional[str] = None,
        device_id: Optional[int] = None,
    ):
        om_dir = os.getenv("FEATURE_OM_DIR", "resources")
        if om_path is None:
            om_path = os.getenv("FEATURE_OM_PATH")
        if om_path is None:
            candidates = sorted(glob.glob(os.path.join(om_dir, "*.om")))
            if not candidates:
                raise FileNotFoundError(f"No .om file found in {om_dir}")
            om_path = candidates[0]
        self.om_path = om_path

        self.input_name = input_name or os.getenv("FEATURE_OM_INPUT_NAME", "input")
        self.input_shape = input_shape or os.getenv("FEATURE_OM_INPUT_SHAPE", "1,3,600,600")
        self.input_dtype = (input_dtype or os.getenv("FEATURE_OM_INPUT_DTYPE", "float32")).lower()
        self.output_dtype = (output_dtype or os.getenv("FEATURE_OM_OUTPUT_DTYPE", self.input_dtype)).lower()
        self.output_shape = output_shape or os.getenv("FEATURE_OM_OUTPUT_SHAPE", "")
        self.device_id = device_id if device_id is not None else int(os.getenv("FEATURE_DEVICE_ID", "0"))

        image_size = self._infer_image_size(self.input_shape)
        self.tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _infer_image_size(self, shape: str) -> int:
        parts = [int(p) for p in shape.split(",") if p.strip().isdigit()]
        if len(parts) >= 4:
            return parts[-1]
        return 600

    def _parse_shape(self, shape: str) -> Optional[List[int]]:
        if not shape:
            return None
        parts = [int(p) for p in shape.split(",") if p.strip().isdigit()]
        return parts if parts else None

    def _dtype(self, name: str) -> np.dtype:
        name = name.lower()
        if name in {"fp16", "float16"}:
            return np.float16
        if name in {"fp32", "float32"}:
            return np.float32
        raise ValueError(f"Unsupported dtype: {name}")

    def load_image(self, img: bytes, mode: int = 0) -> Image.Image:
        if mode == 0:
            x_img = Image.open(io.BytesIO(img))
        elif mode == 1:
            binary_data = base64.b64decode(img)
            x_img = Image.open(io.BytesIO(binary_data))
        else:
            raise ValueError("mode must be 0 (bytes) or 1 (base64)")
        return x_img.convert("RGB")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        x = self.tfms(image)
        if hasattr(x, "unsqueeze") and getattr(x, "dim", lambda: 0)() == 3:
            x = x.unsqueeze(0)
        x = x.numpy().astype(self._dtype(self.input_dtype))
        return x

    def _run_ais_bench(self, x: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.bin")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)
            x.tofile(input_path)

            cmd = [
                "python",
                "-m",
                "ais_bench",
                "--model",
                self.om_path,
                "--input",
                input_path,
                "--output",
                output_dir,
                "--device",
                str(self.device_id),
                "--input_shape",
                f"{self.input_name}:{self.input_shape}",
                "--input_type",
                f"{self.input_name}:{self.input_dtype}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stdout + "\n" + result.stderr)

            output_bins = []
            for root, _, files in os.walk(output_dir):
                for name in files:
                    if name.endswith(".bin"):
                        output_bins.append(os.path.join(root, name))
            if not output_bins:
                raise FileNotFoundError("No output bin found from ais_bench")
            output_bins.sort()
            output_path = output_bins[0]

            y = np.fromfile(output_path, dtype=self._dtype(self.output_dtype))
            shape = self._parse_shape(self.output_shape)
            if shape and int(np.prod(shape)) == y.size:
                y = y.reshape(shape)
            return y

    def decode(self, outputs: np.ndarray) -> np.ndarray:
        y = outputs
        if y.ndim == 4:
            y = y.mean(axis=(2, 3))
        elif y.ndim == 2:
            pass
        elif y.ndim == 1:
            y = y[None, ...]
        else:
            y = y.reshape((y.shape[0], -1))
        return y[0]

    def postprocess(self, vec: np.ndarray) -> List[float]:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()

    def forward(self, img: bytes, mode: int = 0) -> List[float]:
        image = self.load_image(img, mode=mode)
        x = self.preprocess(image)
        outputs = self._run_ais_bench(x)
        vec = self.decode(outputs)
        return self.postprocess(vec)

class EfficientNetBottom(torch.nn.Module):
    def __init__(self, original_model_path="resources/tf_efficientnet_b7_ns-1dbc32de.pth"):

        super(EfficientNetBottom, self).__init__()
        self.original_model = timm.create_model(
            'tf_efficientnet_b7_ns', pretrained=False, checkpoint_path=original_model_path)
        self.features = torch.nn.Sequential(
            *list(self.original_model.children())[:-5])

        image_size = 600
        self.tfms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]),])

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("init...")
        print(self.device)
        self.features = self.features.to(self.device)
        self.features.eval()

    def forward(self, img, mode=0):
        if mode == 0:
            x = Image.open(io.BytesIO(img))
        elif mode == 1:
            binary_data = base64.b64decode(img)
            x = Image.open(io.BytesIO(binary_data))
        x = x.convert('RGB')
        x = self.tfms(x).unsqueeze(0).to(self.device)
        x = self.features(x)
        print(x.shape)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        result = x.cpu().detach().numpy().flatten()
        result = result / np.linalg.norm(result)
        return result.tolist()


if __name__ == "__main__":
    #print(torch.cuda.is_available(), torch.cuda.get_device_name())
    intermediate_model = EfficientNetBottom("resources/tf_efficientnet_b7_ns-1dbc32de.pth")
    onnx_model = EfficientNetB7Onnx("resources/tf_efficientnet_b7_ns-1dbc32de.onnx")

    path = "test/生成真实人物图片.png"
    with open(path, 'rb') as f:
        img_bytes = f.read()
        features1 = intermediate_model.forward(img_bytes)
        features2 = onnx_model.forward(img_bytes)

        print(len(features1), features1[:8])
        print(len(features2),features2[:8])
