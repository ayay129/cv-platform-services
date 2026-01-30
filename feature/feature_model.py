import base64
import glob
import io
import os
from typing import List, Optional

import numpy as np
from PIL import Image
import timm
import torch
from torchvision import transforms

from .base_model_onnx import BaseOnnxModel
from ais_bench.infer.interface import InferSession

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
    _session = None
    _session_key = None

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
        self.output_shape = output_shape or os.getenv("FEATURE_OM_OUTPUT_SHAPE", "1,64,19,19")
        self.device_id = device_id if device_id is not None else int(os.getenv("FEATURE_DEVICE_ID", "0"))

        image_size = self._infer_image_size(self.input_shape)
        self.tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._ensure_session()

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
        x = x.numpy().astype(self._dtype(self.input_dtype), copy=False)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def _ensure_session(self) -> None:
        key = (self.device_id, self.om_path)
        if self.__class__._session is None or self.__class__._session_key != key:
            self.__class__._session = InferSession(self.device_id, self.om_path)
            self.__class__._session_key = key
        self.session = self.__class__._session

    def _input_realsize(self) -> Optional[int]:
        if hasattr(self.session, "get_inputs"):
            inputs = self.session.get_inputs()
            if inputs:
                return getattr(inputs[0], "realsize", None)
        return None

    def _run_session(self, x: np.ndarray) -> np.ndarray:
        mode = os.getenv("FEATURE_AIS_BENCH_MODE")
        custom_sizes = os.getenv("FEATURE_AIS_BENCH_CUSTOM_SIZES")
        if custom_sizes:
            try:
                custom_sizes = int(custom_sizes)
            except ValueError:
                custom_sizes = None
        if mode:
            outputs = self.session.infer([x], mode, custom_sizes=custom_sizes)
        else:
            try:
                outputs = self.session.infer([x])
            except Exception:
                outputs = self.session.infer([[x]])
        if isinstance(outputs, list) and outputs:
            return np.asarray(outputs[0])
        return np.asarray(outputs)

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
        realsize = self._input_realsize()
        if realsize and x.nbytes != realsize:
            raise RuntimeError(f"Input bytes {x.nbytes} != model input size {realsize}")
        outputs = self._run_session(x)
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
    om_model = EfficientNetB7AisBench("resources/tf_efficientnet_b7_ns-1dbc32de.om")
    path = "test/生成真实人物图片.png"
    with open(path, 'rb') as f:
        img_bytes = f.read()
        features1 = intermediate_model.forward(img_bytes)
        features2 = onnx_model.forward(img_bytes)
        features3 = om_model.forward(img_bytes)

        print(len(features1), features1[:8])
        print(len(features2),features2[:8])
        print(len(features3),features3[:8])
    print(timm.__version__)
