import torch
import timm
from torchvision import transforms
import base64
import io
from typing import List
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("onnxruntime is required for ONNX inference. Please install it.") from e

from .base_model_onnx import BaseOnnxModel


class EfficientNetB7Onnx(BaseOnnxModel):
    def __init__(self, onnx_model_path: str = "resources/tf_efficientnet_b7_ns-1dbc32de.onnx"):
        image_size = 600
        tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Select available providers (CUDA if present, otherwise CPU)
        available = set(ort.get_available_providers())
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]

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