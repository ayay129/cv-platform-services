from feature.feature_model import EfficientNetBottom, EfficientNetB7Onnx
from loguru import logger


class FeatureService:

    def __init__(self):
        #self.model = EfficientNetBottom()
        self.model = EfficientNetB7Onnx()
    async def feature(self, bytes) -> list[float]:
        return self.model.forward(bytes, 0)
        
        