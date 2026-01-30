import os
from feature.feature_model import EfficientNetB7Onnx, EfficientNetB7AisBench
from loguru import logger


class FeatureService:

    def __init__(self):
        backend = os.getenv("FEATURE_BACKEND", "onnx").lower()
        if backend == "ais_bench":
            self.model = EfficientNetB7AisBench(
                om_path=os.getenv("FEATURE_OM_PATH"),
                input_name=os.getenv("FEATURE_OM_INPUT_NAME", "input"),
                input_shape=os.getenv("FEATURE_OM_INPUT_SHAPE", "1,3,600,600"),
                input_dtype=os.getenv("FEATURE_OM_INPUT_DTYPE", "float32"),
                output_dtype=os.getenv("FEATURE_OM_OUTPUT_DTYPE", None),
                output_shape=os.getenv("FEATURE_OM_OUTPUT_SHAPE", "1,64,19,19"),
                device_id=int(os.getenv("FEATURE_DEVICE_ID", "0")),
            )
            logger.info("Feature backend: ais_bench (OM)")
        else:
            #self.model = EfficientNetBottom()
            self.model = EfficientNetB7Onnx()
            logger.info("Feature backend: onnxruntime")
    async def feature(self, bytes) -> list[float]:
        return self.model.forward(bytes, 0)
        
        
