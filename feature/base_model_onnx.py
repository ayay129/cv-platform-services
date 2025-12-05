import base64
import io
from typing import Callable, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("onnxruntime is required for ONNX inference. Please install it.") from e


class BaseOnnxModel:
    """
    ONNX 模型基础类，封装以下通用能力：
    - 会话初始化：自动选择可用的推理后端（优先 CUDA，其次 CPU），也支持自定义。
    - 图片加载：支持二进制字节与 Base64 输入，统一转换为 RGB。
    - 处理流程：预处理 -> ONNX 推理 -> 解码 -> L2 归一化，返回 Python list。

    子类扩展说明：
    - 当不同模型的“应用层”输出结构不一致时，重写 `decode(outputs)`；例如分类取 softmax，检测/分割做后处理等。
    - 如需自定义输入预处理（尺寸、归一化、颜色通道等），可在初始化传入 `transforms` 或覆盖 `preprocess(image)`。
    """

    def __init__(
        self,
        onnx_model_path: str,
        transforms: Optional[Callable[[Image.Image], Union[np.ndarray, "torch.Tensor"]]] = None,
        input_name: Optional[str] = None,
        providers: Optional[List[str]] = None,
        session_options: Optional["ort.SessionOptions"] = None,
    ) -> None:
        # 选择可用的推理后端（优先 CUDA，其次 CPU）；如均不可用则交由 ORT 默认策略
        available = set(ort.get_available_providers())
        default = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        selected = [p for p in default if p in available]

        # If none selected, let ORT decide the default provider
        if session_options is not None:
            self.session = ort.InferenceSession(
                onnx_model_path, providers=selected or None, sess_options=session_options
            )
        else:
            self.session = ort.InferenceSession(onnx_model_path, providers=selected or None)

        self.input_name = input_name or self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.transforms = transforms

    def load_image(self, img: Union[bytes, str], mode: int = 0) -> Image.Image:
        if mode == 0:
            x_img = Image.open(io.BytesIO(img))
        elif mode == 1:
            binary_data = base64.b64decode(img)
            x_img = Image.open(io.BytesIO(binary_data))
        else:
            raise ValueError("mode must be 0 (bytes) or 1 (base64)")
        return x_img.convert("RGB")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        默认预处理流程：
        - 若传入 `transforms`：先应用该变换；
          - 若返回值为 `torch.Tensor`：转换为 `numpy`，并在维度为 `CHW` 时添加批次维（-> `NCHW`）。
          - 若返回值为 `numpy`：确保 `float32` 类型；若为 `HWC/CHW` 则补齐批次维到 `NCHW`。
        - 若未传入 `transforms`：使用最小化预处理（缩放至 `224x224`，归一化到 `[0,1]`，并转换为 `NCHW`）。

        返回值：`np.ndarray`，形状为 `NCHW` 且 dtype 为 `float32`。
        """
        if self.transforms is not None:
            x = self.transforms(image)
            # 若返回的是 torch.Tensor：
            if hasattr(x, "numpy"):
                # 当维度为 CHW（无 batch）时，补一个批次维 -> NCHW
                if hasattr(x, "unsqueeze") and getattr(x, "dim", lambda: 0)() == 3:
                    x = x.unsqueeze(0)
                x = x.numpy().astype(np.float32)
            else:
                # 若返回的是 numpy 数组：确保类型为 float32，并补齐批次维
                x = np.asarray(x, dtype=np.float32)
                if x.ndim == 3:
                    x = x[None, ...]
            return x

        # 最小化预处理（无 transforms）：
        # 1) 缩放到固定尺寸 224x224
        image = image.resize((224, 224))
        # 2) 转成 numpy 并归一化到 [0,1]
        arr = np.array(image).astype(np.float32) / 255.0  # HWC, [0,1]
        # 3) HWC -> CHW
        arr = arr.transpose(2, 0, 1)  # CHW
        # 4) 添加批次维 -> NCHW
        return arr[None, ...]  # NCHW

    def prepare_input(self, x: np.ndarray) -> dict:
        return {self.input_name: x}

    def run(self, x: np.ndarray) -> List[np.ndarray]:
        outputs = self.session.run(None, self.prepare_input(x))
        return [np.asarray(o) for o in outputs]

    def decode(self, outputs: List[np.ndarray]) -> np.ndarray:
        """
        默认解码逻辑：
        - 取第一个输出（多数模型主输出在第一个），若为 4D 特征图（形状 `NCHW`），在空间维 `H,W` 上做全局平均池化得到 `NC`；
        - 若输出为 2D（形状 `NC`），直接使用；
        - 其他维度则扁平化为一维向量。

        注意：不同任务的应用层可能差异较大，子类可重写该方法实现特定后处理，例如：
        - 分类：`softmax`/`argmax` 或截取特征层；
        - 检测/分割：NMS、阈值、掩码重映射等；
        - 检索：可能直接取特征并做归一化。
        """
        y = outputs[0]
        if y.ndim == 4:
            # Global average pool over H, W: (N, C, H, W) -> (N, C)
            y = y.mean(axis=(2, 3))
        elif y.ndim == 2:
            # (N, C) already
            pass
        else:
            # Fallback: flatten per batch item
            y = y.reshape((y.shape[0], -1))
        return y[0]

    def postprocess(self, vec: np.ndarray) -> List[float]:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()

    def forward(self, img: Union[bytes, str], mode: int = 0) -> List[float]:
        """
        前向推理入口：完成从原始输入到最终特征向量的整条链路。
        参数：
        - `img`：图片的二进制字节或 Base64 字符串。
        - `mode`：输入类型标记；`0` 表示二进制字节，`1` 表示 Base64。
        流程：
        1) `load_image` 读取并转为 RGB；
        2) `preprocess` 做尺寸归一化与通道变换，得到 `NCHW`、`float32`；
        3) `run` 调用 ONNX 会话进行推理；
        4) `decode` 对输出进行任务相关的解码；
        5) `postprocess` 对向量做 L2 归一化并返回 Python list。
        """
        image = self.load_image(img, mode=mode)
        x = self.preprocess(image).astype(np.float32)
        outputs = self.run(x)
        vec = self.decode(outputs)
        return self.postprocess(vec)