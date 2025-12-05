from typing import List

import numpy as np

# 兼容导入：支持包运行（python -m feature.compare_similarity）和脚本直跑（python feature/compare_similarity.py）
try:
    from .feature_model import EfficientNetB7Onnx,EfficientNetBottom
except ImportError:
    from feature.feature_model import EfficientNetB7Onnx,EfficientNetBottom


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算余弦相似度。
    由于特征已做 L2 归一化，cosine(a, b) = dot(a, b)。
    """
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    return float(np.dot(a_np, b_np))


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """计算欧氏距离。"""
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a_np - b_np))


def main():
    
    model1 = EfficientNetB7Onnx("resources/tf_efficientnet_b7_ns-1dbc32de.onnx")
    model2 = EfficientNetBottom("resources/tf_efficientnet_b7_ns-1dbc32de.pth")

    with open("test/生成真实人物图片.png", "rb") as f:
        bytes1 = f.read()

    vec1 = model1.forward(bytes1, mode=0)
    vec2 = model2.forward(bytes1, mode=0)

    score = cosine_similarity(vec1, vec2)
    print(f"metric=cosine, similarity={score:.6f}")
    ecu_score = euclidean_distance(vec1, vec2)
    print(f"metric=euclidean, distance={ecu_score:.6f}")


if __name__ == "__main__":
    main()