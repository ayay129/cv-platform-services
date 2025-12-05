# 算法特征服务（EfficientNet-B7 ONNX）

基于 EfficientNet-B7 的图像特征提取服务，支持将 PyTorch `pth` 权重导出为 ONNX，并通过 ONNX Runtime 在 CPU/GPU 环境进行推理。提供 HTTP 接口与命令行工具进行特征计算与相似度比较。

## 功能概览
- 使用 `timm` 加载 EfficientNet-B7(NS) 权重并导出中间特征模块到 ONNX。
- 通过 ONNX Runtime 推理、全局平均池化与 L2 归一化，返回稳定的特征向量。
- FastAPI 服务：上传图片返回特征向量。
- CLI 工具：比较两张图片的余弦相似度或欧氏距离。
- Docker 打包运行。

## 环境要求
- Python `3.10`
- 主要依赖（见 `pyproject.toml`）：
  - `fastapi`, `uvicorn`, `orjson`, `loguru`
  - `timm`, `efficientnet-pytorch`（用于权重加载）
  - `onnxruntime`（推理，CPU 版），可选 `onnxruntime-gpu`
  - `onnx`（模型文件读写、合并外部数据）

## 项目结构
- `feature/`：服务与模型代码
  - `feature_model.py`：PyTorch 版（加载 `pth`）
  - `feature_model_onnx.py`：ONNX 版特征提取
  - `base_model_onnx.py`：ONNX 基础类，封装加载/预处理/推理/解码
  - `feature_service.py`、`main.py`：FastAPI 服务
  - `convert_to_onnx.py`：将 `pth` 导出为 ONNX 的脚本
  - `compare_similarity.py`：两图相似度比较 CLI
- `resources/`：模型文件（建议只追踪 `.onnx`）
- `test/`：示例图片

## 安装依赖
建议在虚拟环境中安装：

```bash
python -m pip install --upgrade pip
pip install -e .
```

如果需要 PyTorch（CPU 版）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 将 `pth` 导出为 ONNX
使用内置脚本导出中间特征模块（输出为 4D 特征图，在推理阶段做池化与归一化）：

```bash
python -m feature.convert_to_onnx \
  --checkpoint resources/tf_efficientnet_b7_ns-1dbc32de.pth \
  --output resources/tf_efficientnet_b7_ns-1dbc32de.onnx \
  --image_size 600 \
  --opset 13
```

说明：
- 某些环境会生成 `*.onnx.data` 外部数据文件；脚本会在导出后自动尝试合并到单一 `.onnx` 文件。
- 若模型超大（>2GB）合并会失败，此时只能使用外部数据格式。

## 运行 FastAPI 服务
本地运行（开发模式）：

```bash
uvicorn feature.main:create_app --host 0.0.0.0 --port 8000 --factory
```

- 默认 `FeatureService` 内部使用 ONNX 版模型。如果你的 ONNX 文件不是 `resources/efnet_b7.onnx`，可以在 `feature_service.py` 中给 `EfficientNetB7Onnx` 传入路径，或把你的模型文件重命名/软链接为该路径。
- 接口文档：访问 `http://localhost:8000/docs`

接口示例（上传图片返回特征向量）：

```bash
curl -X POST "http://localhost:8000/img_file/" \
  -F "file=@test/生成真实人物图片.png"
```

## 命令行：比较两图相似度
使用 ONNX 模型提取特征并比较相似度：

```bash
python -m feature.compare_similarity \
  --img1 test/生成真实人物图片.png \
  --img2 test/生成真实人物图片.png \
  --metric cosine
```

- `--metric` 支持：`cosine`（默认，点积，特征已 L2 归一化）或 `euclidean`（欧氏距离）。
- 指定模型路径：`--model resources/efnet_b7.onnx`

## Docker 运行
项目包含 `Dockerfile`，可直接构建运行：

```bash
docker build -t algo-feature:latest .
docker run --rm -p 8000:8000 algo-feature:latest
```

说明：
- 镜像内使用 `uvicorn feature.main:create_app --factory` 启动服务。
- 如需多进程：在 `CMD` 中添加 `--workers 2` 等参数。

## 开发与调试建议
- 包内模块运行请使用 `python -m feature.<module>` 方式，避免相对导入错误。
- 如果直接脚本运行，需要兼容性导入（已在部分脚本加入）。
- ONNX Runtime 在 macOS 上默认 CPU；若需 GPU 推理请安装 `onnxruntime-gpu` 并在支持 CUDA 的环境下使用。

## 常见问题
- `attempted relative import with no known parent package`：在项目根用包方式运行，例如：`python -m feature.feature_model_onnx`。
- `export() got an unexpected keyword argument 'use_external_data_format'`：你的 `torch.onnx.export` 版本不支持该参数，已在脚本中移除并通过导出后合并确保单文件。
- 生成了 `*.onnx.data`：脚本会尝试合并；若仍存在，说明模型过大或未安装 `onnx`，请安装后重试或保留外部数据文件。

## 许可证
未设置统一开源许可证，如需开源请补充相应声明。