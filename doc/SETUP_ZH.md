# 环境设置与安装中文指南

本项目依赖 Python 和一些特定的库来运行。建议使用虚拟环境进行管理。

## 1. Python 环境

确保您已安装 Python 3.8 或更高版本。

## 2. 创建虚拟环境

项目推荐使用虚拟环境。`run.sh` 脚本会检查 `.venv` 目录。

**选项 A: 使用 `uv` (推荐, 更快)**
如果您尚未安装 `uv`，请先参考其官方文档安装：[https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
```bash
# 在项目根目录下执行
uv venv
```

**选项 B: 使用 Python 内置 `venv`**
```bash
# 在项目根目录下执行
python3 -m venv .venv
```

## 3. 激活虚拟环境

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows (Git Bash / WSL):**
```bash
source .venv/Scripts/activate
```
激活成功后，您的命令行提示符前通常会显示 `(.venv)`。

## 4. 安装依赖

项目所需的 Python 依赖项记录在 `requirements.txt` 文件中。

**选项 A: 使用 `uv` (在已激活的 `.venv` 环境中)**
```bash
uv pip install -r requirements.txt
```
或者，如果 `requirements.txt` 尚未完善，至少需要安装核心依赖：
```bash
uv pip install ultralytics opencv-python tqdm
```

**选项 B: 使用 `pip` (在已激活的 `.venv` 环境中)**
```bash
pip install -r requirements.txt
```
或者，如果 `requirements.txt` 尚未完善，至少需要安装核心依赖：
```bash
pip install ultralytics opencv-python tqdm
```

## 5. 预训练模型 (可选)

如果您希望使用特定的预训练模型权重开始（例如 `yolov8n.pt`），请将其放置在项目根目录下。`run.sh` 脚本在训练前会检查是否存在这些文件。如果找不到，YOLO 通常会自动下载默认的基础模型。

## 6. 数据集准备

- **YOLO格式数据集**: 如果您要进行模型训练，请确保 `data/processed_yolo_data/` 目录下有符合 YOLO 格式的 `images` 和 `labels`，并且 `data/dataset.yaml` 文件已正确配置。
- **标注转换**: 如果您的原始标注不是 YOLO TXT 格式，可以使用 `src/convert_annotations.py` 脚本进行转换。具体用法请参考其内部注释或 `USAGE_ZH.md`。
- **数据集划分**: 可以使用 `src/split_dataset.py` 将数据集划分为训练集和验证集。

环境设置完成后，您可以通过 `run.sh` 脚本来运行训练或检测任务。 