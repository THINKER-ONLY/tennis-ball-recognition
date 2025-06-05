# 环境设置与安装指南

本项目依赖 Python 和一些特定的库来运行。强烈建议在 Python 虚拟环境中管理这些依赖。

## 1. 前提条件

*   **Python**: 请确保您已安装 Python 3.8 或更高版本。
*   **Git**: 需要使用 Git 来克隆项目（如果尚未获取项目代码）。

## 2. 获取项目代码 (如果需要)

如果您尚未下载项目，请使用 Git 克隆：
```bash
git clone <项目仓库URL> # 例如：git clone https://github.com/your-username/tennis-ball-recognition.git
cd tennis-ball-recognition
```
如果您已拥有项目文件夹，请直接进入该目录：
```bash
cd tennis-ball-recognition
```
**后续所有命令都假定您位于项目根目录 (`tennis-ball-recognition/`) 下执行，除非另有说明。**

## 3. 创建 Python 虚拟环境

项目推荐使用虚拟环境。脚本（如 `run.sh`）通常会检查项目根目录下的 `.venv` 文件夹。

**选项 A: 使用 `uv` (推荐, 更快)**
如果您尚未安装 `uv`，请先参考其官方文档安装：[https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
```bash
# 确保您在项目根目录下 (例如 tennis-ball-recognition/)
uv venv
```

**选项 B: 使用 Python 内置 `venv`**
```bash
# 确保您在项目根目录下 (例如 tennis-ball-recognition/)
python3 -m venv .venv
```

## 4. 激活虚拟环境

**Linux / macOS / Windows (Git Bash / WSL):**
```bash
# 确保您在项目根目录下
source .venv/bin/activate
```
如果您使用的是 Windows CMD 或 PowerShell，激活命令可能不同 (例如 `.venv\\Scripts\\activate.bat` 或 `.venv\\Scripts\\Activate.ps1`)。本项目主要以 Linux/macOS 环境为基准编写说明。

激活成功后，您的命令行提示符前通常会显示 `(.venv)`。

**重要提示**: 在执行本项目中的任何 Python 脚本或 `run.sh` 脚本之前，**务必先激活虚拟环境**。

## 5. 安装依赖

项目所需的 Python 依赖项记录在项目根目录的 `requirements.txt` 文件中。

**选项 A: 使用 `uv` (在已激活的 `.venv` 环境中)**
```bash
uv pip install -r requirements.txt
```

**选项 B: 使用 `pip` (在已激活的 `.venv` 环境中)**
```bash
pip install -r requirements.txt
```
主要依赖包括 `ultralytics`, `opencv-python`, `tqdm`。请确保 `requirements.txt` 文件是最新的。

## 6. 模型与数据准备

详细的模型准备和数据准备（用于训练）步骤已在项目根目录的 `README.md` 文件中有详细说明。此处仅作简要提示：

*   **预训练模型**: 如需使用特定的YOLOv8预训练权重（如 `yolov8n.pt` 或 `yolo11n.pt`）作为训练起点或直接用于检测，可以将其放置在项目根目录下。具体请参考根 `README.md` 的 "**模型准备**" 部分。

*   **数据集准备 (用于训练)**: 如果您计划训练自己的模型，需要准备YOLO格式的数据集。
    *   详细的数据收集、标注转换 (`src/convert_annotations.py`)、数据集划分 (`src/split_dataset.py`) 以及 `dataset.yaml` 文件的配置方法，请参考根 `README.md` 的 "**数据准备 (用于训练新模型)**" 部分。

## 7. 完成设置

完成以上步骤后，您的开发环境就准备好了。您可以：
*   参考根 `README.md` 中的 "**快速开始：进行目标检测**" 部分来立即运行检测。
*   参考根 `README.md` 中的 "**数据准备 (用于训练新模型)**" 和后续的训练说明来训练您自己的模型。

如果您在设置过程中遇到任何问题，请仔细核对每个步骤，并检查根 `README.md` 中的相关说明。 