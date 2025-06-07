# 网球目标检测项目 (YOLOv8)

本项目旨在使用 YOLOv8 实现网球的实时目标检测。项目包含数据预处理、模型训练和目标检测的完整流程，并通过 `run.sh` 脚本提供了一键执行检测或训练的功能。

## 目录结构

```
tennis-ball-recognition/
├── .venv/                  # Python 虚拟环境
├── data/
│   ├── source_dataset/
│   │   ├── original_images/  # 存放原始训练图片 (例如 .jpg, .png)
│   │   └── annotations.json  # (示例) 原始标注JSON文件
│   └── processed_yolo_data/  # YOLO格式的训练数据
│       ├── all_labels/       # (中间产物) convert_annotations.py 输出的所有 YOLO TXT 标签
│       ├── images_for_splitting/ # (中间产物) 准备用于 split_dataset.py 的图片副本
│       ├── images/
│       │   ├── train/        # 最终的训练图片
│       │   └── val/          # 最终的验证图片
│       └── labels/
│           ├── train/        # 最终的训练标签 (YOLO TXT)
│           └── val/          # 最终的验证标签 (YOLO TXT)
├── doc/                    # 项目文档 (可选)
├── imgs/                   # 【重要】存放待检测的输入图片文件 (run.sh 默认读取此目录)
├── results/                # 检测结果输出目录 (由脚本创建)
│                           # 每次运行会创建一个带时间戳的子目录, 例如: 
│                           # results/20240101_120000_output/
├── slide/                  # 演示文稿 (可选)
├── src/                    # Python 源代码
│   ├── convert_annotations.py # 原始标注 (JSON) 转 YOLO TXT
│   ├── process.py          # 核心检测逻辑 (提供 process_img 函数)
│   └── run_cli.py          # 项目的命令行接口 (CLI)
├── tennis_ball_runs/       # YOLO 训练输出 (例如 first_train/, first_train2/)
├── video/                  # 存放一些项目相关的视频文件 (可选, 例如测试视频)
├── .git/                   # Git 版本控制目录
├── .gitignore              # Git 忽略文件配置
├── dataset.yaml            # YOLO 数据集配置文件 (指向 processed_yolo_data)
├── README.md               # 本文档
├── requirements.txt        # Python 依赖包
├── run.sh                  # 项目主运行脚本 (推荐使用)
└── yolov8n.pt              # (可选) 预训练基础模型 (YOLOv8 Nano)
└── yolo11n.pt              # (可选) 预训练基础模型 (例如 YOLOv8.3.0 Nano)
└── best.pt                 # (可选) 用户训练好的最佳模型 (可放置于根目录供 run.sh 优先检测)
└── last.pt                 # (可选) 用户训练的最后模型 (可放置于根目录供 run.sh 优先检测)
```

## 环境设置

1.  **创建虚拟环境**
    *   推荐使用 `uv` (来自 Astral):
        ```bash
        uv venv
        ```
    *   或者使用 Python 内置的 `venv`:
        ```bash
        python3 -m venv .venv
        ```

2.  **激活虚拟环境**
    ```bash
    source .venv/bin/activate
    ```
    **重要提示**: 在执行本项目中的任何 Python 脚本或 `run.sh` 脚本之前，**务必先激活虚拟环境**。

3.  **安装依赖**
    *   如果使用 `uv` (虚拟环境激活后):
        ```bash
        uv pip install -r requirements.txt
        ```
    *   如果使用 `pip` (虚拟环境激活后):
        ```bash
        pip install -r requirements.txt
        ```
    该 `requirements.txt` 已配置为可自动安装适用于您机器 (CPU 或 GPU) 的 PyTorch 版本。

## 快速开始：进行目标检测 (使用 `run.sh`)

这是使用本项目进行目标检测最简单直接的方法：

1.  **确保环境已设置**：请确保您已按照上面的“环境设置”部分创建了虚拟环境并安装了所有依赖。

2.  **激活虚拟环境**：
    ```bash
    cd tennis-ball-recognition
    source .venv/bin/activate
    ```

3.  **准备输入文件**：
    将您想要检测的图片文件 (例如 `.jpg`, `.png`) 放入项目根目录下的 `imgs/` 文件夹中。如果 `imgs/` 文件夹不存在，请创建它。

4.  **运行检测脚本**：
    ```bash
    bash run.sh
    ```
    此脚本会自动查找可用的最佳模型（优先使用您在根目录放置的 `best.pt` 或 `last.pt`，其次是最新训练生成的模型），然后对 `imgs/` 目录中的所有图片进行检测。

5.  **查看结果**：
    *   检测结果将保存在 `results/` 目录下，每次运行都会创建一个带时间戳的新子目录 (例如 `results/20240101_120000_output/`)。
    *   **标注图片**: 所有处理过的、带有检测框的图片都保存在该子目录中。
    *   **JSON 数据**: 一个名为 `_predictions.json` 的文件会保存在该子目录中，它包含了该次运行所有图片及其检测结果的完整记录。

## 数据准备 (用于训练新模型)

关于如何准备自定义数据集以及训练新模型，请参阅详细的 **[模型训练指南 (`doc/TRAINING_ZH.md`)](./doc/TRAINING_ZH.md)**。

## 模型准备

### 基础预训练模型 (用于开始新训练)
*   `run.sh` 脚本在启动新训练前，会优先在项目根目录 (`tennis-ball-recognition/`) 查找预训练的基础模型。
*   你可以手动下载 YOLOv8 nano 模型的权重文件 (例如从 Ultralytics GitHub Releases)，并将其命名为 `yolo11n.pt` 或 `yolov8n.pt` 后放置在项目根目录下。
    *   脚本会优先查找 `yolo11n.pt`。
    *   如果未找到，则查找 `yolov8n.pt`。
*   如果上述文件均未在项目根目录找到，YOLO 将尝试自动从网络下载默认的基础模型 (通常是 `yolov8n.pt`)。

### 已训练好的模型 (用于直接检测)
*   如果你有之前训练好的模型权重 (例如通过本项目的训练流程生成的)，并希望 `run.sh` 脚本优先使用它们进行检测，可以将它们放置在项目根目录下。
    *   将性能最佳的模型权重文件命名为 `best.pt`。
    *   将最后一次训练迭代的模型权重文件命名为 `last.pt`。
*   `run.sh` 会优先查找并使用根目录下的 `best.pt`，其次是 `last.pt`。如果这些不存在，它会查找 `tennis_ball_runs/` 下最新训练运行中的 `weights/best.pt` 或 `weights/last.pt`。

## 详细使用说明

### 使用 `run.sh` 运行项目 (推荐)

**前提条件**: 请确保已按照“环境设置”部分的说明激活了Python虚拟环境 (例如，通过 `source .venv/bin/activate`)。

项目的主入口点是 `run.sh` 脚本。它会自动处理环境激活（再次确认）、模型查找、训练或检测的逻辑。

**放置输入文件进行检测**: 
如“快速开始”部分所述，将您的图片文件放入项目根目录下的 `imgs/` 文件夹。

**执行脚本**:
```bash
bash run.sh
```

**`run.sh` 的行为逻辑**:

1.  **激活虚拟环境**: 再次尝试激活 `.venv` (如果尚未激活)。
2.  **智能模型查找与决策**:
    *   **优先检测 (项目根目录)**: 检查项目根目录下是否存在 `best.pt`。如果存在，则使用此模型进行目标检测。
    *   如果根目录下没有 `best.pt`，则检查是否存在 `last.pt`。如果存在，则使用此模型进行目标检测。
    *   **其次检测 (`tennis_ball_runs/` 目录)**: 如果项目根目录下没有找到用户提供的 `best.pt` 或 `last.pt`，脚本会查找 `tennis_ball_runs/` 目录下由YOLO训练产生的最新 `first_trainX` 子目录，并尝试使用该目录下的 `weights/best.pt` (优先) 或 `weights/last.pt` (其次) 进行目标检测。
    *   **执行检测**: 如果找到了任何可用的模型，脚本会调用 `src/run_cli.py` 进行目标检测。
        *   **默认输入源**: `imgs/` 目录。`run.sh` 脚本内部的 `INPUT_SOURCE` 变量指向此目录。
        *   **检测结果**:
            *   结果保存在 `results/` 下的一个带时间戳的子目录中。
            *   该目录包含所有标注后的图片和一个 `_predictions.json` 汇总文件。
        *   你可以在 `run.sh` 脚本顶部修改 `INPUT_SOURCE` 变量来指定不同的输入源。
    *   **执行训练**: 如果以上步骤均未找到可用的已训练模型，脚本将进入训练模式：
        *   它会优先使用项目根目录下的 `yolo11n.pt` (首选) 或 `yolov8n.pt` (次选) 作为训练的基础模型。
        *   如果根目录下没有这些基础模型，YOLO 将尝试从网络自动下载默认的 `yolov8n.pt`。
        *   训练命令使用 `dataset.yaml`，默认训练 50 个 epochs，参数为 `plots=False` (可以在 `run.sh` 中修改)。
        *   训练输出 (权重、日志等) 会保存在 `tennis_ball_runs/` 下的 `first_train` (或 `first_train2`, `first_train3`...) 目录中。
        *   训练完成后，脚本会尝试使用新训练出的最佳模型 (`best.pt`) 或最后模型 (`last.pt`) 对默认输入源 (`imgs/`) 进行一次检测。

### 直接运行 Python 脚本 (高级)

**前提条件**: 请确保已按照“环境设置”部分的说明激活了Python虚拟环境 (例如，通过 `source .venv/bin/activate`)。

#### 运行目标检测 (`src/run_cli.py`)
`src/run_cli.py` 是项目功能丰富的命令行接口 (CLI)。`run.sh` 实际上是围绕此脚本的包装器。您可以直接运行它以获得更多控制选项，例如处理视频、实时摄像头或指定特定参数。

```bash
# 示例：检测单个图片，并指定输出目录
python src/run_cli.py \
    --weights best.pt \
    --source path/to/your/image.jpg \
    --output-dir results/my_custom_run

# 示例：处理整个目录的图片和视频，并关闭实时显示
python src/run_cli.py \
    --weights best.pt \
    --source path/to/your/folder/ \
    --output-dir results/another_run \
    --hide

# 示例：使用摄像头进行实时检测
python src/run_cli.py \
    --weights best.pt \
    --source 0
```
*   `--source`: 可以是单个图片、视频文件、包含媒体文件的目录，或摄像头ID (如 `0`)。
*   `--output-dir`: 检测结果的输出目录。如果未提供，则会自动在 `results/` 下创建一个带时间戳的目录。
*   `--hide`: 隐藏实时显示的检测窗口。
*   更多参数请运行 `python src/run_cli.py --help`。

#### 运行 YOLO 训练 (直接使用 `yolo` CLI)
```bash
# 确保虚拟环境已激活
yolo train \
    model=yolov8n.pt \
    data=dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    project=tennis_ball_runs \
    name=my_custom_train_run \
    plots=True
```
更多参数请参考 Ultralytics YOLOv8 文档。

#### 运行数据转换脚本 (`src/convert_annotations.py`)
该功能已从主流程中移除以简化项目。如需进行数据格式转换，请参考 `doc/TRAINING_ZH.md` 中的数据准备步骤。

#### 运行数据集划分脚本 (`src/split_dataset.py`)
该功能已从主流程中移除以简化项目。如需进行数据集划分，请参考 `doc/TRAINING_ZH.md` 中的数据准备步骤。

## 输出结果

*   **训练结果**: 默认保存在 `tennis_ball_runs/` 目录下，每个训练运行对应一个子目录 (如 `first_train/`, `my_custom_train_run/`)，其中包含 `weights/` (存放 `best.pt`, `last.pt`) 和其他训练日志及图表。
*   **检测结果**:
    *   **使用 `run.sh` 或 `run_cli.py`**:
        *   结果保存在 `results/` 下的一个唯一的、带时间戳的子目录中。
        *   该子目录包含所有经处理并带有标注框的图片/视频帧。
        *   该子目录还包含一个 `_predictions.json` 文件，其中记录了所有检测到的对象的详细信息（文件名、类别、坐标、置信度）。

## 注意事项
*   `run.sh` 中的 `INPUT_SOURCE` 变量可以修改为单个文件路径或摄像头ID。
*   所有路径参数都可以根据需要调整为绝对路径或相对路径。脚本内部通常会尝试解析相对路径。
*   `dataset.yaml` 中的 `path` 字段对于 `yolo train` 命令的正确执行至关重要。