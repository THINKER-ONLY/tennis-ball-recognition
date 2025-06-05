# 模型训练指南

本文档详细介绍如何使用本项目训练自定义的网球检测 YOLO 模型。

**重要前提**：在执行任何训练操作之前，请确保您已根据 `SETUP_ZH.md` 或项目根 `README.md` 中的"环境设置"部分，正确创建并激活了Python虚拟环境。
例如 (在项目根目录下执行)：
```bash
# cd path/to/tennis-ball-recognition # 如果您不在项目根目录，请先进入
source .venv/bin/activate
```

## 1. 数据集准备概述

高质量且格式正确的数据集是成功训练模型的基石。

### 1.1 最终数据集结构 (YOLO TXT 格式)

项目训练时期望的数据集结构如下，通常应位于 `data/processed_yolo_data/` 目录中：

```
tennis-ball-recognition/
└── data/
    └── processed_yolo_data/
        ├── images/
        │   ├── train/      # 训练图片 (image1.jpg, image2.png, ...)
        │   └── val/        # 验证图片 (image3.jpg, ...)
        ├── labels/
        │   ├── train/      # 训练标签 (image1.txt, image2.txt, ...)
        │   └── val/        # 验证标签 (image3.txt, ...)
        └── (应在项目根目录) dataset.yaml
```

### 1.2 YOLO TXT 标签格式

每个 `.txt` 标签文件对应一个图片，每行代表图片中的一个目标对象，格式为：
`<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`

*   `<class_id>`: 类别索引 (从0开始)。
*   `<x_center_norm>`, `<y_center_norm>`: 边界框中心的归一化坐标。
*   `<width_norm>`, `<height_norm>`: 边界框的归一化宽度和高度。

### 1.3 `dataset.yaml` 文件

此文件位于项目根目录，用于告知 YOLO 训练脚本数据集的位置和类别信息。

```yaml
# dataset.yaml 示例内容 (位于项目根目录)
path: data/processed_yolo_data  # 数据集的根路径 (相对于项目根目录)
train: images/train             # 训练图片的相对路径 (相对于上面的 path)
val: images/val                 # 验证图片的相对路径 (相对于上面的 path)
# test: images/test             # (可选) 测试图片目录

names:
  0: tennis_ball                # 类别名称，ID 0 对应 tennis_ball
# 1: other_object              # 如果有其他类别
```
确保 `path` 指向正确，并且 `names` 中的类别与您在 `src/convert_annotations.py` (如果使用) 中为 `--class_id` 设置的值或您的标签文件中的ID一致。

### 1.4 详细的数据准备步骤

本项目提供了从原始图片和JSON标注到最终YOLO格式数据集的完整处理流程，包括使用 `src/convert_annotations.py` 和 `src/split_dataset.py` 脚本。

**强烈建议您查阅项目根 `README.md` 文件中的 "数据准备 (用于训练新模型)" 部分，获取详细的、按部就班的指导。** 该部分包含了正确的脚本使用方法和参数说明。

## 2. 开始训练

### 2.1 通过 `run.sh` 自动训练 (推荐用于快速验证)

如果您已按照根 `README.md` 的指导正确准备了数据集 (`data/processed_yolo_data/` 和 `dataset.yaml`)，`run.sh` 脚本可以在找不到任何预训练或已训练模型时自动启动训练过程。

1.  **确保数据集就绪**：
    *   `data/processed_yolo_data/images/{train,val}` 包含图片。
    *   `data/processed_yolo_data/labels/{train,val}` 包含对应的TXT标签。
    *   项目根目录下的 `dataset.yaml` 已正确配置。
2.  **执行 `run.sh`**：
    ```bash
    bash run.sh
    ```
    如果脚本未找到任何位于项目根目录的 `.pt` 文件 (如 `best.pt`, `last.pt`, `yolo11n.pt`, `yolov8n.pt`) 或 `tennis_ball_runs/` 下的任何已训练模型，它将自动执行训练命令。

3.  **训练参数配置**：
    *   `run.sh` 内部的训练命令 (`yolo train ...`) 使用了一些预设参数，例如：
        *   基础模型：优先使用项目根目录的 `yolo11n.pt`，其次是 `yolov8n.pt`。如果都没有，YOLO 会尝试下载默认的 `yolov8n.pt`。
        *   数据集配置：`data=dataset.yaml`。
        *   Epochs: 默认为 50 (例如 `epochs=50`)。
        *   Batch size: 默认为 8 (例如 `batch=8`)。
        *   Project name (输出目录): `project=tennis_ball_runs`。
        *   Run name: `name=first_train` (如果已存在，会自动变为 `first_train2` 等)。
    *   如果您需要修改这些参数 (例如增加 epoch 数量)，您需要**直接编辑 `run.sh` 脚本**中调用 `yolo train` 的那一行。

### 2.2 直接使用 YOLO CLI 进行训练 (高级/完全控制)

为了更精细地控制训练参数和过程，推荐直接使用 `ultralytics` 提供的 `yolo` 命令行界面。

**前提**：
*   虚拟环境已激活。
*   `dataset.yaml` 已按1.3节所述配置完毕。
*   训练数据已按1.1节所述准备完毕。

**命令示例**：
```bash
yolo train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    project=tennis_ball_runs \
    name=my_tennis_ball_v1 \
    plots=True # 生成并保存训练过程中的各种图表
```

**常用参数解释**：
*   `data`: 指向您的 `dataset.yaml` 文件路径。
*   `model`: 指定用于开始训练的基础模型权重 (例如 `yolov8n.pt`, `yolov8s.pt`，或您之前训练的 `.pt` 文件进行继续训练)。
*   `epochs`: 训练的总轮数。
*   `imgsz`: 训练时图片输入尺寸 (例如 640)。
*   `batch`: 每批处理的图片数量。根据您的 GPU 显存调整。
*   `project`: 训练输出保存的项目主目录名称。默认是 `runs/train`。
*   `name`: 本次训练运行的特定名称。结果将保存在 `project/name` 目录下 (例如 `tennis_ball_runs/my_tennis_ball_v1/`)。
*   `plots`: 是否在训练结束后生成并保存各种可视化图表（如损失曲线、mAP曲线等）。

请查阅 Ultralytics YOLOv8 官方文档获取更详尽的训练参数列表和说明。

## 3. 训练过程与输出

*   **控制台输出**: 训练时，YOLO 会在控制台实时打印每个 epoch 的进度、损失值 (loss components)、验证集评估指标 (如 mAP50, mAP50-95) 等。
*   **结果保存**: 
    *   训练的所有输出，包括模型权重、日志、图表、验证结果等，都会保存在您指定的 `project/name` 目录下 (例如，`tennis_ball_runs/my_tennis_ball_v1/`)。
    *   关键文件和文件夹包括：
        *   `weights/`: 存放训练好的模型权重。
            *   `best.pt`: 在验证集上取得最佳 mAP50-95 指标的模型。
            *   `last.pt`: 完成所有 epochs 后最后一次保存的模型。
        *   `results.csv`: 包含每个 epoch 的详细指标数据的 CSV 文件。
        *   各种 `.png` 图表文件: 如 `confusion_matrix.png`, `results.png` (包含损失和mAP曲线), `F1_curve.png`, `PR_curve.png` 等 (如果 `plots=True`)。
        *   训练参数 (`args.yaml`) 和事件日志文件。

## 4. 使用训练好的模型进行检测

训练完成后，通常使用 `project/name/weights/best.pt` 作为您最终的模型。

*   **通过 `run.sh`**:
    *   将您训练得到的 `best.pt` (例如从 `tennis_ball_runs/my_tennis_ball_v1/weights/best.pt`) 复制到项目根目录，并可以重命名为 `best.pt` 或 `last.pt`。
    *   然后正常执行 `bash run.sh`，它会自动优先使用根目录下的这些模型。
    *   或者，修改 `run.sh` 脚本顶部的 `INPUT_SOURCE` 为您的测试数据，并确保模型查找逻辑会找到您新训练的模型（例如，如果它是 `tennis_ball_runs` 中最新的）。

*   **通过 `src/process.py` 直接检测**:
    ```bash
    python src/process.py \
        --weights tennis_ball_runs/my_tennis_ball_v1/weights/best.pt \
        --source path/to/your/test_image_or_video \
        --output_dir results/my_trained_model_test_output
    ```

## 5. 训练技巧与最佳实践

*   **数据集质量是关键**: 确保您的标注尽可能准确，图片具有多样性，能覆盖各种实际场景。
*   **数据增强**: YOLOv8 训练时会自动应用一系列数据增强方法。高级用户可以查阅文档了解如何自定义增强配置。
*   **超参数调整**: `epochs`, `batch`, `imgsz`, `lr0` (初始学习率), `lrf` (最终学习率) 等都是影响训练结果的关键超参数。通常需要实验来找到最优组合。
*   **从小处着手**: 可以先使用较少的 `epochs` (例如 10-20) 和较小的数据集子集（如果适用）进行快速测试，确保整个流程工作正常，然后再进行长时间的完整训练。
*   **监控验证集性能**: 密切关注验证集上的指标 (尤其是 mAP50-95)，`best.pt` 是根据此选出的，这有助于避免模型在训练集上过拟合。
*   **从预训练权重开始**: 通常，从官方提供的预训练权重 (如 `yolov8n.pt`) 开始训练，比完全从头随机初始化权重能更快达到更好的效果，特别是在数据集规模不是非常庞大的情况下。

有关训练的更深入信息和高级选项，请参考 Ultralytics YOLOv8 的官方文档。 