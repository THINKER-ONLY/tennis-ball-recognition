# 模型训练指南

本文档详细介绍如何使用本项目训练自定义的网球检测 YOLO 模型。

**重要前提**：在执行任何训练操作之前，请确保您已根据 `SETUP_ZH.md` 的说明，正确创建并激活了Python虚拟环境。
```bash
# 示例: 进入项目根目录并激活环境
cd /path/to/tennis-ball-recognition
source .venv/bin/activate
```

## 1. 数据集准备

高质量且格式正确的数据集是成功训练模型的基石。

### 1.1 数据集结构 (YOLO TXT 格式)

训练时期望的数据集结构如下：
```
tennis-ball-recognition/
└── data/
    └── processed_yolo_data/
        ├── images/
        │   ├── train/      # 训练图片
        │   └── val/        # 验证图片
        ├── labels/
        │   ├── train/      # 训练标签 (.txt)
        │   └── val/        # 验证标签 (.txt)
        └── (应在项目根目录) dataset.yaml
```

### 1.2 `dataset.yaml` 文件

此文件位于项目根目录，用于告知 YOLO 训练脚本数据集的位置和类别信息。
```yaml
# dataset.yaml 示例
path: data/processed_yolo_data  # 数据集的根路径 (相对于项目根目录)
train: images/train             # 训练图片的相对路径
val: images/val                 # 验证图片的相对路径

names:
  0: tennis_ball                # 类别ID 0 对应 tennis_ball
```

### 1.3 详细的数据准备步骤

本项目提供了从原始图片和JSON标注到最终YOLO格式数据集的完整处理流程，包括使用 `src/convert_annotations.py` 和 `src/split_dataset.py` 脚本。

**请查阅项目根 `README.md` 文件中的相关部分，获取详细的、按部就班的指导。**

## 2. 开始训练

### 2.1 方式一：通过 `run.sh` 自动训练 (推荐)

如果您已正确准备了数据集，`run.sh` 脚本可以在找不到任何可用模型时自动启动训练。

1.  **确保数据集和 `dataset.yaml` 就绪**。
2.  **移除旧模型 (如果需要触发训练)**:
    确保项目根目录下没有 `best.pt` 或 `last.pt`，并且 `tennis_ball_runs/` 目录为空或不存在。
3.  **执行 `run.sh`**:
    ```bash
    bash run.sh
    ```
    脚本在发现无模型可用时，会自动执行 `yolo train` 命令。

4.  **训练参数**: `run.sh` 内部的训练命令使用了一些预设参数（如 `epochs=50`）。如需修改，请直接编辑 `run.sh` 文件中调用 `yolo train` 的那一行。

### 2.2 方式二：直接使用 YOLO CLI (完全控制)

为了更精细地控制训练过程，推荐直接使用 `yolo` 命令行工具。

**命令示例**:
```bash
# 确保虚拟环境已激活
yolo train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    project=tennis_ball_runs \
    name=my_run_v1
```

**常用参数**:
*   `data`: 指向您的 `dataset.yaml` 文件。
*   `model`: 用于开始训练的基础模型 (例如 `yolov8n.pt`)。
*   `epochs`: 训练的总轮数。
*   `imgsz`: 输入图片尺寸。
*   `batch`: 每批处理的图片数量。
*   `project`: 训练输出保存的项目主目录。
*   `name`: 本次训练的特定名称。结果将保存在 `project/name` 目录下。

请查阅 Ultralytics YOLOv8 官方文档获取更详尽的参数列表。

## 3. 训练输出

训练的所有结果（模型权重、日志、图表等）都会保存在 `project/name` 目录下 (例如 `tennis_ball_runs/my_run_v1/`)。

关键文件位于 `weights/` 子目录：
*   `best.pt`: 在验证集上性能最佳的模型。**这是您通常最关心的模型。**
*   `last.pt`: 完成所有 epochs 后最后一次保存的模型。

## 4. 使用训练好的模型

训练完成后，您通常会使用 `best.pt` 模型进行检测。

*   **通过 `run.sh` (推荐)**:
    `run.sh` 会自动查找并使用最新训练出的 `best.pt` 模型。您只需正常执行 `bash run.sh` 即可。

*   **通过 `src/run_cli.py` 直接检测 (手动指定)**:
    如果您想用特定的模型进行测试，可以这样做：
    ```bash
    python src/run_cli.py \
        --weights tennis_ball_runs/my_run_v1/weights/best.pt \
        --source imgs/some_test_image.jpg
    ```

## 5. 训练技巧与最佳实践

*   **数据集质量是关键**: 确保您的标注尽可能准确，图片具有多样性，能覆盖各种实际场景。
*   **数据增强**: YOLOv8 训练时会自动应用一系列数据增强方法。高级用户可以查阅文档了解如何自定义增强配置。
*   **超参数调整**: `epochs`, `batch`, `imgsz`, `lr0` (初始学习率), `lrf` (最终学习率) 等都是影响训练结果的关键超参数。通常需要实验来找到最优组合。
*   **从小处着手**: 可以先使用较少的 `epochs` (例如 10-20) 和较小的数据集子集（如果适用）进行快速测试，确保整个流程工作正常，然后再进行长时间的完整训练。
*   **监控验证集性能**: 密切关注验证集上的指标 (尤其是 mAP50-95)，`best.pt` 是根据此选出的，这有助于避免模型在训练集上过拟合。
*   **从预训练权重开始**: 通常，从官方提供的预训练权重 (如 `yolov8n.pt`) 开始训练，比完全从头随机初始化权重能更快达到更好的效果，特别是在数据集规模不是非常庞大的情况下。

有关训练的更深入信息和高级选项，请参考 Ultralytics YOLOv8 的官方文档。 