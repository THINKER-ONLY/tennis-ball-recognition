# 模型训练中文指南

本文档详细介绍如何使用本项目训练自定义的网球检测 YOLO 模型。

## 1. 数据集准备

成功训练模型的关键在于高质量且格式正确的数据集。

### 1.1 数据集结构 (YOLO TXT 格式)

项目期望数据集遵循以下结构，通常放置在 `data/processed_yolo_data/` 目录下：

```
data/
└── processed_yolo_data/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.png
    │   │   └── ...
    │   └── val/ (或 test/)
    │       ├── image3.jpg
    │       └── ...
    ├── labels/
    │   ├── train/
    │   │   ├── image1.txt
    │   │   ├── image2.txt
    │   │   └── ...
    │   └── val/ (或 test/)
    │       ├── image3.txt
    │       └── ...
    └── dataset.yaml
```

- **`images/train/` 和 `images/val/`**: 分别存放训练集和验证集的图片文件。
- **`labels/train/` 和 `labels/val/`**: 分别存放对应的 YOLO TXT 格式的标签文件。每个图片文件在 `labels` 目录下都有一个同名的 `.txt` 文件。

### 1.2 YOLO TXT 标签格式

每个 `.txt` 标签文件包含一行或多行，每行对应图片中的一个目标对象。格式如下：

`<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`

- `<class_id>`: 目标类别的整数索引 (从 0 开始)。这应与 `dataset.yaml` 中定义的类别顺序一致。
- `<x_center_norm>`: 目标边界框中心点的 x 坐标，相对于图片宽度进行归一化 (0.0 到 1.0)。
- `<y_center_norm>`: 目标边界框中心点的 y 坐标，相对于图片高度进行归一化 (0.0 到 1.0)。
- `<width_norm>`: 目标边界框的宽度，相对于图片宽度进行归一化 (0.0 到 1.0)。
- `<height_norm>`: 目标边界框的高度，相对于图片高度进行归一化 (0.0 到 1.0)。

**示例 `image1.txt`:**
```
0 0.5 0.5 0.2 0.3  # 假设类别 0 是 'tennis_ball'
```

### 1.3 `dataset.yaml` 文件

此文件是 YOLO 训练所必需的，它定义了数据集的路径和类别信息。通常放置在 `data/dataset.yaml` 或 `data/processed_yolo_data/dataset.yaml`。

```yaml
# dataset.yaml 示例内容

path: ../data/processed_yolo_data  # 数据集根目录，相对于此yaml文件的路径或绝对路径
train: images/train  # 训练图片目录 (相对于 'path')
val: images/val    # 验证图片目录 (相对于 'path')
# test: images/test  # (可选) 测试图片目录 (相对于 'path')

# Classes
nc: 1  # 类别数量
names: ['tennis_ball']  # 类别名称列表，顺序必须与标签文件中的 class_id 对应
                        # 例如，如果 tennis_ball 是唯一的类别，class_id 就是 0
```

**重要提示:**
- `path` 字段应指向包含 `images` 和 `labels` 文件夹的目录。
- `train` 和 `val` (以及可选的 `test`) 字段是相对于 `path` 字段的路径。
- `names` 列表中的类别顺序决定了标签文件中 `<class_id>` 的含义。

### 1.4 数据集辅助脚本

- **`src/split_dataset.py`**: 如果您有一个包含所有图片和标签的文件夹，可以使用此脚本将其划分为 `train`, `val`, (和 `test`) 集。
  ```bash
  python src/split_dataset.py --source_dir ./data/source_dataset/all_my_data --output_dir ./data/processed_yolo_data --split_ratio 0.8 0.2
  ```
- **`src/convert_annotations.py`**: 如果您的标注是 COCO JSON 等其他格式，可以使用此脚本转换为 YOLO TXT 格式。
  ```bash
  python src/convert_annotations.py --json_path path/to/coco_annotations.json --output_dir ./data/yolo_labels --image_dir path/to/images
  ```
  转换后，您需要手动整理生成的 `.txt` 文件到 `labels/train` 和 `labels/val` 目录，并确保图片也在相应的 `images` 目录下。

## 2. 开始训练

训练主要通过 `run.sh` 脚本进行。

### 2.1 通过 `run.sh` 训练

确保您的 `data/dataset.yaml` 配置正确，并且数据集已按上述结构准备好。

**基本训练命令 (如果找不到模型，会自动训练):**
```bash
./run.sh --run_detection --input_image dummy.jpg --train_if_no_model
```
在这种情况下，如果脚本没有找到任何预训练模型 (`.pt` 文件或 `tennis_ball_runs/train/weights/best.pt`)，它将使用 `data/dataset.yaml` 中的配置开始训练。

**强制开始新训练:**
```bash
./run.sh --train_model
```

**指定训练参数:**
```bash
./run.sh --train_model --epochs 150 --batch_size 8 --img_size 640 --model_to_train yolov8s.pt --dataset_yaml ./data/my_custom_dataset.yaml
```
- `--epochs <int>`: 训练的总轮数。
- `--batch_size <int>`: 每批处理的图片数量。根据您的 GPU 显存调整。
- `--img_size <int>`: 训练时图片缩放到的尺寸 (例如 640, 1280)。
- `--model_to_train <model.pt>`: 指定用于迁移学习或从头开始训练的基础模型权重 (例如 `yolov8n.pt`, `yolov8s.pt`)。如果省略，YOLO 通常会使用默认的 `yolov8n.pt` 或从上次中断的地方继续。
- `--dataset_yaml <path>`: 指定要使用的 `dataset.yaml` 文件路径。

### 2.2 直接使用 YOLO CLI (高级)

如果您想更精细地控制训练过程，可以直接使用 Ultralytics YOLO 的命令行界面 (确保已在虚拟环境中安装 `ultralytics`)。

```bash
# 激活虚拟环境后
yolo train data=../data/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 name=tennis_ball_experiment1
```
- `data`: 指向 `dataset.yaml` 文件的路径。
- `model`: 基础模型权重。
- `epochs`, `imgsz`, `batch`: 与 `run.sh` 参数类似。
- `name`: 训练运行的名称，结果将保存在 `runs/detect/<name>` 或 `runs/train/<name>` (YOLO v8) 或 `tennis_ball_runs/<name>` (如果项目配置了特定的输出目录) 中。

## 3. 训练过程与输出

- **控制台输出**: 训练过程中，YOLO 会在控制台打印详细信息，包括每个 epoch 的损失 (loss)、评估指标 (mAP50, mAP50-95) 等。
- **结果保存**: 训练结果通常保存在 `tennis_ball_runs/train/` 或 Ultralytics 默认的 `runs/train/` 目录下。具体子目录名称可能包含您指定的 `name` 或自动生成的时间戳。
  主要内容包括：
  - `weights/`: 包含训练好的模型权重文件。
    - `best.pt`: 在验证集上表现最好的模型。
    - `last.pt`: 最新一轮训练结束后的模型。
  - `results.csv`: 包含每个 epoch 指标的 CSV 文件。
  - 各种图表 (PNG 文件): 如损失曲线、mAP 曲线、混淆矩阵等。
  - 训练参数日志。

## 4. 使用训练好的模型

训练完成后，`best.pt` 就是您自定义的网球检测模型。`run.sh` 脚本会自动优先使用 `tennis_ball_runs/train/weights/best.pt` (如果存在) 进行检测。您也可以通过 `--custom_model_path` 参数明确指定要使用的模型。

```bash
./run.sh --run_detection --input_image my_test_image.jpg --custom_model_path ./tennis_ball_runs/train/weights/best.pt
```

## 5. 提示与最佳实践

- **数据集质量**: "垃圾进，垃圾出"。确保标注准确、多样，覆盖各种场景。
- **数据增强**: YOLO 会自动应用标准的数据增强。您可以在 `dataset.yaml` 中或通过 YOLO CLI 参数调整增强策略。
- **超参数调整**: `epochs`, `batch_size`, `learning_rate` (学习率，可在 YOLO CLI 中设置) 等是重要的超参数。可能需要多次实验来找到最佳组合。
- **从小处着手**: 先用较小的 `epochs` 和默认参数进行快速测试，确保整个流程工作正常，然后再进行长时间的完整训练。
- **监控验证集性能**: `best.pt` 是基于验证集指标选出的，这有助于防止过拟合。 