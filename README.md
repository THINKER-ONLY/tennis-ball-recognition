# 网球目标检测项目 (YOLOv8)

本项目旨在使用 YOLOv8 实现网球的实时目标检测。项目包含数据预处理、模型训练和目标检测的完整流程。

## 目录结构

```
tennis-ball-recognition/
├── .venv/                  # Python 虚拟环境
├── data/
│   ├── source_dataset/
│   │   └── original_competition_data/ # 存放原始图片和标注JSON文件 (例如 result.txt)
│   └── processed_yolo_data/ # YOLO格式的数据
│       ├── all_labels/      # 转换后的所有 YOLO TXT 格式标签 (中间产物)
│       ├── images/
│       │   ├── train/       # 训练图片
│       │   └── val/         # 验证图片
│       └── labels/
│           ├── train/       # 训练标签
│           └── val/         # 验证标签
├── doc/                    # 项目文档 (可选)
├── results/                # 检测结果输出目录
├── slide/                  # 演示文稿 (可选)
├── src/                    # Python 源代码
│   ├── convert_annotations.py # 原始标注转 YOLO TXT
│   ├── process.py         # 执行目标检测 (推理)
│   └── split_dataset.py     # 划分训练/验证集
├── tennis_ball_runs/       # YOLO 训练输出 (例如 first_train/, first_train2/)
├── video/                  # 存放视频文件 (可选)
├── .gitignore              # Git 忽略文件配置
├── dataset.yaml            # YOLO 数据集配置文件
├── README.md               # 本文档
├── requirements.txt        # Python 依赖包
├── run.sh                  # 项目主运行脚本
└── yolov8n.pt              # (可选) 预训练基础模型，可放置于此
└── yolo11n.pt             # (可选) 预训练基础模型 (如YOLOv8.3.0 nano)，可放置于此
└── best.pt                 # (可选) 用户训练好的最佳模型，可放置于此供优先检测
└── last.pt                 # (可选) 用户训练的最后模型，可放置于此供优先检测
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

3.  **安装依赖**
    *   如果使用 `uv`:
        ```bash
        uv pip install -r requirements.txt
        ```
    *   如果使用 `pip`:
        ```bash
        pip install -r requirements.txt
        ```
    主要依赖包括 `ultralytics`, `opencv-python`, `tqdm`。

## 数据准备

1.  **放置原始数据**:
    *   将你的原始图片（例如 `.jpg` 文件）和对应的标注文件（通常是一个包含所有图片标注的 JSON 文件，本项目中示例为 `result.txt`）放入 `data/source_dataset/original_competition_data/` 目录。
    *   原始 JSON 标注格式应为：`{"image_filename.jpg": [{"x": x_abs, "y": y_abs, "w": w_abs, "h": h_abs}, ...]}`，其中 `x_abs, y_abs` 是边界框左上角的绝对像素坐标，`w_abs, h_abs` 是绝对像素宽度和高度。

2.  **转换标注为 YOLO TXT 格式**:
    运行 `src/convert_annotations.py` 脚本。确保脚本内的原始图片路径和 JSON 文件名与你的设置一致。
    ```bash
    # 确保虚拟环境已激活
    # 示例命令，请根据convert_annotations.py的实际参数进行调整（如果它接受参数）
    # 如果脚本是硬编码路径，请直接运行
    python src/convert_annotations.py
    ```
    这会将转换后的 `.txt` 标签文件输出到 `data/processed_yolo_data/all_labels/`。每个图像对应一个 `.txt` 文件，每行格式为 `<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`。

3.  **划分训练集和验证集**:
    运行 `src/split_dataset.py` 脚本。此脚本会将 `data/source_dataset/original_competition_data/` 中的图片复制到 `data/processed_yolo_data/images/` (如果尚未完成)，然后将 `data/processed_yolo_data/images/` 中的图片和 `data/processed_yolo_data/all_labels/` 中的对应标签，按比例（默认为 80/20）划分到 `train` 和 `val` 子目录中。
    ```bash
    # 确保虚拟环境已激活
    python src/split_dataset.py
    ```
    执行后，数据结构应如下：
    *   `data/processed_yolo_data/images/train/`
    *   `data/processed_yolo_data/images/val/`
    *   `data/processed_yolo_data/labels/train/`
    *   `data/processed_yolo_data/labels/val/`

4.  **配置 `dataset.yaml`**:
    确保项目根目录下的 `dataset.yaml` 文件内容正确，指向处理后的数据。它应该如下所示：
    ```yaml
    path: data/processed_yolo_data  # 数据集的根路径 (相对于项目根目录)
    train: images/train             # 训练图片的相对路径 (相对于上面的 path)
    val: images/val                 # 验证图片的相对路径 (相对于上面的 path)
    names:
      0: tennis_ball                # 类别名称，0 是网球的类别ID
    ```

## 模型准备

### 基础预训练模型 (用于开始新训练)
*   `run.sh` 脚本在启动新训练前，会优先在项目根目录 (`tennis-ball-recognition/`) 查找预训练的基础模型。
*   你可以手动下载 YOLOv8 nano 模型的权重文件 (例如从 Ultralytics GitHub Releases)，并将其命名为 `yolo11n.pt` 或 `yolov8n.pt` 后放置在项目根目录下。
    *   脚本会优先查找 `yolo11n.pt`。
    *   如果未找到，则查找 `yolov8n.pt`。
*   如果上述文件均未在项目根目录找到，YOLO 将尝试自动从网络下载默认的基础模型 (通常是 `yolov8n.pt`，这可能会被解析为下载特定版本如 `yolo11n.pt`)。

### 已训练好的模型 (用于直接检测)
*   如果你有之前训练好的模型权重 (例如通过本项目的训练流程生成的)，并希望 `run.sh` 脚本优先使用它们进行检测，可以将它们放置在项目根目录下。
    *   将性能最佳的模型权重文件命名为 `best.pt`。
    *   将最后一次训练迭代的模型权重文件命名为 `last.pt`。
*   `run.sh` 会优先查找并使用根目录下的 `best.pt`，其次是 `last.pt`。

## 使用 `run.sh` 运行项目

项目的主入口点是 `run.sh` 脚本。它会自动处理环境激活、模型查找、训练或检测的逻辑。

**执行脚本**:
```bash
bash run.sh
```

**`run.sh` 的行为逻辑**:

1.  **激活虚拟环境**: 自动激活 `.venv`。
2.  **智能模型查找与决策**:
    *   **优先检测 (项目根目录)**: 检查项目根目录下是否存在 `best.pt`。如果存在，则使用此模型进行目标检测。
    *   如果根目录下没有 `best.pt`，则检查是否存在 `last.pt`。如果存在，则使用此模型进行目标检测。
    *   **其次检测 (`tennis_ball_runs/` 目录)**: 如果项目根目录下没有找到 `best.pt` 或 `last.pt`，脚本会查找 `tennis_ball_runs/` 目录下由YOLO训练产生的最新 `first_trainX` 子目录，并尝试使用该目录下的 `weights/best.pt` (优先) 或 `weights/last.pt` (其次) 进行目标检测。
    *   **执行检测**: 如果找到了任何可用的模型，脚本会调用 `src/process.py` 进行目标检测。
        *   默认输入源是 `data/source_dataset/original_competition_data/1747468171.jpg`。
        *   检测结果（带标注的图片/视频、JSON文件）默认保存在 `results/detection_output_[model_name]/` 目录下。
        *   你可以在 `run.sh` 脚本内部修改这些检测参数 (`INPUT_SOURCE`, `OUTPUT_DIR` 等)。
    *   **执行训练**: 如果以上步骤均未找到可用的已训练模型，脚本将进入训练模式：
        *   它会优先使用项目根目录下的 `yolo11n.pt` (首选) 或 `yolov8n.pt` (次选) 作为训练的基础模型。
        *   如果根目录下没有这些基础模型，YOLO 将尝试从网络自动下载默认的 `yolov8n.pt`。
        *   训练命令使用 `dataset.yaml`，默认训练 50 个 epochs，参数为 `plots=False`。
        *   训练输出 (权重、日志等) 会保存在 `tennis_ball_runs/` 下的 `first_train` (或 `first_train2`, `first_train3`...) 目录中。
        *   训练完成后，脚本会列出新生成的权重文件路径。

## 直接运行 Python 脚本 (高级)

### 运行目标检测 (`src/process.py`)
```bash
# 确保虚拟环境已激活
python src/process.py \
    --weights path/to/your/model.pt \
    --source path/to/image_or_video_or_directory \
    --output_dir path/to/save/results/ \
    --target_class "tennis ball" \
    --show_vid \
    --save_vid \
    --save_json \
    --output_json_path path/to/save/results.json
```
更多参数请运行 `python src/process.py --help`。

### 运行 YOLO 训练
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

## 输出结果

*   **训练结果**: 默认保存在 `tennis_ball_runs/` 目录下，每个训练运行对应一个子目录 (如 `first_train/`, `my_custom_train_run/`)，其中包含 `weights/` (存放 `best.pt`, `last.pt`) 和其他训练日志及图表。
*   **检测结果**: 使用 `run.sh` 或直接运行 `src/process.py` 并启用保存选项时，结果（带标注的图片/视频、JSON 文件）会保存在 `results/` 下的指定子目录中 (例如 `results/detection_output_best/`)。