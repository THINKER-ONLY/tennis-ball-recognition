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
├── input/                  # 【重要】存放待检测的输入图片或视频文件 (run.sh 默认读取此目录)
├── results/                # 检测结果输出目录 (由脚本创建)
│                           # 当使用 run.sh 且 INPUT_SOURCE 为目录时, 例如: 
│                           # results/detection_output_best/input_image_name/input_image_name.txt
│                           # 当直接使用 process.py 且 source 为目录时类似。
├── slide/                  # 演示文稿 (可选)
├── src/                    # Python 源代码
│   ├── convert_annotations.py # 原始标注 (JSON) 转 YOLO TXT
│   ├── process.py          # 执行目标检测 (推理) 的核心脚本
│   └── split_dataset.py      # 划分训练/验证集
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
    主要依赖包括 `ultralytics`, `opencv-python`, `tqdm`。

## 快速开始：进行目标检测 (使用 `run.sh`)

这是使用本项目进行目标检测最简单直接的方法：

1.  **确保环境已设置**：请确保您已按照上面的“环境设置”部分创建了虚拟环境并安装了所有依赖。

2.  **激活虚拟环境**：
    ```bash
    cd tennis-ball-recognition
    source .venv/bin/activate
    ```

3.  **准备输入文件**：
    将您想要检测的图片文件 (例如 `.jpg`, `.png`) 或视频文件 (例如 `.mp4`) 放入项目根目录下的 `input/` 文件夹中。如果 `input/` 文件夹不存在，请创建它。

4.  **运行检测脚本**：
    ```bash
    bash run.sh
    ```
    此脚本会自动查找可用的最佳模型（优先使用您在根目录放置的 `best.pt` 或 `last.pt`，其次是最新训练生成的模型，最后是预训练的 `yolo11n.pt` 或 `yolov8n.pt`），然后对 `input/` 目录中的所有文件进行检测。

5.  **查看结果**：
    *   检测结果（包括带标注的图片/视频帧以及JSON格式的检测数据）将保存在 `results/` 目录下。
    *   具体的子目录结构通常是 `results/detection_output_[model_name]/`。
    *   如果 `input/` 中有多个文件，或者输入的是一个包含多张图片的子目录，那么在 `detection_output_[model_name]/` 下可能会为每个输入图片或其父目录创建进一步的子文件夹来存放对应的结果。
    *   例如，如果检测了 `input/my_image.jpg`，结果可能在 `results/detection_output_best/my_image/my_image.jpg` (标注图) 和 `results/detection_output_best/my_image/my_image.txt` (JSON数据)。

## 数据准备 (用于训练新模型)

1.  **放置原始数据**:
    *   将您的原始训练图片（例如 `.jpg`, `.png` 文件）放入 `data/source_dataset/original_images/` 目录。
    *   创建一个 JSON 标注文件 (例如 `data/source_dataset/annotations.json`)。
        *   JSON 格式应为：`{"image_filename.jpg": [{"x": x_abs, "y": y_abs, "w": w_abs, "h": h_abs}, ...], "image2.png": [...]}`。
        *   `x_abs, y_abs` 是边界框左上角的绝对像素坐标。
        *   `w_abs, h_abs` 是边界框的绝对像素宽度和高度。

2.  **转换标注为 YOLO TXT 格式**:
    运行 `src/convert_annotations.py` 脚本。
    ```bash
    # 确保虚拟环境已激活
    python src/convert_annotations.py \
        --json_file data/source_dataset/annotations.json \
        --images_dir data/source_dataset/original_images/ \
        --output_labels_dir data/processed_yolo_data/all_labels/ \
        --class_id 0 # 假设网球的类别ID为0
    ```
    这会将转换后的 `.txt` 标签文件输出到 `data/processed_yolo_data/all_labels/`。每个图像对应一个 `.txt` 文件。

3.  **准备图片进行划分**:
    由于 `split_dataset.py` 脚本会移动文件，建议先将原始图片复制到一个临时目录。
    ```bash
    # 确保虚拟环境已激活
    mkdir -p data/processed_yolo_data/images_for_splitting
    cp data/source_dataset/original_images/* data/processed_yolo_data/images_for_splitting/
    # (如果图片很多，请确保此复制操作适合您的文件数量和类型)
    ```

4.  **划分训练集和验证集**:
    运行 `src/split_dataset.py` 脚本。此脚本会将 `images_for_splitting/` 中的图片和 `all_labels/` 中的对应标签，按比例（默认为 80/20）移动到 `train` 和 `val` 子目录中。
    ```bash
    # 确保虚拟环境已激活
    python src/split_dataset.py \
        --base_images_dir data/processed_yolo_data/images_for_splitting/ \
        --base_labels_dir data/processed_yolo_data/all_labels/ \
        --output_root_images data/processed_yolo_data/images \
        --output_root_labels data/processed_yolo_data/labels \
        --train_ratio 0.8 \
        --seed 42
    ```
    执行后，数据结构应如下：
    *   `data/processed_yolo_data/images/train/`
    *   `data/processed_yolo_data/images/val/`
    *   `data/processed_yolo_data/labels/train/`
    *   `data/processed_yolo_data/labels/val/`
    原来的 `images_for_splitting/` 和 `all_labels/` 目录中的文件会被移走。

5.  **配置 `dataset.yaml`**:
    确保项目根目录下的 `dataset.yaml` 文件内容正确，指向处理后的数据。它应该如下所示：
    ```yaml
    path: data/processed_yolo_data  # 数据集的根路径 (相对于项目根目录)
    train: images/train             # 训练图片的相对路径 (相对于上面的 path)
    val: images/val                 # 验证图片的相对路径 (相对于上面的 path)

    names:
      0: tennis_ball                # 类别名称，0 是网球的类别ID (与 convert_annotations.py 中的 class_id 对应)
    ```
    **注意**: `dataset.yaml`中的 `path` 字段应设置为从项目根目录（即 `yolo` 命令执行的目录）到 `processed_yolo_data` 目录的相对路径。如果 `yolo train data=dataset.yaml` 从项目根目录运行，上述 `path: data/processed_yolo_data` 是正确的。

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
如“快速开始”部分所述，将您的图片或视频文件放入项目根目录下的 `input/` 文件夹。

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
    *   **执行检测**: 如果找到了任何可用的模型，脚本会调用 `src/process.py` 进行目标检测。
        *   **默认输入源**: `input/` 目录。你可以将图片或视频放入此目录。`run.sh` 脚本内部的 `INPUT_SOURCE` 变量可以修改以指向特定文件或摄像头。
        *   **检测结果**:
            *   带标注的图片/视频会根据 `run.sh` 中的 `SAVE_OUTPUT="true"` 设置保存。
            *   JSON 格式的检测结果会根据 `SAVE_JSON_OUTPUT="true"` 设置保存。
            *   **JSON 输出路径**: `run.sh` 尝试通过 `--output_json_path` 参数指定一个统一的JSON输出文件 (例如 `results/detection_output_best/result.txt`)。然而，`src/process.py` 脚本目前会为输入源中的每个单独图片生成一个对应的 `.txt` 文件 (内容为JSON)。
                *   如果 `INPUT_SOURCE` (在 `run.sh` 中配置) 是一个**目录** (例如默认的 `input/`)，并且包含如 `image1.jpg`, `image2.png` 的文件，则 JSON 文件会保存在类似 `results/detection_output_[model_name]/image1/image1.txt` 和 `results/detection_output_[model_name]/image2/image2.txt` 的路径下。
                *   如果 `INPUT_SOURCE` 是一个**单一文件** (例如 `input/my_video.mp4` 或 `input/my_image.jpg`)，对应的 `.txt` 文件 (例如 `my_video.txt` 或 `my_image.txt`) 会直接保存在 `results/detection_output_[model_name]/` 目录下。
            *   标注后的图片/视频的保存路径与上述 JSON 文件的组织方式类似，通常在同一子目录或主输出目录下。
        *   你可以在 `run.sh` 脚本顶部修改检测参数 (`INPUT_SOURCE`, `SHOW_OUTPUT`, `SAVE_OUTPUT`, `SAVE_JSON_OUTPUT` 等)。
    *   **执行训练**: 如果以上步骤均未找到可用的已训练模型，脚本将进入训练模式：
        *   它会优先使用项目根目录下的 `yolo11n.pt` (首选) 或 `yolov8n.pt` (次选) 作为训练的基础模型。
        *   如果根目录下没有这些基础模型，YOLO 将尝试从网络自动下载默认的 `yolov8n.pt`。
        *   训练命令使用 `dataset.yaml`，默认训练 50 个 epochs，参数为 `plots=False` (可以在 `run.sh` 中修改)。
        *   训练输出 (权重、日志等) 会保存在 `tennis_ball_runs/` 下的 `first_train` (或 `first_train2`, `first_train3`...) 目录中。
        *   训练完成后，脚本会尝试使用新训练出的最佳模型 (`best.pt`) 或最后模型 (`last.pt`) 对默认输入源 (`input/`) 进行一次检测。

### 直接运行 Python 脚本 (高级)

**前提条件**: 请确保已按照“环境设置”部分的说明激活了Python虚拟环境 (例如，通过 `source .venv/bin/activate`)。

#### 运行目标检测 (`src/process.py`)
```bash
# 确保虚拟环境已激活
python src/process.py \
    --weights path/to/your/model.pt \
    --source path/to/image_or_video_or_directory_or_camera_id \
    --output_dir results/my_custom_detection_output \
    --target_class "tennis ball" \
    --conf_thres 0.3 \
    --iou_thres 0.5 \
    --show_vid \
    --save_vid \
    --save_json
```
*   `--source`: 可以是单个图片、视频文件、包含图片的目录，或摄像头ID (如 `0`)。
*   `--output_dir`: 检测结果的基础输出目录。
    *   如果 `--source` 是单个文件 (例如 `image.jpg`)，标注后的图片/视频会直接保存在 `--output_dir` (例如 `results/my_custom_detection_output/processed_image.jpg`)。如果 `--save_json` 为 true，则 `image.txt` 也会保存在 `--output_dir` 中。
    *   如果 `--source` 是一个目录 (例如 `my_images/`)，`process.py` 会在 `--output_dir` 下为该目录中的每个图片创建一个子目录 (以图片文件名命名，不含扩展名，例如 `results/my_custom_detection_output/image_stem/`)，并将标注后的图片和对应的 `[image_stem].txt` (JSON格式) 保存在该子目录中。
*   `--save_json`: 为每个输入图片或视频的每一帧（如果适用）生成包含检测结果的 `.txt` 文件（内容为JSON格式）。文件名通常是基于输入图片名。
*   `--output_json_path`: 此参数在 `src/process.py` 中已弃用，脚本不使用它来决定JSON输出路径或文件名。
*   更多参数请运行 `python src/process.py --help`。

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
```bash
# 确保虚拟环境已激活
python src/convert_annotations.py \
    --json_file data/source_dataset/annotations.json \
    --images_dir data/source_dataset/original_images/ \
    --output_labels_dir data/processed_yolo_data/all_labels/ \
    --class_id 0
```

#### 运行数据集划分脚本 (`src/split_dataset.py`)
```bash
# 确保虚拟环境已激活
python src/split_dataset.py \
    --base_images_dir data/processed_yolo_data/images_for_splitting/ \
    --base_labels_dir data/processed_yolo_data/all_labels/ \
    --output_root_images data/processed_yolo_data/images \
    --output_root_labels data/processed_yolo_data/labels \
    --train_ratio 0.8
```

## 输出结果

*   **训练结果**: 默认保存在 `tennis_ball_runs/` 目录下，每个训练运行对应一个子目录 (如 `first_train/`, `my_custom_train_run/`)，其中包含 `weights/` (存放 `best.pt`, `last.pt`) 和其他训练日志及图表。
*   **检测结果**:
    *   **使用 `run.sh`**: 
        *   JSON 文件会如“使用 `run.sh` 运行项目”部分的“JSON 输出路径”所述生成。
        *   带标注的图片/视频会保存在相应的输出目录中，结构与JSON文件类似。
    *   **直接运行 `src/process.py`**:
        *   带标注的图片/视频会保存在 `--output_dir` 指定的目录或其子目录中。
        *   如果启用了 `--save_json`，对应的 `.txt` 文件 (JSON内容，通常命名为 `[image_stem].txt`) 会与标注图片一起保存。

## 注意事项
*   `run.sh` 中的 `INPUT_SOURCE` 变量可以修改为单个文件路径或摄像头ID。
*   所有路径参数都可以根据需要调整为绝对路径或相对路径。脚本内部通常会尝试解析相对路径。
*   `dataset.yaml` 中的 `path` 字段对于 `yolo train` 命令的正确执行至关重要。