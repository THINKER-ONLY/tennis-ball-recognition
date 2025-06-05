# 项目使用指南

本文档旨在帮助您理解并有效使用本网球目标检测项目。推荐的使用方式是通过项目根目录下的 `run.sh` 脚本，它简化了许多操作。同时也提供了直接调用Python脚本的高级用法。

**重要前提**：在执行任何操作之前，请确保您已根据 `SETUP_ZH.md` 或项目根 `README.md` 中的"环境设置"部分，正确创建并激活了Python虚拟环境。
例如 (在项目根目录下执行)：
```bash
# cd path/to/tennis-ball-recognition # 如果您不在项目根目录，请先进入
source .venv/bin/activate
```

## 1. 快速上手：使用 `run.sh` 进行检测

这是最简单直接的检测方法：

1.  **准备输入文件**：
    将您想要检测的图片文件 (例如 `.jpg`, `.png`) 或视频文件 (例如 `.mp4`) 放入项目根目录下的 `input/` 文件夹中。如果 `input/` 文件夹不存在，请创建它。

2.  **运行检测脚本**：
    在已激活虚拟环境的项目根目录下，执行：
    ```bash
    bash run.sh
    ```

3.  **工作流程**：
    *   `run.sh` 脚本会自动尝试激活虚拟环境（如果尚未激活）。
    *   它会智能查找可用的模型：
        1.  优先使用项目根目录下的 `best.pt`。
        2.  其次是项目根目录下的 `last.pt`。
        3.  然后是在 `tennis_ball_runs/` 目录下最新训练生成的 `weights/best.pt`。
        4.  再次是在 `tennis_ball_runs/` 目录下最新训练生成的 `weights/last.pt`。
        5.  如果上述模型都未找到，它会尝试使用项目根目录下的 `yolo11n.pt` (首选) 或 `yolov8n.pt` (次选) 作为预训练模型进行检测。
    *   如果找到了任何模型，脚本将使用该模型对 `input/` 目录中的所有文件进行检测。
    *   如果**没有**找到任何可用的已训练模型或预置的基础模型，脚本会自动进入**训练模式** (详见"模型训练"部分)。训练完成后，它会尝试使用新训练的模型进行一次检测。

4.  **查看结果**：
    *   检测结果（包括带标注的图片/视频帧以及JSON格式的检测数据）将保存在 `results/` 目录下。
    *   具体的子目录结构通常是 `results/detection_output_[model_name]/`。
    *   详情请参考 `OUTPUT_FORMAT_ZH.md` 或根 `README.md` 中的"输出结果"部分。

## 2. `run.sh` 脚本详解

`run.sh` 脚本旨在简化项目的常规操作。其核心逻辑是：**优先检测，若无模型则尝试训练，训练后再次尝试检测。**

### 2.1 主要行为

1.  **环境激活**：尝试激活 `.venv` 虚拟环境。
2.  **模型查找**：按照上一节"工作流程"中描述的顺序查找可用模型。
3.  **执行检测 (如果找到模型)**：
    *   调用 `src/process.py` 脚本。
    *   **输入源**：默认配置为项目根目录下的 `input/` 文件夹。您可以在 `run.sh` 脚本顶部修改 `INPUT_SOURCE` 变量来指向特定的单个文件、不同的目录或摄像头ID (例如 `INPUT_SOURCE="0"` 表示默认摄像头)。
    *   **其他检测参数**：诸如是否显示实时画面 (`SHOW_OUTPUT`)、是否保存标注后的图片/视频 (`SAVE_OUTPUT`)、是否保存JSON结果 (`SAVE_JSON_OUTPUT`) 等，都可以在 `run.sh` 脚本顶部的变量区域进行配置。
4.  **执行训练 (如果未找到可用模型)**：
    *   使用 `dataset.yaml` 配置文件。
    *   使用项目根目录下的 `yolo11n.pt` (首选)或 `yolov8n.pt` (次选) 作为基础模型开始训练。如果这些也不存在，YOLO会尝试自动下载默认的 `yolov8n.pt`。
    *   默认训练参数（如 epochs=50, batch=8, plots=False）硬编码在 `run.sh` 的训练命令中，您可以直接修改脚本以调整这些参数。
    *   训练输出保存在 `tennis_ball_runs/[train_name]/` 目录下。
5.  **训练后检测**：训练成功后，会自动使用新生成的最佳模型对 `INPUT_SOURCE` 进行一次检测。

### 2.2 在 `run.sh` 中自定义参数

打开 `run.sh` 文件，您会在脚本的靠前部分看到一些可配置的变量，例如：

```bash
# --- 全局配置 ---
TRAIN_OUTPUT_PROJECT_DIR="tennis_ball_runs"
TRAIN_BASE_NAME="first_train"

INPUT_SOURCE="input" # 修改这里来改变检测源
TARGET_CLASS="tennis_ball"
SHOW_OUTPUT="true"         # "true" 或 "false"
SAVE_OUTPUT="true"         # "true" 或 "false"
SAVE_JSON_OUTPUT="true"    # "true" 或 "false"
# ... 以及可能的其他训练参数如 EPOCHS, BATCH_SIZE 等，如果脚本支持在这里配置
```
直接修改这些变量的值即可改变脚本的行为，无需通过命令行参数。

## 3. 高级用法：直接运行 Python 脚本

对于更细致的控制或特定任务，您可以直接运行 `src/` 目录下的 Python 脚本。
**提醒**：执行前请确保虚拟环境已激活。

### 3.1 目标检测 (`src/process.py`)

用于执行目标检测。

**命令示例**：
```bash
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
**主要参数说明**：
*   `--weights`: (必需) 模型权重文件路径 (例如 `yolov8n.pt` 或 `tennis_ball_runs/first_train/weights/best.pt`)。
*   `--source`: (必需) 输入源。可以是单个图片路径、视频文件路径、包含图片的目录路径，或摄像头ID (例如 `0` 代表默认摄像头)。
*   `--output_dir`: 检测结果的基础输出目录。默认: `runs/detect/exp_cli`。
*   `--conf_thres`: 置信度阈值 (0.0 到 1.0)。默认: 0.25。
*   `--iou_thres`: NMS (非极大值抑制) 的 IOU 阈值。默认: 0.45。
*   `--show_vid`: 是否实时显示处理结果窗口。
*   `--save_vid`: 是否保存处理后的视频或标注后的图片。
*   `--save_json`: 是否为每个输入图片/帧生成包含检测结果的 `.txt` 文件 (内容为JSON格式)。
*   `--target_class`: (可选) 指定要检测的目标类别名称 (例如 `tennis_ball`)。如果为空，则检测模型支持的所有类别。

运行 `python src/process.py --help` 查看所有可用参数。

### 3.2 标注数据转换 (`src/convert_annotations.py`)

用于将特定格式的原始标注数据 (当前实现为项目特定的JSON格式) 转换为YOLO TXT格式。

**命令示例**：
```bash
python src/convert_annotations.py \
    --json_file data/source_dataset/annotations.json \
    --images_dir data/source_dataset/original_images/ \
    --output_labels_dir data/processed_yolo_data/all_labels/ \
    --class_id 0
```
**主要参数说明**：
*   `--json_file`: (必需) 包含标注信息的JSON文件路径。
*   `--images_dir`: (必需) 包含原始图片的目录路径，用于获取图片尺寸。
*   `--output_labels_dir`: (必需) 输出YOLO TXT标签文件的目标目录。
*   `--class_id`: 分配给标注的类别ID (整数)。默认为 0。

运行 `python src/convert_annotations.py --help` 查看所有参数。

### 3.3 数据集划分 (`src/split_dataset.py`)

用于将YOLO格式的数据集（图片和对应的TXT标签）划分为训练集和验证集。

**命令示例**：
```bash
python src/split_dataset.py \
    --base_images_dir data/processed_yolo_data/images_for_splitting/ \
    --base_labels_dir data/processed_yolo_data/all_labels/ \
    --output_root_images data/processed_yolo_data/images \
    --output_root_labels data/processed_yolo_data/labels \
    --train_ratio 0.8 \
    --seed 42
```
**主要参数说明**：
*   `--base_images_dir`: (必需) 包含所有待划分图片的源目录。
*   `--base_labels_dir`: (必需) 包含所有对应YOLO TXT标签的源目录。
*   `--output_root_images`: (必需) 输出划分后图片的目标根目录 (脚本会在此下创建 `train/` 和 `val/` 子目录)。
*   `--output_root_labels`: (必需) 输出划分后标签的目标根目录 (脚本会在此下创建 `train/` 和 `val/` 子目录)。
*   `--train_ratio`: 训练集所占比例 (0.0 到 1.0)。默认为 0.8。
*   `--seed`: 随机种子，用于可复现的划分。默认为 42。

运行 `python src/split_dataset.py --help` 查看所有参数。

### 3.4 模型训练 (直接使用 `yolo` CLI)

如果您想完全自定义训练过程，可以直接使用 `ultralytics` 提供的 `yolo` 命令行工具进行训练。

**命令示例**：
```bash
# 确保虚拟环境已激活，且 dataset.yaml 已正确配置
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
请参考 Ultralytics YOLOv8 官方文档获取更多关于 `yolo train` 的参数和用法。

## 4. 关于输入与输出

*   **检测输入**: 
    *   使用 `run.sh` 时，默认从 `input/` 目录读取。
    *   使用 `src/process.py` 时，通过 `--source` 参数指定。
*   **检测输出**: 
    *   通常保存在 `results/` 目录下的子文件夹中。
    *   详细的输出文件结构（标注图片/视频、JSON数据文件）请参考 `OUTPUT_FORMAT_ZH.md` 或项目根 `README.md` 的"输出结果"部分。
*   **训练输出**: 
    *   保存在 `tennis_ball_runs/` 目录下，每个训练会有一个独立的子文件夹。

更多细节请参考项目根目录的 `README.md`。 